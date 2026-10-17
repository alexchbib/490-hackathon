import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

def run_all_models_and_plots():
    out_file = OUTPUT_DIR / "08_models_and_sensitivity_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        log("=" * 100)
        log("PHASE 5 & 6: OUT-OF-SAMPLE MODEL BENCHMARKS (JUNE 2026 TEST SET), NEGATIVE CONTROL & SENSITIVITY")
        log("=" * 100)

        df = pd.read_csv(DATA_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df["Date_dt"].dt.day_name()
        df["DayOfWeek_num"] = df["Date_dt"].dt.dayofweek
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        df["TimeBlock"] = pd.cut(df["Hour"], bins=[0, 10, 18, 24], labels=["Morning_7_10", "Daytime_11_18", "Evening_19_23"])
        df["HasCustomer"] = df["Customer ID"].notna()
        df["row_num"] = df.groupby("Transaction ID").cumcount() + 1
        df["tx_rows"] = df["Transaction ID"].map(df.groupby("Transaction ID")["Product"].count())

        # =========================================================================================
        # TRACK 1: NEXT-ITEM BASKET UPSELL CLASSIFIER & MARGIN-AWARE PROFIT OPTIMIZER
        # Train: Jan-May 2026 (Months 1-5) | Test: June 2026 (Month 6)
        # =========================================================================================
        log("\n" + "=" * 100)
        log("[MODEL TRACK 1: NEXT-ITEM BASKET UPSELL CLASSIFIER & PROFIT OPTIMIZER (TRAIN: JAN-MAY -> TEST: JUNE)]")
        log("=" * 100)

        multi_tx_ids = df[df["tx_rows"] >= 2]["Transaction ID"].unique()
        r1 = df[(df["Transaction ID"].isin(multi_tx_ids)) & (df["row_num"] == 1)].set_index("Transaction ID")
        r2 = df[(df["Transaction ID"].isin(multi_tx_ids)) & (df["row_num"] == 2)].set_index("Transaction ID")

        baskets = pd.DataFrame({
            "Month": r1["Month"],
            "Branch": r1["Branch"],
            "Hour": r1["Hour"],
            "TimeBlock": r1["TimeBlock"].astype(str),
            "DayOfWeek": r1["DayOfWeek"],
            "OrderType": r1["Order Type"],
            "HasCustomer": r1["HasCustomer"].astype(int),
            "Cat_0": r1["Category"],
            "Prod_0": r1["Product"],
            "Cat_1": r2["Category"],
            "Prod_1": r2["Product"],
            "GP_1": r2["Line Gross Profit (USD)"],
            "Rev_1": r2["Line Revenue (USD)"]
        })

        train_b = baskets[baskets["Month"] < "2026-06"].copy()
        test_b = baskets[baskets["Month"] == "2026-06"].copy()
        cats = sorted(baskets["Cat_1"].unique())
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        y_train = train_b["Cat_1"].map(cat_to_idx).values
        y_test = test_b["Cat_1"].map(cat_to_idx).values

        log(f"Multi-item baskets — Train (Jan-May): {len(train_b):,} | Out-of-Sample Test (June): {len(test_b):,}")

        # Average Gross Profit per Category & Branch on Training Set (for profit-aware ranking)
        cat_branch_gp = df[df["Month"] < "2026-06"].groupby(["Branch", "Category"])["Line Gross Profit (USD)"].mean().unstack("Category")[cats]
        log("\nHistorical Mean Gross Profit per Line Item ($) by Branch & Category (Jan-May Train):")
        log(cat_branch_gp.round(2).to_string())

        # Helper to evaluate probability predictions
        def eval_upsell_model(name, probs_test, use_profit_ranking=False):
            ll = log_loss(y_test, probs_test, labels=list(range(len(cats))))
            if not use_profit_ranking:
                top1_idx = np.argmax(probs_test, axis=1)
                top2_idx = np.argsort(probs_test, axis=1)[:, -2:]
            else:
                # Expected Gross Profit E[GP] = P(Cat_1 | Context) * Mean_GP(Branch, Cat_1)
                branch_gp_mat = test_b["Branch"].map(lambda b: cat_branch_gp.loc[b].values)
                branch_gp_mat = np.vstack(branch_gp_mat.values)
                expected_gp_mat = probs_test * branch_gp_mat
                top1_idx = np.argmax(expected_gp_mat, axis=1)
                top2_idx = np.argsort(expected_gp_mat, axis=1)[:, -2:]

            top1_acc = (top1_idx == y_test).mean() * 100
            top2_acc = np.array([y_test[i] in top2_idx[i] for i in range(len(y_test))]).mean() * 100

            # Realized Gross Profit per transaction when Top-1 or Top-2 recommendation matches customer's actual 2nd item
            top1_hit = (top1_idx == y_test)
            top2_hit = np.array([y_test[i] in top2_idx[i] for i in range(len(y_test))])
            realized_gp_top1 = (test_b["GP_1"].values * top1_hit).mean()
            realized_gp_top2 = (test_b["GP_1"].values * top2_hit).mean()
            avg_gp_when_hit_top1 = test_b["GP_1"].values[top1_hit].mean() if top1_hit.sum() > 0 else 0.0
            return {
                "Model": name,
                "Log_Loss": ll,
                "Top1_Acc_%": top1_acc,
                "Top2_Acc_%": top2_acc,
                "Realized_GP_Top1_$/tx": realized_gp_top1,
                "Realized_GP_Top2_$/tx": realized_gp_top2,
                "Mean_GP_per_Top1_Hit_$": avg_gp_when_hit_top1
            }

        results_t1 = []

        # Baseline 0: Marginal popularity P(Cat_1)
        marg_probs = train_b["Cat_1"].value_counts(normalize=True).reindex(cats).values
        probs_b0 = np.tile(marg_probs, (len(test_b), 1))
        results_t1.append(eval_upsell_model("Baseline 0: Marginal Popularity Only (Always Bakery/Hot Coffee)", probs_b0))

        # Simple Baseline 1: 9x9 Transition Table P(Cat_1 | Cat_0)
        trans_mat = pd.crosstab(train_b["Cat_0"], train_b["Cat_1"], normalize="index").reindex(index=cats, columns=cats).fillna(1.0/len(cats))
        probs_b1 = np.vstack([trans_mat.loc[c].values for c in test_b["Cat_0"]])
        results_t1.append(eval_upsell_model("Baseline 1: 9x9 Conditional Lookup P(Cat_1 | Cat_0)", probs_b1))

        # Model A: Regularized Logistic Regression (Cat_0 + TimeBlock + Branch)
        enc_a = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_train_a = enc_a.fit_transform(train_b[["Cat_0", "TimeBlock", "Branch"]])
        X_test_a = enc_a.transform(test_b[["Cat_0", "TimeBlock", "Branch"]])
        clf_a = LogisticRegression(C=0.05, max_iter=500, random_state=42)
        clf_a.fit(X_train_a, y_train)
        probs_a = clf_a.predict_proba(X_test_a)
        results_t1.append(eval_upsell_model("Model A1: Parsimonious Logistic (Cat_0 + TimeBlock + Branch)", probs_a))

        # Model A2 (Upgrade 1): Profit-Aware Decision Rule using Model A1 probabilities * Branch/Category Margin
        results_t1.append(eval_upsell_model("Model A2: Margin-Aware Profit Optimizer (max P * Branch_GP)", probs_a, use_profit_ranking=True))

        # Model B: Complex Gradient Boosting (Prod_0 + Cat_0 + Hour + DayOfWeek + Branch + OrderType + HasCustomer)
        enc_b = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        feat_cols_b = ["Prod_0", "Cat_0", "Hour", "DayOfWeek", "Branch", "OrderType", "HasCustomer"]
        X_train_b = enc_b.fit_transform(train_b[feat_cols_b])
        X_test_b = enc_b.transform(test_b[feat_cols_b])
        clf_b = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=6, random_state=42)
        clf_b.fit(X_train_b, y_train)
        probs_b = clf_b.predict_proba(X_test_b)
        results_t1.append(eval_upsell_model("Model B: Complex Gradient Boosting (All Features: Prod_0+Hour+...)", probs_b))

        t1_df = pd.DataFrame(results_t1)
        log("\nOut-of-Sample Evaluation on Unseen June 2026 Test Set (3,210 Multi-Item Baskets):")
        log(t1_df.round(4).to_string(index=False))

        # Show what the Margin-Aware Optimizer recommends vs Popularity Lookup for each Anchor Category!
        log("\nRecommended POS Upsell Prompt Table by Anchor Category (Row 1) & Time Block:")
        rec_rows = []
        for c0 in cats:
            p_vec = trans_mat.loc[c0].values
            top2_prob_cats = [cats[i] for i in np.argsort(p_vec)[-2:][::-1]]
            # Expected GP at Harbour vs Garden
            gp_h = p_vec * cat_branch_gp.loc["Harbour"].values
            gp_g = p_vec * cat_branch_gp.loc["Garden"].values
            best_gp_cat_h = cats[np.argmax(gp_h)]
            best_gp_cat_g = cats[np.argmax(gp_g)]
            rec_rows.append({
                "Anchor_Category (Row 1)": c0,
                "Top_1_Prob_AddOn": f"{top2_prob_cats[0]} ({p_vec[cat_to_idx[top2_prob_cats[0]]]*100:.1f}%)",
                "Top_2_Prob_AddOn": f"{top2_prob_cats[1]} ({p_vec[cat_to_idx[top2_prob_cats[1]]]*100:.1f}%)",
                "Profit_Optimal_AddOn (Harbour)": f"{best_gp_cat_h} (E[GP]=${gp_h.max():.2f})",
                "Profit_Optimal_AddOn (Garden)": f"{best_gp_cat_g} (E[GP]=${gp_g.max():.2f})"
            })
        log(pd.DataFrame(rec_rows).to_string(index=False))

        # =========================================================================================
        # TRACK 2: DAILY BRANCH ORDER VOLUME FORECASTER (TRAIN: JAN-MAY -> TEST: JUNE)
        # =========================================================================================
        log("\n" + "=" * 100)
        log("[MODEL TRACK 2: DAILY BRANCH ORDER VOLUME FORECASTER (TRAIN: JAN-MAY 453 DAYS -> TEST: JUNE 90 DAYS)]")
        log("=" * 100)

        tx_first = df.drop_duplicates("Transaction ID").copy()
        daily_br = tx_first.groupby(["Date_dt", "Date", "Month", "DayOfWeek", "DayOfWeek_num", "Branch"]).size().reset_index(name="Orders")
        daily_br = daily_br.sort_values(["Branch", "Date_dt"]).reset_index(drop=True)
        daily_br["DayIndex"] = (daily_br["Date_dt"] - daily_br["Date_dt"].min()).dt.days + 1
        daily_br["Lag_7"] = daily_br.groupby("Branch")["Orders"].shift(7)
        daily_br["Roll_28"] = daily_br.groupby("Branch")["Orders"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())

        train_d = daily_br[daily_br["Month"] < "2026-06"].copy()
        test_d = daily_br[daily_br["Month"] == "2026-06"].copy()

        def eval_forecast(name, y_true, y_pred, is_tuesday_mask):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
            r2 = r2_score(y_true, y_pred)
            tue_bias = (y_pred[is_tuesday_mask] - y_true[is_tuesday_mask]).mean()
            tue_mae = mean_absolute_error(y_true[is_tuesday_mask], y_pred[is_tuesday_mask])
            return {
                "Forecasting_Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE_%": mape,
                "Out_of_Sample_R2": r2,
                "Tuesday_MAE": tue_mae,
                "Tuesday_Mean_Bias": tue_bias
            }

        results_t2 = []
        y_true_d = test_d["Orders"].values
        tue_mask = (test_d["DayOfWeek"] == "Tuesday").values

        # Baseline 1: Historical Branch Mean (Jan-May)
        br_means = train_d.groupby("Branch")["Orders"].mean()
        pred_d_b1 = test_d["Branch"].map(br_means).values
        results_t2.append(eval_forecast("Baseline 1: Historical Branch Average (Jan-May Mean)", y_true_d, pred_d_b1, tue_mask))

        # Baseline 2: 28-Day Rolling Branch Average (Frozen at end of May for June forecast, or rolling 28d)
        # Let's compute both 1-step rolling 28-day and seasonal naive Lag-7
        pred_d_b2 = test_d["Roll_28"].values
        results_t2.append(eval_forecast("Baseline 2: 28-Day Rolling Branch Average", y_true_d, pred_d_b2, tue_mask))

        # Baseline 3: Seasonal Naive (Lag-7 Same Weekday Last Week)
        pred_d_b3 = test_d["Lag_7"].values
        results_t2.append(eval_forecast("Baseline 3: Seasonal Naive (Lag-7 Same Weekday Last Week)", y_true_d, pred_d_b3, tue_mask))

        # Model A: Additive Linear Regression (DayIndex + DayOfWeek + Branch)
        enc_d = OneHotEncoder(drop="first", sparse_output=False)
        X_tr_d = np.hstack([train_d[["DayIndex"]].values, enc_d.fit_transform(train_d[["DayOfWeek", "Branch"]])])
        X_te_d = np.hstack([test_d[["DayIndex"]].values, enc_d.transform(test_d[["DayOfWeek", "Branch"]])])
        lin_reg = LinearRegression().fit(X_tr_d, train_d["Orders"].values)
        pred_d_ma = lin_reg.predict(X_te_d)
        results_t2.append(eval_forecast("Model A: Additive Linear Regression (Trend + DayOfWeek + Branch)", y_true_d, pred_d_ma, tue_mask))

        # Model B: Multiplicative Log-Linear GLM (log(Orders) ~ DayIndex + DayOfWeek + Branch)
        log_reg = LinearRegression().fit(X_tr_d, np.log(train_d["Orders"].values))
        # Smearing correction for log-normal retransformation
        resid_tr = np.log(train_d["Orders"].values) - log_reg.predict(X_tr_d)
        smear = np.exp(resid_tr).mean()
        pred_d_mb = np.exp(log_reg.predict(X_te_d)) * smear
        results_t2.append(eval_forecast("Model B: Multiplicative Log-Linear GLM (log(Orders) ~ Trend + DoW + Branch)", y_true_d, pred_d_mb, tue_mask))

        # Model C: Complex Gradient Boosted Trees (DayIndex + DayOfWeek + Branch + Lag_7 + Roll_28)
        train_d_clean = train_d.dropna(subset=["Lag_7", "Roll_28"]).copy()
        enc_dc = OneHotEncoder(sparse_output=False)
        X_tr_dc = np.hstack([train_d_clean[["DayIndex", "Lag_7", "Roll_28"]].values, enc_dc.fit_transform(train_d_clean[["DayOfWeek", "Branch"]])])
        X_te_dc = np.hstack([test_d[["DayIndex", "Lag_7", "Roll_28"]].values, enc_dc.transform(test_d[["DayOfWeek", "Branch"]])])
        hgb_reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=5, random_state=42)
        hgb_reg.fit(X_tr_dc, train_d_clean["Orders"].values)
        pred_d_mc = hgb_reg.predict(X_te_dc)
        results_t2.append(eval_forecast("Model C: Complex Gradient Boosting (Trend + Lags + Rolling + DoW)", y_true_d, pred_d_mc, tue_mask))

        t2_df = pd.DataFrame(results_t2)
        log("\nOut-of-Sample Daily Branch Volume Forecast on June 2026 (90 Branch-Days):")
        log(t2_df.round(3).to_string(index=False))

        # =========================================================================================
        # TRACK 3 (UPGRADE 2): NEGATIVE-CONTROL CUSTOMER ID MODEL (EMPIRICAL PROOF OF NOISE)
        # Train on Customer History (Jan-May) -> Predict Customer Behavior in Unseen June 2026
        # =========================================================================================
        log("\n" + "=" * 100)
        log("[MODEL TRACK 3 (NEGATIVE CONTROL): CUSTOMER ID PERSONALIZATION / CHURN BENCHMARK ON JUNE 2026]")
        log("=" * 100)

        tx_totals = df.groupby("Transaction ID").agg(
            Month=("Month", "first"),
            Date_dt=("Date_dt", "first"),
            Branch=("Branch", "first"),
            OrderType=("Order Type", "first"),
            Hour=("Hour", "first"),
            CustomerID=("Customer ID", "first"),
            tx_rev=("Line Revenue (USD)", "sum"),
            tx_gp=("Line Gross Profit (USD)", "sum"),
            tx_qty=("Quantity", "sum"),
            AnchorCat=("Category", "first")
        ).reset_index()

        cust_train_tx = tx_totals[(tx_totals["Month"] < "2026-06") & (tx_totals["CustomerID"].notna())].copy()
        cust_test_tx = tx_totals[(tx_totals["Month"] == "2026-06") & (tx_totals["CustomerID"].notna())].copy()

        # Build RFM + Behavioral features per customer in Jan-May
        max_train_date = pd.to_datetime("2026-05-31")
        cust_feats = cust_train_tx.groupby("CustomerID").agg(
            hist_visits=("Transaction ID", "count"),
            recency_days=("Date_dt", lambda s: (max_train_date - s.max()).days),
            hist_mean_rev=("tx_rev", "mean"),
            hist_mean_qty=("tx_qty", "mean"),
            hist_mean_hour=("Hour", "mean"),
            hist_takeaway_pct=("OrderType", lambda s: (s == "Takeaway").mean()),
            fav_branch=("Branch", lambda s: s.mode().iloc[0]),
            fav_cat=("AnchorCat", lambda s: s.mode().iloc[0])
        ).reset_index()

        # Target 1: June Visit Count across all 3,491 Jan-May customers
        june_visits = cust_test_tx.groupby("CustomerID").size().rename("june_visits")
        june_mean_rev = cust_test_tx.groupby("CustomerID")["tx_rev"].mean().rename("june_mean_rev")
        cust_eval = cust_feats.merge(june_visits, on="CustomerID", how="left").merge(june_mean_rev, on="CustomerID", how="left")
        cust_eval["june_visits"] = cust_eval["june_visits"].fillna(0)

        # Split Jan-May customers into 70% fit / 30% eval to test predicting June visits & June basket spend from Jan-May RFM
        rng = np.random.default_rng(42)
        idx_perm = rng.permutation(len(cust_eval))
        cut = int(0.7 * len(cust_eval))
        ce_tr, ce_te = cust_eval.iloc[idx_perm[:cut]], cust_eval.iloc[idx_perm[cut:]]

        num_f = ["hist_visits", "recency_days", "hist_mean_rev", "hist_mean_qty", "hist_mean_hour", "hist_takeaway_pct"]
        # 3A: Predict June Visit Count
        base_v_pred = np.full(len(ce_te), ce_tr["june_visits"].mean())
        ridge_v = Ridge(alpha=10.0).fit(ce_tr[num_f], ce_tr["june_visits"])
        rf_v = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(ce_tr[num_f], ce_tr["june_visits"])
        log(f"Task 3A — Predicting Customer June Visit Count from Jan-May RFM History (n={len(ce_te)} test customers):")
        log(f"  - Simple Baseline (Global Mean = {ce_tr['june_visits'].mean():.2f} visits): RMSE = {np.sqrt(mean_squared_error(ce_te['june_visits'], base_v_pred)):.4f} | R^2 = {r2_score(ce_te['june_visits'], base_v_pred):.4f}")
        log(f"  - Ridge Regression on RFM Features:                 RMSE = {np.sqrt(mean_squared_error(ce_te['june_visits'], ridge_v.predict(ce_te[num_f]))):.4f} | R^2 = {r2_score(ce_te['june_visits'], ridge_v.predict(ce_te[num_f])):.4f}")
        log(f"  - Random Forest on RFM Features:                    RMSE = {np.sqrt(mean_squared_error(ce_te['june_visits'], rf_v.predict(ce_te[num_f]))):.4f} | R^2 = {r2_score(ce_te['june_visits'], rf_v.predict(ce_te[num_f])):.4f}")

        # 3B: Predict June Basket Spend for returning customers
        ce_tr_ret = ce_tr.dropna(subset=["june_mean_rev"])
        ce_te_ret = ce_te.dropna(subset=["june_mean_rev"])
        base_rev_pred = np.full(len(ce_te_ret), ce_tr_ret["june_mean_rev"].mean())
        ridge_rev = Ridge(alpha=10.0).fit(ce_tr_ret[num_f], ce_tr_ret["june_mean_rev"])
        log(f"\nTask 3B — Predicting Customer June Basket Spend ($) from Jan-May Spend & RFM (n={len(ce_te_ret)} returning customers):")
        log(f"  - Simple Baseline (Global Mean = ${ce_tr_ret['june_mean_rev'].mean():.2f}): RMSE = ${np.sqrt(mean_squared_error(ce_te_ret['june_mean_rev'], base_rev_pred)):.4f} | R^2 = {r2_score(ce_te_ret['june_mean_rev'], base_rev_pred):.4f}")
        log(f"  - Customer Historical Spend Lookup (Predict Hist Mean): RMSE = ${np.sqrt(mean_squared_error(ce_te_ret['june_mean_rev'], ce_te_ret['hist_mean_rev'])):.4f} | R^2 = {r2_score(ce_te_ret['june_mean_rev'], ce_te_ret['hist_mean_rev']):.4f}")
        log(f"  - Ridge Regression on RFM Features:                    RMSE = ${np.sqrt(mean_squared_error(ce_te_ret['june_mean_rev'], ridge_rev.predict(ce_te_ret[num_f]))):.4f} | R^2 = {r2_score(ce_te_ret['june_mean_rev'], ridge_rev.predict(ce_te_ret[num_f])):.4f}")

        # 3C: Predict which Branch and which Category a returning customer will choose in June!
        june_with_hist = cust_test_tx.merge(cust_feats[["CustomerID", "fav_branch", "fav_cat"]], on="CustomerID", how="inner")
        acc_branch_global = (june_with_hist["Branch"] == "Harbour").mean() * 100
        acc_branch_cust = (june_with_hist["Branch"] == june_with_hist["fav_branch"]).mean() * 100
        acc_cat_global = (june_with_hist["AnchorCat"] == "Bakery").mean() * 100
        acc_cat_cust = (june_with_hist["AnchorCat"] == june_with_hist["fav_cat"]).mean() * 100
        log(f"\nTask 3C — Predicting June Transaction Branch & Category from Customer's Jan-May Favorite (n={len(june_with_hist):,} June tx):")
        log(f"  - Branch Prediction:   Simple Baseline (Always 'Harbour') = {acc_branch_global:.2f}% vs Customer's Jan-May Favorite Branch = {acc_branch_cust:.2f}% (Lift = {acc_branch_cust - acc_branch_global:+.2f}%)")
        log(f"  - Category Prediction: Simple Baseline (Always 'Bakery')  = {acc_cat_global:.2f}% vs Customer's Jan-May Favorite Category = {acc_cat_cust:.2f}% (Lift = {acc_cat_cust - acc_cat_global:+.2f}%)")

        # =========================================================================================
        # SENSITIVITY & BREAK-EVEN TABLES (MEASURED VS ESTIMATED)
        # =========================================================================================
        log("\n" + "=" * 100)
        log("[SENSITIVITY & COUNTERFACTUAL VALUATION TABLES]")
        log("=" * 100)

        # 1. Garden Cost Gap Sensitivity
        log("\n1. Garden Branch Unit-Cost Recovery Sensitivity (Annualized = 2x 6-Month Volume):")
        base_annual_gap = 16666.62 * 2 # $33,333.25 vs avg of Harbour & Uptown
        for share_real in [0.0, 0.25, 0.50, 0.75, 1.00]:
            rec_50 = base_annual_gap * share_real * 0.50
            rec_100 = base_annual_gap * share_real * 1.00
            log(f"  - If {share_real*100:>5.0f}% of gap is real operational cost (not accounting allocation): "
                f"50% fix saves ${rec_50:>9,.2f}/yr | 100% fix saves ${rec_100:>9,.2f}/yr")

        # 2. Discount Elimination & Break-Even Elasticity
        log("\n2. Discount Removal Break-Even Demand Elasticity:")
        # For 15% promo at Harbour on Cold Coffee & Desserts:
        # Un-discounted margin on Harbour Cold Coffee & Desserts is 66.13% ($2,081.27 GP on $3,147.19 gross rev across 542 units = $3.84 GP/unit).
        # With 15% discount, GP was $1,609.72 ($2.97 GP/unit), losing $471.55 (22.66% of GP).
        # That means to break even on GP, a 15% discount MUST increase unit volume by: 0.15 / (0.6613 - 0.15) = +29.34%!
        # For 10% discount across all branches:
        # At Garden (46.94% un-discounted margin), a 10% discount cuts GP by 21.27%. To break even, volume must rise by: 0.10 / (0.4694 - 0.10) = +27.07%!
        # At Harbour (65.40% un-discounted margin), a 10% discount cuts GP by 15.30%. To break even, volume must rise by: 0.10 / (0.6540 - 0.10) = +18.05%!
        log("  - Harbour 15% Promo (Cold Coffee & Desserts, 66.1% base margin): Required volume lift to break even = +29.3% | Observed lift = -0.3% (Loss = $471.55 / 6mo)")
        log("  - Garden 10% Discount (46.9% base margin):                      Required volume lift to break even = +27.1% | Observed lift =  0.0% (Loss = $555.90 / 6mo)")
        log("  - Uptown 10% Discount (63.0% base margin):                      Required volume lift to break even = +18.9% | Observed lift =  0.0% (Loss = $322.15 / 6mo)")
        log("  - Harbour 10% Discount (65.4% base margin):                     Required volume lift to break even = +18.1% | Observed lift =  0.0% (Loss = $828.49 / 6mo)")

        # 3. POS Cross-Sell Prompt Sensitivity (18,175 single-item orders over 6 months = 36,350/yr)
        log("\n3. POS Contextual Add-On Prompt Sensitivity (Applied to 18,175 Single-Item Orders / 6mo = 36,350/yr):")
        # Average recommended add-on item has Price ~$5.20 and Gross Profit ~$3.40 (when using Margin-Aware ranking)
        for conv in [0.01, 0.02, 0.03, 0.05, 0.08]:
            added_tx_6m = 18175 * conv
            added_rev_yr = added_tx_6m * 2 * 5.20
            added_gp_yr = added_tx_6m * 2 * 3.40
            log(f"  - Conversion rate = {conv*100:>4.1f}% -> +{added_tx_6m*2:>6,.0f} add-on items/yr | Est. Annual Revenue = +${added_rev_yr:>8,.0f}/yr | Est. Annual Gross Profit = +${added_gp_yr:>8,.0f}/yr")

        # =========================================================================================
        # GENERATE 4 PUBLICATION-GRADE CHARTS IN figures/
        # =========================================================================================
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#333333"
        plt.rcParams["axes.linewidth"] = 0.8

        # FIGURE 1: Garden Cost Gap & Margin Erosion
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
        cat_order = ["Hot Coffee", "Cold Coffee", "Cold Drinks", "Matcha & Tea", "Bakery", "Breakfast", "Meals", "Desserts", "Sandwiches"]
        cat_rev_b = df.pivot_table(index="Category", columns="Branch", values="Line Revenue (USD)", aggfunc="sum").reindex(cat_order)
        cat_gp_b = df.pivot_table(index="Category", columns="Branch", values="Line Gross Profit (USD)", aggfunc="sum").reindex(cat_order)
        cat_mar_b = (cat_gp_b / cat_rev_b) * 100

        y_pos = np.arange(len(cat_order))
        h = 0.26
        axes[0].barh(y_pos + h, cat_mar_b["Harbour"], height=h, label="Harbour (Avg 65.0%)", color="#1b4965")
        axes[0].barh(y_pos, cat_mar_b["Uptown"], height=h, label="Uptown (Avg 62.6%)", color="#62b6cb")
        axes[0].barh(y_pos - h, cat_mar_b["Garden"], height=h, label="Garden (Avg 46.9%)", color="#e63946")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(cat_order, fontsize=9.5)
        axes[0].set_xlabel("Gross Margin (%)", fontsize=10, fontweight="bold")
        axes[0].set_title("A. Gross Margin (%) by Category & Branch\n(Garden lags by ~17 percentage points across all 9 categories)", fontsize=10.5, fontweight="bold")
        axes[0].set_xlim(0, 82)
        axes[0].axvline(46.91, color="#e63946", linestyle="--", alpha=0.6, linewidth=1)
        axes[0].axvline(64.96, color="#1b4965", linestyle="--", alpha=0.6, linewidth=1)
        axes[0].legend(loc="lower right", frameon=True, fontsize=8.5)
        axes[0].grid(axis="x", linestyle=":", alpha=0.5)

        # Panel B: Unit Cost Ratio across all 45 products
        piv_cost = df.pivot_table(index="Product", columns="Branch", values="Unit Cost (USD)", aggfunc="first")
        piv_price = df.pivot_table(index="Product", columns="Branch", values="Unit Price (USD)", aggfunc="first")
        other_cost_avg = (piv_cost["Harbour"] + piv_cost["Uptown"]) / 2.0
        other_price_avg = (piv_price["Harbour"] + piv_price["Uptown"]) / 2.0
        cost_mult = (piv_cost["Garden"] / other_cost_avg).sort_values()
        price_mult = (piv_price["Garden"] / other_price_avg).reindex(cost_mult.index)

        x_idx = np.arange(len(cost_mult))
        axes[1].scatter(x_idx, cost_mult.values, color="#e63946", s=28, label="Garden Unit Cost / Avg(Harbour, Uptown) [Mean = 1.460x]", zorder=3)
        axes[1].scatter(x_idx, price_mult.values, color="#1b4965", s=28, label="Garden Unit Price / Avg(Harbour, Uptown) [Mean = 0.995x]", zorder=3)
        axes[1].axhline(1.460, color="#e63946", linestyle="--", linewidth=1.2)
        axes[1].axhline(1.000, color="#1b4965", linestyle="--", linewidth=1.2)
        axes[1].set_ylim(0.85, 1.60)
        axes[1].set_xlabel("All 45 Menu Products (Sorted by Cost Multiplier)", fontsize=10, fontweight="bold")
        axes[1].set_ylabel("Ratio: Garden vs. Avg(Harbour, Uptown)", fontsize=10, fontweight="bold")
        axes[1].set_title("B. Uniform 1.46x Cost Multiplier Across All 45 Products\n($16.7k/6mo = $33.3k/yr Gap: Allocation vs. Supplier Issue)", fontsize=10, fontweight="bold")
        axes[1].legend(loc="center right", frameon=True, fontsize=8.5)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig1_garden_cost_anomaly.png", dpi=180)
        plt.close()

        # FIGURE 2: Harbour 15% Discount No-Lift Proof & Customer ID Shuffle Test
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=180)
        daily_b = tx_first.pivot_table(index="Date_dt", columns="Branch", values="Transaction ID", aggfunc="count")
        daily_b["Harbour_Ratio"] = daily_b["Harbour"] / (daily_b["Garden"] + daily_b["Uptown"])
        axes[0].plot(daily_b.index, daily_b["Harbour_Ratio"], color="#457b9d", linewidth=1.0, alpha=0.8, label="Daily Harbour / (Garden + Uptown) Traffic Ratio")
        axes[0].axhline(0.940, color="#1d3557", linestyle="--", linewidth=1.3, label="Non-Promo Mean (0.940)")
        axes[0].axvspan(pd.to_datetime("2026-03-12"), pd.to_datetime("2026-03-19"), color="#e63946", alpha=0.25, label="15% Promo Windows (Mar 12-19 & May 12-19; Mean=0.935)")
        axes[0].axvspan(pd.to_datetime("2026-05-12"), pd.to_datetime("2026-05-19"), color="#e63946", alpha=0.25)
        axes[0].set_title("A. Harbour 15% Promo Windows (100% Cold Coffee & Desserts)\nNo Traffic Lift (0.935 vs 0.940) & No Category Lift (22.55% vs 22.62%)", fontsize=9.8, fontweight="bold")
        axes[0].set_ylabel("Harbour / (Garden + Uptown) Tx Ratio", fontsize=9.5, fontweight="bold")
        axes[0].set_xlabel("Date (Jan - Jun 2026)", fontsize=9.5, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # Panel B: Customer ID Permutation Test Z-scores + Out-of-Sample Lift
        metrics_labels = [
            "Top-Branch Loyalty Share\n(0.595 obs vs 0.592 shuf)",
            "Single-Branch Customers %\n(6.21% obs vs 5.91% shuf)",
            "Same-Day Repeat Days\n(413 obs vs 391 shuf)",
            "Hour-of-Day Spread SD\n(4.39h obs vs 4.38h shuf)",
            "Cross-Branch Hops <2h\n(67 obs vs 65.8 shuf)",
            "Single Order-Type Share %\n(10.5% obs vs 10.7% shuf)",
            "Visited All 3 Branches %\n(55.9% obs vs 56.5% shuf)"
        ]
        z_vals = [0.86, 0.86, 0.85, 0.38, 0.12, -0.58, -0.80]
        y_p = np.arange(len(metrics_labels))
        axes[1].barh(y_p, z_vals, color="#2a9d8f", height=0.55)
        axes[1].axvline(0, color="#333333", linewidth=1)
        axes[1].axvline(-1.96, color="#e63946", linestyle="--", linewidth=1.2, label="95% Significance Threshold (±1.96σ)")
        axes[1].axvline(1.96, color="#e63946", linestyle="--", linewidth=1.2)
        axes[1].axvspan(-1.96, 1.96, color="#f1faee", alpha=0.5, zorder=0)
        axes[1].set_yticks(y_p)
        axes[1].set_yticklabels(metrics_labels, fontsize=8.5)
        axes[1].set_xlim(-3.0, 3.0)
        axes[1].set_xlabel("Z-Score vs. 20x Random Customer ID Shuffles", fontsize=9.5, fontweight="bold")
        axes[1].set_title("B. Forensic Proof: Customer ID Has Zero Signal\nAll Metrics Inside Noise (|Z| < 0.9σ); June Test R² ≤ 0", fontsize=9.8, fontweight="bold")
        axes[1].legend(loc="lower right", fontsize=8.5)
        axes[1].grid(axis="x", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig2_discount_and_customer_shuffle.png", dpi=180)
        plt.close()

        # FIGURE 3: Temporal Regimes, Tuesday Spike & 18:00 Dine-In Shift
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=180)
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily_tot = tx_first.groupby(["Date_dt", "DayOfWeek"]).size().reset_index(name="tx")
        dow_means = daily_tot.groupby("DayOfWeek")["tx"].mean().reindex(dow_order)
        colors_dow = ["#e63946" if d == "Tuesday" else "#457b9d" for d in dow_order]
        axes[0].bar(dow_short, dow_means.values, color=colors_dow, width=0.6)
        axes[0].axhline(dow_means.drop("Tuesday").mean(), color="#333333", linestyle="--", linewidth=1.1, label=f"Non-Tue Avg ({dow_means.drop('Tuesday').mean():.0f} tx/day)")
        axes[0].set_title("A. The Tuesday Spike (1.64x Other Days)\n22% of Weekly Revenue Across All Branches", fontsize=10, fontweight="bold")
        axes[0].set_ylabel("Avg Daily Transactions", fontsize=9.5, fontweight="bold")
        axes[0].legend(fontsize=8.5)
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)

        # Panel B: Hourly Volume & Takeaway vs Dine-in Shift at 18:00
        hr_counts = tx_first.groupby("Hour").size()
        hr_dinein = (tx_first.groupby("Hour")["Order Type"].apply(lambda s: (s == "Dine-in").mean()) * 100)
        ax_b2 = axes[1].twinx()
        axes[1].bar(hr_counts.index, hr_counts.values, color="#a8dadc", width=0.7, label="Total Transactions")
        ax_b2.plot(hr_dinein.index, hr_dinein.values, color="#e63946", marker="o", linewidth=2.0, markersize=4, label="Dine-In Share (%)")
        axes[1].axvline(17.5, color="#1d3557", linestyle="--", linewidth=1.2)
        axes[1].set_title("B. 21:00 Peak & 18:00 Dine-In Shift\nDine-In Jumps from 46% (Day) to 72% (Night)", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Hour of Day (07:00 - 23:00)", fontsize=9.5, fontweight="bold")
        axes[1].set_ylabel("Transactions (6 Months)", fontsize=9.5, fontweight="bold", color="#1d3557")
        ax_b2.set_ylabel("Dine-In Share (%)", fontsize=9.5, fontweight="bold", color="#e63946")
        ax_b2.set_ylim(20, 90)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)

        # Panel C: Daypart Category Shift (Morning 7-10, Daytime 11-18, Evening 19-23)
        tb_cat = pd.crosstab(df["TimeBlock"], df["Category"], normalize="index") * 100
        top_cats_plot = ["Bakery", "Hot Coffee", "Matcha & Tea", "Cold Coffee", "Desserts", "Breakfast"]
        x_tb = np.arange(3)
        width_tb = 0.13
        palette_c = ["#1d3557", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        for idx_c, c_name in enumerate(top_cats_plot):
            axes[2].bar(x_tb + (idx_c - 2.5)*width_tb, tb_cat[c_name].values, width=width_tb, label=c_name, color=palette_c[idx_c])
        axes[2].set_xticks(x_tb)
        axes[2].set_xticklabels(["Morning\n(07-10)", "Daytime\n(11-18)", "Evening\n(19-23)"], fontsize=9)
        axes[2].set_ylabel("Share of Items Sold (%)", fontsize=9.5, fontweight="bold")
        axes[2].set_title("C. Sharp Daypart Menu Shift (11:00 & 19:00)\nMorning Coffee/Bakery -> Evening Matcha/Desserts", fontsize=10, fontweight="bold")
        axes[2].legend(fontsize=7.5, ncol=2, loc="upper right")
        axes[2].grid(axis="y", linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig3_temporal_regimes_and_tuesday.png", dpi=180)
        plt.close()

        # FIGURE 4: Basket Transition Lift & Out-of-Sample June 2026 Model Benchmarks
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
        # Panel A: Directed Row 1 -> Row 2 Transition Matrix (%)
        full_trans = pd.crosstab(baskets["Cat_0"], baskets["Cat_1"], normalize="index").reindex(index=cats, columns=cats) * 100
        im = axes[0].imshow(full_trans.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=55)
        axes[0].set_xticks(np.arange(len(cats)))
        axes[0].set_yticks(np.arange(len(cats)))
        axes[0].set_xticklabels(cats, rotation=38, ha="right", fontsize=8.5)
        axes[0].set_yticklabels(cats, fontsize=8.5)
        axes[0].set_xlabel("Second Item Category (Row 2)", fontsize=9.5, fontweight="bold")
        axes[0].set_ylabel("First Anchor Item Category (Row 1)", fontsize=9.5, fontweight="bold")
        axes[0].set_title("A. Directed Basket Cross-Sell Rules P(Cat₂ | Cat₁) %\nMatcha->Bakery (53.5%), Bakery->Coffee (58.2%), Sandwiches->Cold Drinks (26.6%)", fontsize=9.5, fontweight="bold")
        for i in range(len(cats)):
            for j in range(len(cats)):
                val = full_trans.values[i, j]
                axes[0].text(j, i, f"{val:.1f}%", ha="center", va="center", color="white" if val > 28 else "#111111", fontsize=7.5)

        # Panel B: Out-of-Sample June 2026 Daily Forecast Comparison (Actual vs Log-Linear vs Rolling Avg at Harbour)
        h_test = test_d[test_d["Branch"] == "Harbour"].copy()
        h_mask = (test_d["Branch"] == "Harbour").values
        axes[1].plot(h_test["Date_dt"], h_test["Orders"], color="#1d3557", marker="o", linewidth=1.8, markersize=4, label="Actual June Orders (Harbour)")
        axes[1].plot(h_test["Date_dt"], pred_d_mb[h_mask], color="#2a9d8f", linestyle="--", linewidth=1.8, label=f"Multiplicative Log-Linear GLM (MAPE=10.2%, R²=0.941)")
        axes[1].plot(h_test["Date_dt"], pred_d_b2[h_mask], color="#e63946", linestyle=":", linewidth=1.6, label=f"28-Day Rolling Baseline (MAPE=20.9%, R²=0.607)")
        axes[1].set_title("B. Out-of-Sample June 2026 Forecast Benchmark (Harbour)\nLog-Linear GLM Captures Multiplicative Tuesday Peaks & Growth Trend", fontsize=9.5, fontweight="bold")
        axes[1].set_xlabel("Date in Unseen Test Set (June 2026)", fontsize=9.5, fontweight="bold")
        axes[1].set_ylabel("Daily Transactions", fontsize=9.5, fontweight="bold")
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].grid(True, linestyle=":", alpha=0.5)
        axes[1].tick_params(axis="x", rotation=25)

        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig4_basket_cross_sell_and_models.png", dpi=180)
        plt.close()

        log("\nSuccessfully generated all 4 figures in figures/ directory.")

if __name__ == "__main__":
    run_all_models_and_plots()

