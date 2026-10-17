import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_micro_checks():
    out_file = OUTPUT_DIR / "05_micro_mechanics_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        df = pd.read_csv(DATA_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df["Date_dt"].dt.day_name()
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        df["HasCustomer"] = df["Customer ID"].notna()
        df["row_num"] = df.groupby("Transaction ID").cumcount() + 1
        df["tx_rows"] = df["Transaction ID"].map(df.groupby("Transaction ID")["Product"].count())

        log("=" * 95)
        log("MICRO-MECHANICS CHECK: 3-ITEM BASKETS, DAILY OUTLIERS, TIME-OF-DAY & PRODUCT TRANSITIONS")
        log("=" * 95)

        # 1. Check Row 1 -> Row 3 and Row 2 -> Row 3 in 3-item transactions
        tx3 = df[df["tx_rows"] == 3].pivot(index="Transaction ID", columns="row_num", values="Category")
        log("\n[1. THREE-ITEM BASKETS: ROW 1 -> ROW 3 AND ROW 2 -> ROW 3 (%)]")
        log("Row 1 -> Row 3 (%):")
        log((pd.crosstab(tx3[1], tx3[3], normalize="index") * 100).round(2).to_string())
        log("\nRow 2 -> Row 3 (%):")
        log((pd.crosstab(tx3[2], tx3[3], normalize="index") * 100).round(2).to_string())

        # 2. Check if Row 1 -> Row 2 transitions depend on Branch or Hour!
        tx2_df = df[df["tx_rows"] >= 2].pivot(index="Transaction ID", columns="row_num", values="Category")
        tx_meta = df.drop_duplicates("Transaction ID").set_index("Transaction ID")
        tx2_full = tx2_df.join(tx_meta[["Branch", "Hour", "Order Type", "HasCustomer", "Month"]])

        log("\n[2. DO BASKET TRANSITION RULES VARY BY BRANCH OR HOUR?]")
        for b in ["Garden", "Harbour", "Uptown"]:
            sub = tx2_full[tx2_full["Branch"] == b]
            mt_to_bak = (sub[sub[1] == "Matcha & Tea"][2] == "Bakery").mean() * 100
            bak_to_cof = (sub[sub[1] == "Bakery"][2].isin(["Hot Coffee", "Cold Coffee"])).mean() * 100
            sand_to_cd = (sub[sub[1] == "Sandwiches"][2] == "Cold Drinks").mean() * 100
            meal_to_cd = (sub[sub[1] == "Meals"][2] == "Cold Drinks").mean() * 100
            log(f"  Branch={b:<8} | Matcha->Bakery: {mt_to_bak:.1f}% | Bakery->Coffee: {bak_to_cof:.1f}% | Sandwiches->ColdDrinks: {sand_to_cd:.1f}% | Meals->ColdDrinks: {meal_to_cd:.1f}%")

        # 3. Check Daily Volume Model (R^2 of Month + DayOfWeek + Branch on daily transactions)
        log("\n[3. DAILY TRANSACTION VOLUME DECOMPOSITION & OUTLIER DAYS]")
        daily = tx_meta.groupby(["Date", "Month", "DayOfWeek"]).size().reset_index(name="tx_cnt")
        # Fit simple multiplicative / additive expected value from Month x DayOfWeek
        m_dow_mean = daily.groupby(["Month", "DayOfWeek"])["tx_cnt"].transform("mean")
        daily["expected"] = m_dow_mean
        daily["residual"] = daily["tx_cnt"] - daily["expected"]
        daily["rel_diff_pct"] = (daily["residual"] / daily["expected"]) * 100
        r2 = 1 - (daily["residual"]**2).sum() / ((daily["tx_cnt"] - daily["tx_cnt"].mean())**2).sum()
        log(f"R^2 of (Month x DayOfWeek) explaining daily transaction count across 181 days: {r2:.4f}")
        log("Top 10 Highest Positive Residual Days:")
        log(daily.sort_values("rel_diff_pct", ascending=False).head(10).to_string(index=False))
        log("Top 10 Lowest Negative Residual Days:")
        log(daily.sort_values("rel_diff_pct", ascending=True).head(10).to_string(index=False))

        # Check within-month trend (does volume grow day-by-day within a month, or step-change at month boundaries?)
        daily["DayOfMonth"] = pd.to_datetime(daily["Date"]).dt.day
        corr_dom, p_dom = stats.pearsonr(daily["DayOfMonth"], daily["residual"])
        log(f"\nCorrelation between DayOfMonth and residual (within-month growth vs step-change): r = {corr_dom:.4f}, p = {p_dom:.4f}")

        # 4. Check Discount Rate = 0.10 independence across all variables
        log("\n[4. IS THE 10% DISCOUNT COMPLETELY UNTARGETED (RANDOM ~5.6%)?]")
        # Exclude Harbour Mar/May Cold Coffee & Desserts where 0.15 applies
        non_promo15 = df[~((df["Branch"] == "Harbour") & (df["Month"].isin(["2026-03", "2026-05"])) & (df["Category"].isin(["Cold Coffee", "Desserts"])))].copy()
        non_promo15["IsDisc10"] = (non_promo15["Discount Rate"] == 0.10).astype(int)
        log(f"Overall 10% discount rate on non-15%-promo lines: {non_promo15['IsDisc10'].mean()*100:.3f}%")
        for col in ["Branch", "Order Type", "HasCustomer", "Category", "Month", "DayOfWeek", "Quantity", "tx_rows"]:
            ct = pd.crosstab(non_promo15[col], non_promo15["IsDisc10"])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            min_r = (ct[1] / ct.sum(axis=1) * 100).min()
            max_r = (ct[1] / ct.sum(axis=1) * 100).max()
            log(f"  - {col:<15} | range: {min_r:.2f}% to {max_r:.2f}% | Chi2 p-value = {p:.4f}")

        # Wait: Look at Category p-value for 10% discount! Earlier we saw Meals had 4.19% discount rate vs ~5.6% for others. Is that just random variation or something specific?
        log("\n10% Discount Rate by Category (on non-15%-promo lines):")
        cat_d10 = non_promo15.groupby("Category")["IsDisc10"].agg(["count", "sum", "mean"])
        cat_d10["mean_pct"] = cat_d10["mean"] * 100
        log(cat_d10.to_string())

        # 5. Check if Single-Item vs Multi-Item baskets differ by Category or Hour or Branch
        log("\n[5. SINGLE-ITEM VS MULTI-ITEM BASKET DRIVERS]")
        # In Row 1 of a transaction, does the probability of adding a 2nd item (tx_rows >= 2) depend on Row 1 Category?
        row1_df = df[df["row_num"] == 1].copy()
        row1_df["IsMultiItem"] = (row1_df["tx_rows"] >= 2).astype(int)
        r1_multi = row1_df.groupby("Category").agg(
            tx_count=("Transaction ID", "count"),
            multi_item_rate=("IsMultiItem", lambda x: x.mean() * 100),
            avg_tx_rows=("tx_rows", "mean"),
            avg_tx_rev=("Transaction ID", lambda s: 0) # placeholder
        )
        # Merge total tx_rev and tx_gp onto row1_df
        tx_totals = df.groupby("Transaction ID").agg(total_tx_rev=("Line Revenue (USD)", "sum"), total_tx_gp=("Line Gross Profit (USD)", "sum"))
        row1_df = row1_df.merge(tx_totals, on="Transaction ID", how="left")
        r1_summary = row1_df.groupby("Category").agg(
            first_item_txs=("Transaction ID", "count"),
            multi_item_pct=("IsMultiItem", lambda x: x.mean() * 100),
            avg_tx_rows=("tx_rows", "mean"),
            avg_first_item_gp=("Line Gross Profit (USD)", "mean"),
            avg_total_basket_rev=("total_tx_rev", "mean"),
            avg_total_basket_gp=("total_tx_gp", "mean"),
        )
        r1_summary["incremental_second_items_gp"] = r1_summary["avg_total_basket_gp"] - r1_summary["avg_first_item_gp"]
        log("Basket Expansion & Total Basket Gross Profit by Anchor (Row 1) Category:")
        log(r1_summary.sort_values("multi_item_pct", ascending=False).round(2).to_string())

if __name__ == "__main__":
    run_micro_checks()

