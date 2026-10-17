import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def verify_all_and_shuffle():
    out_file = OUTPUT_DIR / "07_verification_and_shuffle_audit.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        df = pd.read_csv(DATA_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df["Date_dt"].dt.day_name()
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
        df["HasCustomer"] = df["Customer ID"].notna()
        df["row_num"] = df.groupby("Transaction ID").cumcount() + 1
        df["tx_rows"] = df["Transaction ID"].map(df.groupby("Transaction ID")["Product"].count())

        log("=" * 95)
        log("COMPREHENSIVE VERIFICATION OF ALL CLAIMS + MONTE CARLO SHUFFLE / SYNTHETIC TESTS")
        log("=" * 95)

        # 1. Exact Duplicates Verification
        dups_second_only = df[df.duplicated(keep="first")]
        log("\n[1. EXACT DUPLICATES CHECK]")
        log(f"Count of extra exact duplicate rows: {len(dups_second_only)}")
        log(f"Revenue of extra exact duplicate rows: ${dups_second_only['Line Revenue (USD)'].sum():.2f}")
        log(f"Gross Profit of extra exact duplicate rows: ${dups_second_only['Line Gross Profit (USD)'].sum():.2f}")
        log("By branch:")
        log(dups_second_only["Branch"].value_counts().to_string())

        # Also check all 166 duplicate (Transaction ID, Product) pairs
        dup_tx_prod = df[df.duplicated(subset=["Transaction ID", "Product"], keep="first")]
        log(f"Total duplicate (Transaction ID, Product) extra rows: {len(dup_tx_prod)} (132 exact + {len(dup_tx_prod)-132} non-exact differing only in Qty/Disc)")

        # 2. Garden Cost Multiplier Verification
        log("\n[2. GARDEN UNIT COST & PRICE MULTIPLIER CHECK]")
        piv = df.pivot_table(index="Product", columns="Branch", values=["Unit Cost (USD)", "Unit Price (USD)"], aggfunc="first")
        c_g = piv[("Unit Cost (USD)", "Garden")]
        c_h = piv[("Unit Cost (USD)", "Harbour")]
        c_u = piv[("Unit Cost (USD)", "Uptown")]
        c_other_avg = (c_h + c_u) / 2.0
        ratio_vs_harbour = c_g / c_h
        ratio_vs_uptown = c_g / c_u
        ratio_vs_avg = c_g / c_other_avg
        log(f"Garden Unit Cost vs Harbour: mean={ratio_vs_harbour.mean():.4f}x (range {ratio_vs_harbour.min():.4f} to {ratio_vs_harbour.max():.4f}) [Note: Harbour = 0.70 * Garden]")
        log(f"Garden Unit Cost vs Uptown:  mean={ratio_vs_uptown.mean():.4f}x (range {ratio_vs_uptown.min():.4f} to {ratio_vs_uptown.max():.4f}) [Note: Uptown = 0.67 * Garden]")
        log(f"Garden Unit Cost vs Avg(Harbour, Uptown): mean={ratio_vs_avg.mean():.4f}x (range {ratio_vs_avg.min():.4f} to {ratio_vs_avg.max():.4f})")

        garden_df = df[df["Branch"] == "Garden"]
        excess_vs_h = (garden_df["Quantity"] * (garden_df["Unit Cost (USD)"] - garden_df["Product"].map(c_h))).sum()
        excess_vs_u = (garden_df["Quantity"] * (garden_df["Unit Cost (USD)"] - garden_df["Product"].map(c_u))).sum()
        excess_vs_avg = (garden_df["Quantity"] * (garden_df["Unit Cost (USD)"] - garden_df["Product"].map(c_other_avg))).sum()
        log(f"6-Month Excess Cost at Garden:")
        log(f"  - vs Harbour unit cost:          ${excess_vs_h:,.2f} (${excess_vs_h*2:,.2f}/yr)")
        log(f"  - vs Avg(Harbour, Uptown) cost:  ${excess_vs_avg:,.2f} (${excess_vs_avg*2:,.2f}/yr)")
        log(f"  - vs Uptown unit cost:           ${excess_vs_u:,.2f} (${excess_vs_u*2:,.2f}/yr)")

        # Wait: What if Garden's cost was meant to scale with Price just like Harbour and Uptown?
        # Look: Uptown_Price / Harbour_Price = 0.96 / 1.05 = 0.9143; Uptown_Cost / Harbour_Cost = 0.67 / 0.70 = 0.9571
        # Garden_Price is between Uptown (0.96) and Harbour (1.05), so Avg(Harbour, Uptown) price is 1.005 * Garden_Price!
        # That's why Avg(Harbour, Uptown) cost is the exact price-neutral benchmark for Garden ($16,666.62 in 6 months = $33,333/yr)!

        # 3. Exact Date Window & Rule of the 15% Discount at Harbour!
        log("\n[3. EXACT MECHANISM OF THE 15% DISCOUNT AT HARBOUR]")
        disc15 = df[df["Discount Rate"] == 0.15]
        log(f"Total 15% discount lines: {len(disc15)}")
        log(f"Branches with 15% discount: {disc15['Branch'].unique().tolist()}")
        log(f"Categories with 15% discount: {disc15['Category'].value_counts().to_dict()}")
        log("Exact Dates with 15% discount:")
        log(disc15.groupby("Date").size().to_string())

        # Check within Harbour during Mar 12-19 and May 12-19:
        in_win = df["Date"].between("2026-03-12", "2026-03-19") | df["Date"].between("2026-05-12", "2026-05-19")
        h_win = df[(df["Branch"] == "Harbour") & in_win]
        log("\nInside Harbour during Mar 12-19 & May 12-19, Discount Rate by Category:")
        log(pd.crosstab(h_win["Category"], h_win["Discount Rate"], margins=True).to_string())
        log(pd.crosstab(h_win["Category"], h_win["Discount Rate"], normalize="index").round(4).to_string())

        # Check Harbour traffic & Cold Coffee/Desserts volume during Mar 12-19 & May 12-19 vs other days!
        tx_first = df.drop_duplicates("Transaction ID").copy()
        tx_first["In15Win"] = tx_first["Date"].between("2026-03-12", "2026-03-19") | tx_first["Date"].between("2026-05-12", "2026-05-19")
        daily_branch_tx = tx_first.pivot_table(index="Date", columns="Branch", values="Transaction ID", aggfunc="count")
        daily_branch_tx["Harbour_to_Others"] = daily_branch_tx["Harbour"] / (daily_branch_tx["Garden"] + daily_branch_tx["Uptown"])
        daily_branch_tx["In15Win"] = daily_branch_tx.index.isin(disc15["Date"].unique())
        log("\nHarbour traffic ratio to (Garden + Uptown) inside vs outside 15% promo windows:")
        log(daily_branch_tx.groupby("In15Win")["Harbour_to_Others"].agg(["count", "mean", "std"]).to_string())

        # Also check Cold Coffee + Desserts share of Harbour lines inside vs outside the window!
        df["In15Win"] = in_win
        h_all = df[df["Branch"] == "Harbour"]
        log("\nHarbour Category Share (%) inside Mar 12-19 & May 12-19 vs all other days:")
        log((pd.crosstab(h_all["In15Win"], h_all["Category"], values=h_all["Quantity"], aggfunc="sum", normalize="index") * 100).round(2).to_string())

        # 4. Customer ID Monte Carlo Shuffle Test (20 shuffles)
        log("\n[4. MONTE CARLO SHUFFLE TEST FOR CUSTOMER ID (20 SHUFFLES)]")
        cust_tx = tx_first[tx_first["HasCustomer"]].copy().sort_values("Timestamp")
        # Observed metrics:
        def compute_cust_metrics(c_series, branch_arr, hour_arr, ot_arr, ts_arr):
            tmp = pd.DataFrame({"cid": c_series, "branch": branch_arr, "hour": hour_arr, "ot": ot_arr, "ts": ts_arr})
            # 1. Top branch share per customer
            br_counts = pd.crosstab(tmp["cid"], tmp["branch"])
            top_br_share = (br_counts.max(axis=1) / br_counts.sum(axis=1)).mean()
            # 2. All 3 branches visited share
            all3_share = ((br_counts > 0).sum(axis=1) == 3).mean()
            # 3. Single branch share
            one_br_share = ((br_counts > 0).sum(axis=1) == 1).mean()
            # 4. Hour std per customer
            hr_std = tmp.groupby("cid")["hour"].std().dropna().mean()
            # 5. Single order type share
            ot_counts = pd.crosstab(tmp["cid"], tmp["ot"])
            single_ot_share = ((ot_counts > 0).sum(axis=1) == 1).mean()
            # 6. Branch hops within 2 hours
            tmp_sorted = tmp.sort_values(["cid", "ts"])
            tmp_sorted["prev_cid"] = tmp_sorted["cid"].shift(1)
            tmp_sorted["prev_br"] = tmp_sorted["branch"].shift(1)
            tmp_sorted["prev_ts"] = tmp_sorted["ts"].shift(1)
            same_c = tmp_sorted["cid"] == tmp_sorted["prev_cid"]
            diff_b = tmp_sorted["branch"] != tmp_sorted["prev_br"]
            dt_hours = (tmp_sorted["ts"] - tmp_sorted["prev_ts"]).dt.total_seconds() / 3600.0
            hops_2h = (same_c & diff_b & (dt_hours <= 2.0)).sum()
            # 7. Same-day repeat customer-days
            tmp["date"] = tmp["ts"].dt.date
            c_days = tmp.groupby(["cid", "date"]).size()
            multi_visit_days = (c_days >= 2).sum()
            return {
                "top_branch_share": top_br_share,
                "all3_branches_pct": all3_share * 100,
                "single_branch_pct": one_br_share * 100,
                "hour_std": hr_std,
                "single_ot_pct": single_ot_share * 100,
                "branch_hops_2h": hops_2h,
                "multi_visit_cust_days": multi_visit_days
            }

        obs_metrics = compute_cust_metrics(
            cust_tx["Customer ID"].values,
            cust_tx["Branch"].values,
            cust_tx["Hour"].values,
            cust_tx["Order Type"].values,
            cust_tx["Timestamp"].values
        )
        rng = np.random.default_rng(42)
        shuf_list = []
        c_vals = cust_tx["Customer ID"].values.copy()
        for _ in range(20):
            c_shuf = rng.permutation(c_vals)
            shuf_list.append(compute_cust_metrics(
                c_shuf,
                cust_tx["Branch"].values,
                cust_tx["Hour"].values,
                cust_tx["Order Type"].values,
                cust_tx["Timestamp"].values
            ))
        shuf_df = pd.DataFrame(shuf_list)
        comp_df = pd.DataFrame({
            "Observed": pd.Series(obs_metrics),
            "Shuffle_Mean": shuf_df.mean(),
            "Shuffle_Std": shuf_df.std(),
            "Z_Score": (pd.Series(obs_metrics) - shuf_df.mean()) / shuf_df.std()
        })
        log("Customer ID Real vs 20x Random Permutation Test:")
        log(comp_df.round(4).to_string())

        # Also check why Summary 2 saw "days that start with takeaway have a second transaction 2.5% vs 1.6% after dine-in":
        # Wait: Takeaway is 54% in Morning/Daytime (07:00-17:59) and only 28% in Evening (18:00-23:59)!
        # If two random transactions happen on the same day, the EARLIER transaction is naturally in the Morning/Daytime (hence 54% Takeaway!)
        # while a single transaction on a day is 44% in the Evening (where Takeaway is 28%)!
        # Let's prove that in our shuffle test! In a pure random shuffle, what is the repeat rate of days starting with Takeaway vs Dine-in?!
        cust_tx["Date_only"] = cust_tx["Timestamp"].dt.date
        first_of_day = cust_tx.sort_values("Timestamp").drop_duplicates(["Customer ID", "Date_only"])
        day_counts = cust_tx.groupby(["Customer ID", "Date_only"]).size().rename("n_tx_day")
        first_of_day = first_of_day.merge(day_counts, on=["Customer ID", "Date_only"])
        first_of_day["is_repeat_day"] = first_of_day["n_tx_day"] >= 2
        log("\n[5. EXPLAINING SUMMARY 2'S 'SAME-DAY REPEAT' ARTIFACT!]")
        log("Observed repeat-day rate by first transaction's Order Type:")
        log((first_of_day.groupby("Order Type")["is_repeat_day"].mean() * 100).round(3).to_string())
        log("Why? Because the first transaction of a multi-visit day occurs earlier in the day (avg Hour = "
            f"{first_of_day[first_of_day['is_repeat_day']]['Hour'].mean():.2f} vs single-visit day avg Hour = "
            f"{first_of_day[~first_of_day['is_repeat_day']]['Hour'].mean():.2f}), and Takeaway is 54% before 18:00 vs 28% after 18:00!")

        # 6. Check Basket Pairing Lift Controlling for Time-of-Day AND Branch (Fixing Summary 1's Caveat!)
        log("\n[6. BASKET PAIRING TEST: RAW LIFT VS TIME-OF-DAY & BRANCH CONDITIONED SHUFFLE]")
        # Wait: earlier we found that Row 1 -> Row 2 has DIRECTED rules, AND also let's check unordered pairs vs shuffled within (Branch, Hour)!
        multi_df = df[df["tx_rows"] >= 2].copy()
        # Let's shuffle Category across rows WITHIN the same (Branch, Hour) so time-of-day and branch popularity are 100% preserved!
        def get_pair_counts(d):
            tx_c = d.groupby("Transaction ID")["Category"].apply(list)
            pairs = {}
            for cats in tx_c:
                # unordered distinct category pairs or same-category pair in 2+ item baskets
                if len(cats) >= 2:
                    for i in range(len(cats)):
                        for j in range(i+1, len(cats)):
                            p = tuple(sorted([cats[i], cats[j]]))
                            pairs[p] = pairs.get(p, 0) + 1
            return pd.Series(pairs)

        obs_pairs = get_pair_counts(multi_df)
        shuf_pairs_list = []
        for _ in range(50):
            m_copy = multi_df.copy()
            # Permute Category strictly within each (Branch, Hour) stratum!
            m_copy["Category"] = m_copy.groupby(["Branch", "Hour"])["Category"].transform(lambda x: rng.permutation(x.values))
            shuf_pairs_list.append(get_pair_counts(m_copy))
        shuf_pairs_df = pd.DataFrame(shuf_pairs_list).fillna(0)
        pair_comp = pd.DataFrame({
            "Observed": obs_pairs,
            "Expected_Given_Branch_Hour": shuf_pairs_df.mean(),
            "Std": shuf_pairs_df.std()
        })
        pair_comp["Lift_Given_Branch_Hour"] = pair_comp["Observed"] / pair_comp["Expected_Given_Branch_Hour"]
        pair_comp["Z_Score"] = (pair_comp["Observed"] - pair_comp["Expected_Given_Branch_Hour"]) / pair_comp["Std"]
        log("Top & Bottom Category Pairs vs Null Model Conditioned on (Branch, Hour) [50 Monte Carlo shuffles]:")
        log(pair_comp.sort_values("Z_Score", ascending=False).round(2).to_string())

if __name__ == "__main__":
    verify_all_and_shuffle()

