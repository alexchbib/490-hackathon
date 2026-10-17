import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_exhaustive_verification():
    out_file = OUTPUT_DIR / "04_exhaustive_verification.txt"
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

        tx_rows_map = df.groupby("Transaction ID")["Product"].count()
        df["tx_rows"] = df["Transaction ID"].map(tx_rows_map)

        log("=" * 95)
        log("EXHAUSTIVE SIGNAL VERIFICATION & DEEP HYPOTHESIS TESTING")
        log("=" * 95)

        # 1. Basket Co-Occurrence & Row Transition Matrix
        log("\n[1. MULTI-ITEM BASKET TRANSITION MATRIX (ROW 1 -> ROW 2)]")
        tx2 = df[df["tx_rows"] >= 2].pivot(index="Transaction ID", columns="row_num", values="Category")
        ct_12 = pd.crosstab(tx2[1], tx2[2], normalize="index") * 100
        log("Row 1 Category -> Row 2 Category (% of Row 1):")
        log(ct_12.round(2).to_string())

        ct_12_counts = pd.crosstab(tx2[1], tx2[2], margins=True)
        log("\nRow 1 Category -> Row 2 Category (Counts):")
        log(ct_12_counts.to_string())

        # Compute Lift of Category pairs in all multi-item baskets
        log("\n[2. CATEGORY PAIR CO-OCCURRENCE LIFT ACROSS ALL MULTI-ITEM BASKETS]")
        tx_cats = df.groupby("Transaction ID")["Category"].apply(set)
        all_cats = sorted(df["Category"].unique())
        cat_probs = {c: tx_cats.apply(lambda s: c in s).mean() for c in all_cats}
        lift_mat = pd.DataFrame(index=all_cats, columns=all_cats, dtype=float)
        cooc_counts = pd.DataFrame(index=all_cats, columns=all_cats, dtype=int)
        for c1 in all_cats:
            for c2 in all_cats:
                both = tx_cats.apply(lambda s: (c1 in s) and (c2 in s)).mean()
                cnt = tx_cats.apply(lambda s: (c1 in s) and (c2 in s)).sum()
                cooc_counts.loc[c1, c2] = cnt
                lift_mat.loc[c1, c2] = both / (cat_probs[c1] * cat_probs[c2])
        log("Category Co-occurrence Lift Matrix (All 33,887 transactions):")
        log(lift_mat.round(3).to_string())

        # Conditional lift within multi-item transactions only
        multi_tx_cats = df[df["tx_rows"] >= 2].groupby("Transaction ID")["Category"].apply(set)
        m_cat_probs = {c: multi_tx_cats.apply(lambda s: c in s).mean() for c in all_cats}
        m_lift_mat = pd.DataFrame(index=all_cats, columns=all_cats, dtype=float)
        for c1 in all_cats:
            for c2 in all_cats:
                both = multi_tx_cats.apply(lambda s: (c1 in s) and (c2 in s)).mean()
                m_lift_mat.loc[c1, c2] = both / (m_cat_probs[c1] * m_cat_probs[c2])
        log("\nCategory Co-occurrence Lift Matrix (Within 15,712 Multi-Item Transactions):")
        log(m_lift_mat.round(3).to_string())

        # 3. Quantify the Exact Financial Loss of Discounts (10% and 15%)
        log("\n[3. FINANCIAL ANATOMY OF DISCOUNTS (10% AND 15%)]")
        df["Gross_Rev_No_Disc"] = df["Quantity"] * df["Unit Price (USD)"]
        df["Discount_Dollar_Loss"] = df["Gross_Rev_No_Disc"] - df["Line Revenue (USD)"]
        df["GP_No_Disc"] = df["Gross_Rev_No_Disc"] - (df["Quantity"] * df["Unit Cost (USD)"])

        disc_summary = df.groupby(["Branch", "Discount Rate"]).agg(
            lines=("Transaction ID", "count"),
            qty=("Quantity", "sum"),
            gross_rev=("Gross_Rev_No_Disc", "sum"),
            actual_rev=("Line Revenue (USD)", "sum"),
            disc_cost=("Discount_Dollar_Loss", "sum"),
            actual_gp=("Line Gross Profit (USD)", "sum"),
            counterfactual_gp=("GP_No_Disc", "sum")
        )
        disc_summary["margin_actual_pct"] = disc_summary["actual_gp"] / disc_summary["actual_rev"] * 100
        disc_summary["margin_no_disc_pct"] = disc_summary["counterfactual_gp"] / disc_summary["gross_rev"] * 100
        disc_summary["gp_erosion_pct"] = (disc_summary["disc_cost"] / disc_summary["counterfactual_gp"]) * 100
        log(disc_summary.round(2).to_string())

        log(f"\nTotal Dollar Margin Given Away in Discounts across dataset: ${df['Discount_Dollar_Loss'].sum():,.2f}")
        log(f"  - From 10% General Discounts (all branches): ${df[df['Discount Rate']==0.10]['Discount_Dollar_Loss'].sum():,.2f}")
        log(f"  - From 15% Harbour Mar/May Promo (Cold Coffee & Desserts): ${df[df['Discount Rate']==0.15]['Discount_Dollar_Loss'].sum():,.2f}")

        # Difference-in-Differences test for Harbour March & May 15% Promo on Cold Coffee & Desserts
        log("\n[4. DIFFERENCE-IN-DIFFERENCES (DiD) TEST: DID HARBOUR'S 15% PROMO LIFT VOLUME?]")
        # Compare Daily Quantity of (Cold Coffee + Desserts) at Harbour vs (Garden + Uptown) and vs Other Categories at Harbour
        # Across Promo Months (Mar, May) vs Non-Promo Months (Jan, Feb, Apr, Jun)
        df["IsPromoMonth"] = df["Month"].isin(["2026-03", "2026-05"])
        df["IsTargetCat"] = df["Category"].isin(["Cold Coffee", "Desserts"])

        # Share of target categories in Harbour vs Garden/Uptown across Promo vs Non-Promo months
        did_table = df.groupby(["Branch", "IsPromoMonth"]).apply(
            lambda g: pd.Series({
                "total_qty": g["Quantity"].sum(),
                "target_qty": g[g["IsTargetCat"]]["Quantity"].sum(),
                "target_qty_share_pct": g[g["IsTargetCat"]]["Quantity"].sum() / g["Quantity"].sum() * 100,
                "cold_coffee_share_pct": g[g["Category"]=="Cold Coffee"]["Quantity"].sum() / g["Quantity"].sum() * 100,
                "desserts_share_pct": g[g["Category"]=="Desserts"]["Quantity"].sum() / g["Quantity"].sum() * 100,
                "avg_tx_size_qty": g["Quantity"].sum() / g["Transaction ID"].nunique(),
                "target_gp_per_unit": g[g["IsTargetCat"]]["Line Gross Profit (USD)"].sum() / g[g["IsTargetCat"]]["Quantity"].sum()
            })
        )
        log("Promo Months (Mar & May) vs Non-Promo Months (Jan, Feb, Apr, Jun) by Branch:")
        log(did_table.round(4).to_string())

        # Statistical test (chi-square / two-proportion z-test) at Harbour: Target category share in Promo vs Non-Promo months
        harbour_only = df[df["Branch"] == "Harbour"]
        h_promo_target = harbour_only[harbour_only["IsPromoMonth"] & harbour_only["IsTargetCat"]]["Quantity"].sum()
        h_promo_other = harbour_only[harbour_only["IsPromoMonth"] & ~harbour_only["IsTargetCat"]]["Quantity"].sum()
        h_nonpromo_target = harbour_only[~harbour_only["IsPromoMonth"] & harbour_only["IsTargetCat"]]["Quantity"].sum()
        h_nonpromo_other = harbour_only[~harbour_only["IsPromoMonth"] & ~harbour_only["IsTargetCat"]]["Quantity"].sum()
        chi2, p_val, _, _ = stats.chi2_contingency([[h_promo_target, h_promo_other], [h_nonpromo_target, h_nonpromo_other]])
        log(f"\nHarbour Target Category Share (Cold Coffee + Desserts): Promo = {h_promo_target/(h_promo_target+h_promo_other)*100:.2f}% vs Non-Promo = {h_nonpromo_target/(h_nonpromo_target+h_nonpromo_other)*100:.2f}% (Chi2 p-value = {p_val:.4f})")

        # 5. Quantify the Garden Unit Cost Anomaly (+42.9% COGS inflation)
        log("\n[5. FINANCIAL ANATOMY OF GARDEN BRANCH COGS ANOMALY]")
        # Map Harbour Unit Cost and Uptown Unit Cost to every product
        harbour_costs = df[df["Branch"] == "Harbour"].groupby("Product")["Unit Cost (USD)"].first()
        garden_costs = df[df["Branch"] == "Garden"].groupby("Product")["Unit Cost (USD)"].first()
        uptown_costs = df[df["Branch"] == "Uptown"].groupby("Product")["Unit Cost (USD)"].first()
        harbour_prices = df[df["Branch"] == "Harbour"].groupby("Product")["Unit Price (USD)"].first()
        garden_prices = df[df["Branch"] == "Garden"].groupby("Product")["Unit Price (USD)"].first()
        uptown_prices = df[df["Branch"] == "Uptown"].groupby("Product")["Unit Price (USD)"].first()

        # Note: Notice that at Harbour, Cost / Garden_Price = ? Let's check Cost / Price ratios across all 3 branches!
        cost_price_df = pd.DataFrame({
            "Garden_Price": garden_prices,
            "Harbour_Price": harbour_prices,
            "Uptown_Price": uptown_prices,
            "Garden_Cost": garden_costs,
            "Harbour_Cost": harbour_costs,
            "Uptown_Cost": uptown_costs,
        })
        cost_price_df["Garden_COGS_pct"] = cost_price_df["Garden_Cost"] / cost_price_df["Garden_Price"] * 100
        cost_price_df["Harbour_COGS_pct"] = cost_price_df["Harbour_Cost"] / cost_price_df["Harbour_Price"] * 100
        cost_price_df["Uptown_COGS_pct"] = cost_price_df["Uptown_Cost"] / cost_price_df["Uptown_Price"] * 100
        cost_price_df["Ratio_HarbourCost_to_GardenCost"] = cost_price_df["Harbour_Cost"] / cost_price_df["Garden_Cost"]
        cost_price_df["Ratio_UptownCost_to_GardenCost"] = cost_price_df["Uptown_Cost"] / cost_price_df["Garden_Cost"]
        log("COGS % of Unit Price across Branches (First 15 products + Summary):")
        log(cost_price_df.head(15).round(3).to_string())
        log("\nSummary of Cost/Price Ratios across all 45 products:")
        log(cost_price_df.describe().round(4).to_string())

        # Calculate exact excess COGS paid by Garden compared to:
        # Benchmark A: Harbour's absolute Unit Cost (1.00x Harbour Cost instead of 1.4286x)
        # Benchmark B: Uptown's COGS-to-Price ratio (where Cost scales with Price: Garden_Price / Harbour_Price * Harbour_Cost = 0.9524 * Harbour_Cost)
        garden_rows = df[df["Branch"] == "Garden"].copy()
        garden_rows["Harbour_Unit_Cost"] = garden_rows["Product"].map(harbour_costs)
        garden_rows["Uptown_Unit_Cost"] = garden_rows["Product"].map(uptown_costs)
        # Notice Uptown Price is 0.96 * Garden Price, and Uptown Cost is 0.67 * Garden Cost (which is 0.957 * Harbour Cost)!
        # That means Harbour and Uptown have the EXACT SAME cost-to-revenue structure (~34.6% COGS at Harbour, ~37.0% COGS at Uptown),
        # whereas Garden has 52.8% COGS!
        total_garden_actual_cost = (garden_rows["Quantity"] * garden_rows["Unit Cost (USD)"]).sum()
        total_garden_at_harbour_cost = (garden_rows["Quantity"] * garden_rows["Harbour_Unit_Cost"]).sum()
        total_garden_at_uptown_cost = (garden_rows["Quantity"] * garden_rows["Uptown_Unit_Cost"]).sum()
        log(f"\nGarden Total Actual COGS (6 months): ${total_garden_actual_cost:,.2f}")
        log(f"Garden Total COGS at Harbour Unit Costs: ${total_garden_at_harbour_cost:,.2f} (Excess COGS = ${total_garden_actual_cost - total_garden_at_harbour_cost:,.2f})")
        log(f"Garden Total COGS at Uptown Unit Costs:  ${total_garden_at_uptown_cost:,.2f} (Excess COGS = ${total_garden_actual_cost - total_garden_at_uptown_cost:,.2f})")

        # Breakdown of Garden Excess COGS by Category
        garden_rows["Excess_COGS_vs_Harbour"] = garden_rows["Quantity"] * (garden_rows["Unit Cost (USD)"] - garden_rows["Harbour_Unit_Cost"])
        cat_excess = garden_rows.groupby("Category").agg(
            qty=("Quantity", "sum"),
            rev=("Line Revenue (USD)", "sum"),
            actual_gp=("Line Gross Profit (USD)", "sum"),
            excess_cogs=("Excess_COGS_vs_Harbour", "sum")
        )
        cat_excess["repaired_gp"] = cat_excess["actual_gp"] + cat_excess["excess_cogs"]
        cat_excess["actual_margin_pct"] = cat_excess["actual_gp"] / cat_excess["rev"] * 100
        cat_excess["repaired_margin_pct"] = cat_excess["repaired_gp"] / cat_excess["rev"] * 100
        cat_excess["share_of_excess_cogs_pct"] = cat_excess["excess_cogs"] / cat_excess["excess_cogs"].sum() * 100
        log("\nGarden Excess COGS Breakdown by Category (vs Harbour Unit Cost):")
        log(cat_excess.sort_values("excess_cogs", ascending=False).round(2).to_string())

if __name__ == "__main__":
    run_exhaustive_verification()

