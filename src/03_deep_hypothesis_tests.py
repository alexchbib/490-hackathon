import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_hypothesis_scan():
    out_file = OUTPUT_DIR / "03_hypothesis_scan_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        df = pd.read_csv(DATA_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
        df["DayOfYear"] = df["Date_dt"].dt.dayofyear
        df["DayOfWeek"] = df["Date_dt"].dt.day_name()
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        df["HasCustomer"] = df["Customer ID"].notna()

        tx_summary = df.groupby("Transaction ID").agg(
            tx_rows=("Product", "count"),
            tx_qty=("Quantity", "sum"),
            tx_rev=("Line Revenue (USD)", "sum"),
            tx_gp=("Line Gross Profit (USD)", "sum"),
            Date_dt=("Date_dt", "first"),
            Month=("Month", "first"),
            DayOfWeek=("DayOfWeek", "first"),
            Hour=("Hour", "first"),
            Branch=("Branch", "first"),
            OrderType=("Order Type", "first"),
            HasCustomer=("HasCustomer", "first"),
            CustomerID=("Customer ID", "first")
        ).reset_index()

        log("=" * 95)
        log("SYSTEMATIC INTERACTION & SIGNAL SCAN ACROSS ALL DIMENSIONS")
        log("=" * 95)

        # 1. Check Category x Hour of Day
        log("\n[1. CATEGORY SHARE (%) BY HOUR OF DAY]")
        cat_hour = pd.crosstab(df["Hour"], df["Category"], normalize="index") * 100
        log(cat_hour.round(2).to_string())

        # 2. Check Order Type x Hour of Day & DayOfWeek & Category
        log("\n[2. ORDER TYPE (TAKEAWAY %) BY HOUR, DAY OF WEEK, AND CATEGORY]")
        log("\nTakeaway % by Hour:")
        log((pd.crosstab(tx_summary["Hour"], tx_summary["OrderType"], normalize="index")["Takeaway"] * 100).round(2).to_string())
        log("\nTakeaway % by DayOfWeek:")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        log((pd.crosstab(tx_summary["DayOfWeek"], tx_summary["OrderType"], normalize="index").reindex(dow_order)["Takeaway"] * 100).round(2).to_string())
        log("\nTakeaway % by Category:")
        log((pd.crosstab(df["Category"], df["Order Type"], normalize="index")["Takeaway"] * 100).round(2).to_string())

        # 3. Check Quantity == 2 distribution across all dimensions
        log("\n[3. WHERE DOES QUANTITY == 2 OCCUR?]")
        df["IsQty2"] = df["Quantity"] == 2
        log(f"Overall Quantity==2 rate: {df['IsQty2'].mean()*100:.2f}%")
        log("\nQuantity==2 rate (%) by Branch:")
        log((df.groupby("Branch")["IsQty2"].mean() * 100).round(2).to_string())
        log("\nQuantity==2 rate (%) by Category:")
        log((df.groupby("Category")["IsQty2"].mean() * 100).round(2).to_string())
        log("\nQuantity==2 rate (%) by Order Type:")
        log((df.groupby("Order Type")["IsQty2"].mean() * 100).round(2).to_string())
        log("\nQuantity==2 rate (%) by Hour:")
        log((df.groupby("Hour")["IsQty2"].mean() * 100).round(2).to_string())
        log("\nQuantity==2 rate (%) by DayOfWeek:")
        log((df.groupby("DayOfWeek")["IsQty2"].mean().reindex(dow_order) * 100).round(2).to_string())

        # 4. Check Harbour 15% Discount in March & May in detail
        log("\n[4. DEEP DIVE: HARBOUR 15% DISCOUNT IN MARCH & MAY]")
        harbour_df = df[df["Branch"] == "Harbour"].copy()
        log("\nHarbour Discount Rate breakdown by Month and Category (Cold Coffee & Desserts vs Others):")
        harbour_df["CatGroup"] = harbour_df["Category"].where(harbour_df["Category"].isin(["Cold Coffee", "Desserts"]), "Other Categories")
        log(pd.crosstab([harbour_df["Month"], harbour_df["CatGroup"]], harbour_df["Discount Rate"], margins=True).to_string())
        log(pd.crosstab([harbour_df["Month"], harbour_df["CatGroup"]], harbour_df["Discount Rate"], normalize="index").round(4).to_string())

        # Check if Cold Coffee & Desserts share of quantity at Harbour changed in March & May!
        log("\nCategory Share of Quantity (%) by Month at HARBOUR:")
        log((pd.crosstab(harbour_df["Month"], harbour_df["Category"], values=harbour_df["Quantity"], aggfunc="sum", normalize="index") * 100).round(2).to_string())

        log("\nCategory Share of Quantity (%) by Month at GARDEN:")
        garden_df = df[df["Branch"] == "Garden"]
        log((pd.crosstab(garden_df["Month"], garden_df["Category"], values=garden_df["Quantity"], aggfunc="sum", normalize="index") * 100).round(2).to_string())

        log("\nCategory Share of Quantity (%) by Month at UPTOWN:")
        uptown_df = df[df["Branch"] == "Uptown"]
        log((pd.crosstab(uptown_df["Month"], uptown_df["Category"], values=uptown_df["Quantity"], aggfunc="sum", normalize="index") * 100).round(2).to_string())

        # 5. Check Product Shares within Category across Branches, Months, Hours
        log("\n[5. PRODUCT SHARE WITHIN CATEGORY BY BRANCH]")
        for cat, grp in df.groupby("Category"):
            log(f"\n--- Category: {cat} ---")
            ct = pd.crosstab(grp["Product"], grp["Branch"], values=grp["Quantity"], aggfunc="sum", normalize="columns") * 100
            log(ct.round(2).to_string())

        # 6. Check Daily Transaction Volume Trends & Outliers
        log("\n[6. DAILY TRANSACTION VOLUME & TEMPORAL STRUCTURE]")
        daily_tx = tx_summary.groupby(["Date_dt", "DayOfWeek", "Month"]).agg(
            tx_cnt=("Transaction ID", "count"),
            garden_tx=("Branch", lambda x: (x == "Garden").sum()),
            harbour_tx=("Branch", lambda x: (x == "Harbour").sum()),
            uptown_tx=("Branch", lambda x: (x == "Uptown").sum()),
        ).reset_index()
        log("Average Daily Transactions by Month and DayOfWeek:")
        log(daily_tx.pivot_table(index="Month", columns="DayOfWeek", values="tx_cnt", aggfunc="mean")[dow_order].round(1).to_string())

        log("\nAverage Daily Transactions by Branch and Month:")
        log(daily_tx.groupby("Month")[["garden_tx", "harbour_tx", "uptown_tx", "tx_cnt"]].mean().round(2).to_string())

        # Check if Branch share of transactions changes over Months or DayOfWeek or Hour
        log("\nBranch Share of Transactions (%) by Month:")
        log((pd.crosstab(tx_summary["Month"], tx_summary["Branch"], normalize="index") * 100).round(2).to_string())

        log("\nBranch Share of Transactions (%) by DayOfWeek:")
        log((pd.crosstab(tx_summary["DayOfWeek"], tx_summary["Branch"], normalize="index").reindex(dow_order) * 100).round(2).to_string())

        log("\nBranch Share of Transactions (%) by Hour:")
        log((pd.crosstab(tx_summary["Hour"], tx_summary["Branch"], normalize="index") * 100).round(2).to_string())

        # 7. Basket Co-occurrence (Do multi-item baskets have category affinity or independent sampling?)
        log("\n[7. BASKET SIZE & CO-OCCURRENCE MECHANICS]")
        log("Basket size (tx_rows) distribution (%) by Branch:")
        log((pd.crosstab(tx_summary["Branch"], tx_summary["tx_rows"], normalize="index") * 100).round(2).to_string())
        log("Basket size (tx_rows) distribution (%) by Hour:")
        log((pd.crosstab(tx_summary["Hour"], tx_summary["tx_rows"], normalize="index") * 100).round(2).to_string())
        log("Basket size (tx_rows) distribution (%) by OrderType:")
        log((pd.crosstab(tx_summary["OrderType"], tx_summary["tx_rows"], normalize="index") * 100).round(2).to_string())
        log("Basket size (tx_rows) distribution (%) by Month:")
        log((pd.crosstab(tx_summary["Month"], tx_summary["tx_rows"], normalize="index") * 100).round(2).to_string())

        # Check within 2-row and 3-row transactions: how often do two rows have the SAME Category or SAME Product?
        multi_df = df[df["Transaction ID"].isin(tx_summary[tx_summary["tx_rows"] > 1]["Transaction ID"])]
        tx_cat_check = multi_df.groupby("Transaction ID").agg(
            rows=("Product", "count"),
            nunique_prod=("Product", "nunique"),
            nunique_cat=("Category", "nunique")
        )
        log("\nIn multi-row transactions, how often are Products or Categories repeated?")
        log(pd.crosstab(tx_cat_check["rows"], tx_cat_check["nunique_prod"], margins=True).to_string())
        log(pd.crosstab(tx_cat_check["rows"], tx_cat_check["nunique_cat"], margins=True).to_string())

        # Check 1st row vs 2nd row vs 3rd row in multi-row transactions!
        df["row_num_in_tx"] = df.groupby("Transaction ID").cumcount() + 1
        df = df.merge(tx_summary[["Transaction ID", "tx_rows"]], on="Transaction ID", how="left")
        log("\nCategory distribution (%) by row position (row_num_in_tx) inside multi-item transactions:")
        log((pd.crosstab([df["tx_rows"], df["row_num_in_tx"]], df["Category"], normalize="index") * 100).round(2).to_string())

        log("\nDiscount Rate distribution (%) by row position (row_num_in_tx):")
        log((pd.crosstab([df["tx_rows"], df["row_num_in_tx"]], df["Discount Rate"], normalize="index") * 100).round(2).to_string())

        # 8. Check Customer ID patterns: Are there customer segments or is Customer ID uniform?
        log("\n[8. CUSTOMER ID FORENSIC CHECK]")
        cust_tx = tx_summary[tx_summary["HasCustomer"]].copy()
        cust_counts = cust_tx["CustomerID"].value_counts()
        log(f"Customer ID min: {cust_tx['CustomerID'].min()}, max: {cust_tx['CustomerID'].max()}, nunique: {cust_tx['CustomerID'].nunique()}")
        # Test if cust_counts follows a Poisson/Binomial distribution (i.e. uniform random assignment across 3500 IDs!)
        total_known_tx = len(cust_tx)
        n_total_possible_ids = 3500
        expected_lambda = total_known_tx / n_total_possible_ids
        log(f"If {total_known_tx} known transactions were randomly assigned across 3,500 IDs (C00001..C03500):")
        log(f"  Expected mean visits per ID: {expected_lambda:.4f}, expected std: {np.sqrt(expected_lambda):.4f}")
        log(f"  Expected IDs with 0 visits: {3500 * np.exp(-expected_lambda):.2f} (Actual missing IDs out of 3500: {3500 - cust_tx['CustomerID'].nunique()})")
        log(f"  Observed mean visits per seen ID: {cust_counts.mean():.4f}, observed std: {cust_counts.std():.4f}")

        # Also check if Customer ID correlates with Branch, OrderType, Hour, or Category
        cust_branch_chisq = stats.chi2_contingency(pd.crosstab(cust_tx["CustomerID"], cust_tx["Branch"]))
        log(f"Chi-square test CustomerID vs Branch: p-value = {cust_branch_chisq.pvalue:.6f}")
        cust_ot_chisq = stats.chi2_contingency(pd.crosstab(cust_tx["CustomerID"], cust_tx["OrderType"]))
        log(f"Chi-square test CustomerID vs OrderType: p-value = {cust_ot_chisq.pvalue:.6f}")

if __name__ == "__main__":
    run_hypothesis_scan()

