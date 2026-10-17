import sys
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_forensic_audit():
    out_file = OUTPUT_DIR / "01_forensic_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        log("=" * 90)
        log("PHASE 1 & 2: FORENSIC DATA QUALITY AUDIT & DEEP PATTERN DISCOVERY")
        log("=" * 90)

        df = pd.read_csv(DATA_PATH)
        log(f"\n[1. BASIC SHAPE & SCHEMA]")
        log(f"Total rows (line items): {len(df):,}")
        log(f"Total columns: {len(df.columns)}")
        log("Columns & Dtypes:")
        for col in df.columns:
            null_cnt = df[col].isna().sum()
            null_pct = (null_cnt / len(df)) * 100
            nunique = df[col].nunique(dropna=True)
            log(f"  - {col:<25} | dtype: {str(df[col].dtype):<10} | nulls: {null_cnt:>6,} ({null_pct:>5.2f}%) | unique: {nunique:>7,}")

        # Exact duplicate rows
        exact_dups = df.duplicated().sum()
        tx_prod_dups = df.duplicated(subset=["Transaction ID", "Product"]).sum()
        log(f"\nExact duplicate rows: {exact_dups:,}")
        log(f"Duplicate (Transaction ID, Product) pairs: {tx_prod_dups:,}")
        if tx_prod_dups > 0:
            log("Sample duplicate (Transaction ID, Product) rows:")
            dup_sample = df[df.duplicated(subset=["Transaction ID", "Product"], keep=False)].sort_values(["Transaction ID", "Product"]).head(16)
            log(dup_sample.to_string())

        # Check intra-transaction consistency
        log(f"\n[2. INTRA-TRANSACTION CONSISTENCY CHECK]")
        tx_groups = df.groupby("Transaction ID")
        n_tx = len(tx_groups)
        log(f"Unique Transaction IDs: {n_tx:,}")
        for col in ["Date", "Time", "Branch", "Order Type"]:
            inconsistent = (tx_groups[col].nunique(dropna=False) > 1).sum()
            log(f"  - Transactions with multiple '{col}': {inconsistent:,}")
        cust_inconsistent = (tx_groups["Customer ID"].nunique(dropna=False) > 1).sum()
        log(f"  - Transactions with multiple 'Customer ID' (incl NaN): {cust_inconsistent:,}")

        # Check Transaction ID numbering sequence
        tx_nums = df["Transaction ID"].str.replace("SYN-", "", regex=False).astype(int)
        log(f"Transaction ID min: SYN-{tx_nums.min():07d}, max: SYN-{tx_nums.max():07d}, unique count: {tx_nums.nunique():,}")
        missing_tx_ids = set(range(tx_nums.min(), tx_nums.max() + 1)) - set(tx_nums.unique())
        log(f"Missing Transaction IDs in sequence: {len(missing_tx_ids):,}")

        # Accounting & Mathematical Integrity Check
        log(f"\n[3. ACCOUNTING & MATHEMATICAL INTEGRITY CHECK]")
        expected_rev = df["Quantity"] * df["Unit Price (USD)"] * (1.0 - df["Discount Rate"])
        rev_diff = (df["Line Revenue (USD)"] - expected_rev).abs()
        log(f"Max abs diff in Line Revenue vs Qty*Price*(1-Disc): {rev_diff.max():.6f}")
        log(f"Rows with Line Revenue diff > $0.01: {(rev_diff > 0.01).sum():,}")
        log(f"Rows with Line Revenue diff > $0.05: {(rev_diff > 0.05).sum():,}")

        expected_gp = df["Line Revenue (USD)"] - (df["Quantity"] * df["Unit Cost (USD)"])
        gp_diff = (df["Line Gross Profit (USD)"] - expected_gp).abs()
        log(f"Max abs diff in Line Gross Profit vs Rev - Qty*Cost: {gp_diff.max():.6f}")
        log(f"Rows with Line Gross Profit diff > $0.01: {(gp_diff > 0.01).sum():,}")
        log(f"Rows with Line Gross Profit diff > $0.05: {(gp_diff > 0.05).sum():,}")

        # Negative / zero / extreme values check
        num_cols = ["Quantity", "Unit Price (USD)", "Discount Rate", "Unit Cost (USD)", "Line Revenue (USD)", "Line Gross Profit (USD)"]
        log("\nNumerical Summary Statistics:")
        log(df[num_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

        neg_gp = df[df["Line Gross Profit (USD)"] <= 0]
        log(f"\nRows with Line Gross Profit <= 0: {len(neg_gp):,} ({len(neg_gp)/len(df)*100:.2f}%)")
        if len(neg_gp) > 0:
            log("Breakdown of rows with Line Gross Profit <= 0 by Branch & Product:")
            log(neg_gp.groupby(["Branch", "Category", "Product", "Discount Rate"]).agg(
                rows=("Transaction ID", "count"),
                avg_price=("Unit Price (USD)", "mean"),
                avg_cost=("Unit Cost (USD)", "mean"),
                total_gp=("Line Gross Profit (USD)", "sum")
            ).to_string())

        # Category & Product mapping
        log(f"\n[4. PRODUCT, CATEGORY, PRICING & COST MECHANICS]")
        prod_cat = df.groupby("Product")["Category"].nunique()
        log(f"Products mapped to >1 Category: {(prod_cat > 1).sum()}")

        # Pivot table of Price and Cost across Branches per Product
        p_pivot = df.pivot_table(index=["Category", "Product"], columns="Branch", values=["Unit Price (USD)", "Unit Cost (USD)"], aggfunc="first")
        p_pivot["Price_Harbour_vs_Garden"] = p_pivot[("Unit Price (USD)", "Harbour")] / p_pivot[("Unit Price (USD)", "Garden")]
        p_pivot["Price_Uptown_vs_Garden"] = p_pivot[("Unit Price (USD)", "Uptown")] / p_pivot[("Unit Price (USD)", "Garden")]
        p_pivot["Cost_Garden_vs_Harbour"] = p_pivot[("Unit Cost (USD)", "Garden")] / p_pivot[("Unit Cost (USD)", "Harbour")]
        p_pivot["Cost_Uptown_vs_Harbour"] = p_pivot[("Unit Cost (USD)", "Uptown")] / p_pivot[("Unit Cost (USD)", "Harbour")]
        log("\nExact Price & Cost Multipliers across Branches by Product:")
        log(p_pivot.to_string())

        # Check if price or cost EVER varies within (Branch, Product)
        bp_var = df.groupby(["Branch", "Product"]).agg(
            price_nunique=("Unit Price (USD)", "nunique"),
            cost_nunique=("Unit Cost (USD)", "nunique")
        )
        log(f"\nAny (Branch, Product) with >1 unique Price? {(bp_var['price_nunique'] > 1).sum()}")
        log(f"Any (Branch, Product) with >1 unique Cost? {(bp_var['cost_nunique'] > 1).sum()}")
        if (bp_var['price_nunique'] > 1).sum() > 0 or (bp_var['cost_nunique'] > 1).sum() > 0:
            log(bp_var[(bp_var['price_nunique'] > 1) | (bp_var['cost_nunique'] > 1)].to_string())

        # Overview by Branch
        log("\n[5. BRANCH-LEVEL PERFORMANCE OVERVIEW]")
        branch_overview = df.groupby("Branch").agg(
            line_items=("Transaction ID", "count"),
            transactions=("Transaction ID", "nunique"),
            unique_customers=("Customer ID", "nunique"),
            known_cust_rows=("Customer ID", lambda x: x.notna().sum()),
            total_qty=("Quantity", "sum"),
            total_rev=("Line Revenue (USD)", "sum"),
            total_gp=("Line Gross Profit (USD)", "sum"),
            avg_disc_rate=("Discount Rate", "mean"),
            disc_incidence=("Discount Rate", lambda x: (x > 0).mean() * 100)
        )
        branch_overview["known_cust_pct"] = (branch_overview["known_cust_rows"] / branch_overview["line_items"]) * 100
        branch_overview["margin_pct"] = (branch_overview["total_gp"] / branch_overview["total_rev"]) * 100
        branch_overview["avg_basket_rev"] = branch_overview["total_rev"] / branch_overview["transactions"]
        branch_overview["avg_basket_gp"] = branch_overview["total_gp"] / branch_overview["transactions"]
        branch_overview["items_per_tx"] = branch_overview["total_qty"] / branch_overview["transactions"]
        log(branch_overview.to_string())

if __name__ == "__main__":
    run_forensic_audit()

