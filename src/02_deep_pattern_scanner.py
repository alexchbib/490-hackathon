import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_deep_scanner():
    out_file = OUTPUT_DIR / "02_deep_patterns_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        df = pd.read_csv(DATA_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
        df["Week"] = df["Date_dt"].dt.isocalendar().week
        df["DayOfWeek"] = df["Date_dt"].dt.day_name()
        df["IsWeekend"] = df["Date_dt"].dt.dayofweek.isin([5, 6])
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        df["HasCustomer"] = df["Customer ID"].notna()

        # Compute basket-level features and merge back
        tx_summary = df.groupby("Transaction ID").agg(
            tx_rows=("Product", "count"),
            tx_qty=("Quantity", "sum"),
            tx_unique_prods=("Product", "nunique"),
            tx_unique_cats=("Category", "nunique"),
            tx_rev=("Line Revenue (USD)", "sum"),
            tx_gp=("Line Gross Profit (USD)", "sum"),
            tx_disc_lines=("Discount Rate", lambda x: (x > 0).sum()),
            tx_max_disc=("Discount Rate", "max")
        ).reset_index()
        df = df.merge(tx_summary, on="Transaction ID", how="left")

        log("=" * 95)
        log("DEEP PATTERN SCANNER: DUPLICATES, DISCOUNTS, TIME REGIMES, MIX, CUSTOMERS & BASKETS")
        log("=" * 95)

        # 1. Inspect Duplicate (Transaction ID, Product) pairs
        log("\n[1. FORENSIC CHECK ON DUPLICATE (Transaction ID, Product) ROWS]")
        dup_mask = df.duplicated(subset=["Transaction ID", "Product"], keep=False)
        dups = df[dup_mask].sort_values(["Transaction ID", "Product"])
        log(f"Total rows involved in (Transaction ID, Product) duplicates: {len(dups)} across {dups['Transaction ID'].nunique()} transactions")
        exact_dup_mask = df.duplicated(keep=False)
        non_exact_dups = df[dup_mask & ~exact_dup_mask].sort_values(["Transaction ID", "Product"])
        log(f"Rows in (Transaction ID, Product) duplicates that are NOT exact duplicates: {len(non_exact_dups)}")
        if len(non_exact_dups) > 0:
            log("All non-exact (Transaction ID, Product) duplicate rows:")
            cols_to_show = ["Transaction ID", "Date", "Branch", "Customer ID", "Order Type", "Product", "Quantity", "Discount Rate", "Line Revenue (USD)"]
            log(non_exact_dups[cols_to_show].to_string())

        log("\nDistribution of (Transaction ID, Product) duplicates by Branch:")
        log(dups.groupby("Branch")["Transaction ID"].nunique().to_string())
        log("\nDistribution of (Transaction ID, Product) duplicates by tx_rows (basket size in rows):")
        log(dups.groupby("tx_rows")["Transaction ID"].nunique().to_string())

        # 2. Discount Mechanics
        log("\n" + "=" * 95)
        log("[2. DISCOUNT RATE MECHANICS (0.0, 0.10, 0.15)]")
        log("=" * 95)
        log("\nValue counts of Discount Rate:")
        log(df["Discount Rate"].value_counts(dropna=False).to_string())

        log("\nDiscount Rate cross-tab by HasCustomer (Known vs Anonymous Customer):")
        log(pd.crosstab(df["HasCustomer"], df["Discount Rate"], margins=True, normalize="index").to_string())
        log(pd.crosstab(df["HasCustomer"], df["Discount Rate"], margins=True).to_string())

        log("\nDiscount Rate cross-tab by Branch:")
        log(pd.crosstab(df["Branch"], df["Discount Rate"], margins=True, normalize="index").to_string())
        log(pd.crosstab(df["Branch"], df["Discount Rate"], margins=True).to_string())

        log("\nDiscount Rate cross-tab by Category:")
        log(pd.crosstab(df["Category"], df["Discount Rate"], margins=True, normalize="index").to_string())
        log(pd.crosstab(df["Category"], df["Discount Rate"], margins=True).to_string())

        log("\nDiscount Rate cross-tab by Order Type:")
        log(pd.crosstab(df["Order Type"], df["Discount Rate"], margins=True, normalize="index").to_string())

        log("\nDiscount Rate cross-tab by Hour of Day:")
        log(pd.crosstab(df["Hour"], df["Discount Rate"], normalize="index").to_string())

        log("\nDiscount Rate cross-tab by DayOfWeek:")
        log(pd.crosstab(df["DayOfWeek"], df["Discount Rate"], normalize="index").to_string())

        log("\nDiscount Rate cross-tab by Month:")
        log(pd.crosstab(df["Month"], df["Discount Rate"], normalize="index").to_string())

        log("\nDiscount Rate cross-tab by Basket Row Count (tx_rows):")
        log(pd.crosstab(df["tx_rows"], df["Discount Rate"], margins=True, normalize="index").to_string())
        log(pd.crosstab(df["tx_rows"], df["Discount Rate"], margins=True).to_string())

        log("\nWithin transactions that have at least one discounted line, how many lines are discounted?")
        disc_txs = tx_summary[tx_summary["tx_disc_lines"] > 0]
        log(pd.crosstab(disc_txs["tx_rows"], disc_txs["tx_disc_lines"], margins=True).to_string())

        # 3. Temporal & Monthly Trends by Branch
        log("\n" + "=" * 95)
        log("[3. TEMPORAL & MONTHLY TRENDS BY BRANCH]")
        log("=" * 95)
        m_branch = df.groupby(["Month", "Branch"]).agg(
            tx_cnt=("Transaction ID", "nunique"),
            line_cnt=("Transaction ID", "count"),
            qty=("Quantity", "sum"),
            rev=("Line Revenue (USD)", "sum"),
            gp=("Line Gross Profit (USD)", "sum"),
            known_cust_pct=("HasCustomer", lambda x: x.mean() * 100),
            disc_pct=("Discount Rate", lambda x: (x > 0).mean() * 100),
            takeaway_pct=("Order Type", lambda x: (x == "Takeaway").mean() * 100)
        )
        m_branch["margin_pct"] = m_branch["gp"] / m_branch["rev"] * 100
        m_branch["rev_per_tx"] = m_branch["rev"] / m_branch["tx_cnt"]
        m_branch["gp_per_tx"] = m_branch["gp"] / m_branch["tx_cnt"]
        log(m_branch.to_string())

        # Also check Monthly Category trends per Branch
        log("\nMonthly Transaction Count by Branch (Pivot):")
        log(df.pivot_table(index="Month", columns="Branch", values="Transaction ID", aggfunc="nunique").to_string())

        log("\nMonthly Gross Profit by Branch (Pivot):")
        log(df.pivot_table(index="Month", columns="Branch", values="Line Gross Profit (USD)", aggfunc="sum").to_string())

        # Check Category Mix by Branch
        log("\n" + "=" * 95)
        log("[4. CATEGORY & ORDER TYPE MIX BY BRANCH]")
        log("=" * 95)
        log("\nCategory Share of Quantity (%) by Branch:")
        cat_qty = pd.crosstab(df["Category"], df["Branch"], values=df["Quantity"], aggfunc="sum", normalize="columns") * 100
        log(cat_qty.to_string())

        log("\nCategory Share of Gross Profit (%) by Branch:")
        cat_gp = pd.crosstab(df["Category"], df["Branch"], values=df["Line Gross Profit (USD)"], aggfunc="sum", normalize="columns") * 100
        log(cat_gp.to_string())

        log("\nCategory Gross Margin (%) by Branch:")
        cat_rev_raw = pd.crosstab(df["Category"], df["Branch"], values=df["Line Revenue (USD)"], aggfunc="sum")
        cat_gp_raw = pd.crosstab(df["Category"], df["Branch"], values=df["Line Gross Profit (USD)"], aggfunc="sum")
        log(((cat_gp_raw / cat_rev_raw) * 100).to_string())

        log("\nOrder Type Share (%) by Branch (Transaction level):")
        tx_first = df.drop_duplicates("Transaction ID")
        log((pd.crosstab(tx_first["Branch"], tx_first["Order Type"], normalize="index") * 100).to_string())

        # 5. Hourly & Day of Week Patterns by Branch
        log("\n" + "=" * 95)
        log("[5. HOURLY & DAY-OF-WEEK PATTERNS BY BRANCH]")
        log("=" * 95)
        log("\nTransactions by Hour and Branch:")
        log(pd.crosstab(tx_first["Hour"], tx_first["Branch"], margins=True).to_string())

        log("\nTransactions by DayOfWeek and Branch:")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        log(pd.crosstab(tx_first["DayOfWeek"], tx_first["Branch"]).reindex(dow_order).to_string())

        # 6. Customer & Loyalty Analysis
        log("\n" + "=" * 95)
        log("[6. CUSTOMER & LOYALTY DYNAMICS]")
        log("=" * 95)
        cust_df = df[df["HasCustomer"]].copy()
        cust_tx = tx_first[tx_first["HasCustomer"]].copy()
        cust_stats = cust_df.groupby("Customer ID").agg(
            tx_count=("Transaction ID", "nunique"),
            branches_visited=("Branch", "nunique"),
            primary_branch=("Branch", lambda x: x.mode().iloc[0]),
            first_date=("Date_dt", "min"),
            last_date=("Date_dt", "max"),
            total_rev=("Line Revenue (USD)", "sum"),
            total_gp=("Line Gross Profit (USD)", "sum"),
            total_qty=("Quantity", "sum"),
            disc_lines=("Discount Rate", lambda x: (x > 0).sum()),
            max_disc=("Discount Rate", "max")
        )
        cust_stats["lifespan_days"] = (cust_stats["last_date"] - cust_stats["first_date"]).dt.days
        log(f"Total unique customers: {len(cust_stats):,}")
        log(f"Customers visiting >1 branch: {(cust_stats['branches_visited'] > 1).sum():,} ({(cust_stats['branches_visited'] > 1).mean()*100:.2f}%)")
        log("\nCustomer Summary Statistics:")
        log(cust_stats[["tx_count", "branches_visited", "total_rev", "total_gp", "total_qty", "disc_lines", "lifespan_days"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

        log("\nCustomer stats by Primary Branch:")
        log(cust_stats.groupby("primary_branch").agg(
            customers=("tx_count", "count"),
            avg_tx=("tx_count", "mean"),
            median_tx=("tx_count", "median"),
            one_and_done_pct=("tx_count", lambda x: (x == 1).mean() * 100),
            avg_rev=("total_rev", "mean"),
            avg_gp=("total_gp", "mean"),
            avg_lifespan=("lifespan_days", "mean")
        ).to_string())

        # Compare Known Customers vs Anonymous Transactions
        log("\nTransaction-Level Comparison: Known Customer vs Anonymous:")
        log(tx_first.groupby(["Branch", "HasCustomer"]).agg(
            tx_count=("Transaction ID", "count"),
            avg_items=("tx_qty", "mean"),
            multi_item_pct=("tx_rows", lambda x: (x > 1).mean() * 100),
            avg_rev=("tx_rev", "mean"),
            avg_gp=("tx_gp", "mean"),
            disc_tx_pct=("tx_disc_lines", lambda x: (x > 0).mean() * 100),
            takeaway_pct=("Order Type", lambda x: (x == "Takeaway").mean() * 100)
        ).to_string())

if __name__ == "__main__":
    run_deep_scanner()

