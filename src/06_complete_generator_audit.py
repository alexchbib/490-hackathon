import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")

def run_complete_audit():
    df = pd.read_csv(DATA_PATH)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date_dt"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["Date_dt"].dt.day_name()
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    df["Minute"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.minute
    df["Second"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.second
    df["HasCustomer"] = df["Customer ID"].notna()
    df["row_num"] = df.groupby("Transaction ID").cumcount() + 1
    df["tx_rows"] = df["Transaction ID"].map(df.groupby("Transaction ID")["Product"].count())

    out_file = OUTPUT_DIR / "06_complete_generator_audit.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        log("=" * 95)
        log("FINAL CHECK OF ALL SUB-DIMENSIONS (PRODUCT X HOUR, ORDER TYPE X BRANCH/HOUR, MINUTES/SECONDS)")
        log("=" * 95)

        # 1. Check if Product share WITHIN Category varies by Hour or Month or Order Type
        log("\n[1. DOES PRODUCT SHARE WITHIN CATEGORY VARY BY HOUR, MONTH, OR ORDER TYPE?]")
        for cat, grp in df.groupby("Category"):
            p_hr = stats.chi2_contingency(pd.crosstab(grp["Product"], grp["Hour"])).pvalue
            p_mo = stats.chi2_contingency(pd.crosstab(grp["Product"], grp["Month"])).pvalue
            p_ot = stats.chi2_contingency(pd.crosstab(grp["Product"], grp["Order Type"])).pvalue
            p_br = stats.chi2_contingency(pd.crosstab(grp["Product"], grp["Branch"])).pvalue
            p_rn = stats.chi2_contingency(pd.crosstab(grp["Product"], grp["row_num"])).pvalue
            log(f"  Category={cat:<15} | p(Hour)={p_hr:.4f} | p(Month)={p_mo:.4f} | p(OrderType)={p_ot:.4f} | p(Branch)={p_br:.4f} | p(row_num)={p_rn:.4f}")

        # 2. Check if Category x Hour varies by Branch
        log("\n[2. DOES CATEGORY X FIELD INTERACTION VARY?]")
        tx1 = df[df["row_num"] == 1]
        log(f"Row 1 Category vs DayOfWeek Chi2 p-value: {stats.chi2_contingency(pd.crosstab(tx1['Category'], tx1['DayOfWeek'])).pvalue:.4f}")
        log(f"Row 1 Category vs Month Chi2 p-value:     {stats.chi2_contingency(pd.crosstab(tx1['Category'], tx1['Month'])).pvalue:.4f}")
        log(f"Row 1 Category vs HasCustomer p-value:    {stats.chi2_contingency(pd.crosstab(tx1['Category'], tx1['HasCustomer'])).pvalue:.4f}")
        log(f"Row 1 Category vs OrderType p-value:      {stats.chi2_contingency(pd.crosstab(tx1['Category'], tx1['Order Type'])).pvalue:.4f}")
        # Wait: earlier we saw Category vs OrderType had some variation (Breakfast 46.9% takeaway vs Matcha 38.0% takeaway)
        # Why? Because Breakfast is bought in Morning (when Takeaway is 54%) and Matcha is bought in Evening (when Takeaway is 28%)!
        # Let's test if Category is independent of Order Type CONDITIONED on Hour regime!
        df["HourRegime"] = pd.cut(df["Hour"], bins=[0, 10, 17, 24], labels=["Morning(7-10)", "Midday(11-17)", "Evening(18-23)"])
        tx1 = df[df["row_num"] == 1]
        for reg, grp in tx1.groupby("HourRegime", observed=False):
            p_val = stats.chi2_contingency(pd.crosstab(grp["Category"], grp["Order Type"])).pvalue
            log(f"  Conditioned on {reg:<16}: Row 1 Category vs Order Type Chi2 p-value = {p_val:.4f}")

        # 3. Check Hour distribution across DayOfWeek and Branch and Month
        log("\n[3. DOES HOURLY PROFILE VARY BY DAY OF WEEK, BRANCH, OR MONTH?]")
        tx_first = df.drop_duplicates("Transaction ID")
        log(f"Hour vs Branch Chi2 p-value:      {stats.chi2_contingency(pd.crosstab(tx_first['Hour'], tx_first['Branch'])).pvalue:.4f}")
        log(f"Hour vs DayOfWeek Chi2 p-value:   {stats.chi2_contingency(pd.crosstab(tx_first['Hour'], tx_first['DayOfWeek'])).pvalue:.4f}")
        log(f"Hour vs Month Chi2 p-value:       {stats.chi2_contingency(pd.crosstab(tx_first['Hour'], tx_first['Month'])).pvalue:.4f}")
        log(f"Hour vs HasCustomer Chi2 p-value: {stats.chi2_contingency(pd.crosstab(tx_first['Hour'], tx_first['HasCustomer'])).pvalue:.4f}")

        # 4. Check Minute and Second distribution
        log(f"\nMinute min={df['Minute'].min()}, max={df['Minute'].max()}, uniform Chi2 p={stats.chisquare(df['Minute'].value_counts()).pvalue:.4f}")
        log(f"Second min={df['Second'].min()}, max={df['Second'].max()}, uniform Chi2 p={stats.chisquare(df['Second'].value_counts()).pvalue:.4f}")

        # 5. Check exact Row 1 Category distribution by Branch and HourRegime
        log("\n[4. ROW 1 CATEGORY DISTRIBUTION (%) BY BRANCH]")
        log((pd.crosstab(tx1["Category"], tx1["Branch"], normalize="columns") * 100).round(2).to_string())

        log("\n[5. ROW 1 CATEGORY DISTRIBUTION (%) BY HOUR REGIME]")
        log((pd.crosstab(tx1["Category"], tx1["HourRegime"], normalize="columns") * 100).round(2).to_string())

        # 6. Check if multi-item probability (tx_rows >= 2 and tx_rows == 3) depends ONLY on Row 1 Category!
        log("\n[6. BASKET SIZE (1, 2, 3 ROWS) DISTRIBUTION (%) BY ROW 1 CATEGORY]")
        log((pd.crosstab(tx1["Category"], tx1["tx_rows"], normalize="index") * 100).round(2).to_string())

if __name__ == "__main__":
    run_complete_audit()

