import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency

# Recomputes, directly from the raw CSV, every headline number quoted in README.md
# (model results are quoted from the outputs of scripts 08-10 and are not repeated here).

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
ADDON_GP = 3.04  # June test-set gross profit per converted add-on (script 08, Model A2)


def run_fact_check():
    out_file = OUTPUT_DIR / "11_readme_fact_check.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        df = pd.read_csv(DATA_PATH)
        raw_dup = df.duplicated(keep="first")  # on the original 14 columns, before helper columns are added
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month
        df["Hour"] = df["Time"].str[:2].astype(int)
        df["row"] = df.groupby("Transaction ID").cumcount() + 1
        df["n_rows"] = df.groupby("Transaction ID")["Product"].transform("size")
        rev, gp = "Line Revenue (USD)", "Line Gross Profit (USD)"
        tx = df.groupby("Transaction ID").agg(Date=("Date", "first"), Branch=("Branch", "first"), Hour=("Hour", "first"),
                                              Month=("Month", "first"), Type=("Order Type", "first"),
                                              Cust=("Customer ID", "first"), rev=(rev, "sum"), n=("Product", "size"))

        log("=" * 90)
        log("README FACT CHECK — every headline number recomputed from the raw CSV")
        log("=" * 90)

        log("\n[1] DATA CHECKS")
        log(f"Line items {len(df):,} | transactions {tx.shape[0]:,} | days {df['Date'].nunique()} "
            f"({df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}) | branches {df['Branch'].nunique()} | "
            f"categories {df['Category'].nunique()} | products {df['Product'].nunique()}")
        log(f"Revenue ${df[rev].sum():,.2f} | gross profit ${df[gp].sum():,.2f} | margin {df[gp].sum() / df[rev].sum():.2%}")
        recon = (df[rev] - df["Quantity"] * df["Unit Price (USD)"] * (1 - df["Discount Rate"])).abs().max()
        log(f"Max |revenue - qty x price x (1 - discount)| = ${recon:.3f}; rows with gross profit <= 0: {(df[gp] <= 0).sum()}")
        log(f"Transactions without Customer ID: {tx['Cust'].isna().mean():.2%}; distinct Customer IDs: {df['Customer ID'].nunique():,}")
        dup = raw_dup
        d2 = df[~dup]
        log(f"Exact duplicate rows: {dup.sum()} (revenue ${df.loc[dup, rev].sum():,.2f}); removing them changes revenue by "
            f"{1 - d2[rev].sum() / df[rev].sum():.2%} and margin by {d2[gp].sum() / d2[rev].sum() - df[gp].sum() / df[rev].sum():+.3%} points")
        static = df.groupby(["Branch", "Product"])[["Unit Price (USD)", "Unit Cost (USD)"]].nunique().max().max()
        log(f"Distinct prices/costs per branch-product over 181 days: max {static} (1 = never changed)")

        log("\n[2] GARDEN COST GAP")
        bm = df.groupby("Branch")[[gp, rev]].sum()
        log("Gross margin: " + ", ".join(f"{b} {r[gp] / r[rev]:.2%}" for b, r in bm.iterrows()))
        pc = df.groupby(["Product", "Branch"])[["Unit Price (USD)", "Unit Cost (USD)"]].first().unstack()
        price, cost = pc["Unit Price (USD)"], pc["Unit Cost (USD)"]
        for b in ["Harbour", "Uptown"]:
            r = price[b] / price["Garden"]
            log(f"Price {b}/Garden across 45 products: {r.min():.3f} to {r.max():.3f}")
        cr = cost["Garden"] / ((cost["Harbour"] + cost["Uptown"]) / 2)
        log(f"Cost Garden / avg(Harbour, Uptown): mean {cr.mean():.3f}, range {cr.min():.3f} to {cr.max():.3f} (all 45 products)")
        log(f"Example Still Water cost: Garden ${cost.loc['Still Water', 'Garden']:.2f}, Harbour ${cost.loc['Still Water', 'Harbour']:.2f}, "
            f"Uptown ${cost.loc['Still Water', 'Uptown']:.2f}")
        g = df[df["Branch"] == "Garden"]
        gq = g.groupby("Product")["Quantity"].sum()
        excess = (gq * (cost["Garden"] - (cost["Harbour"] + cost["Uptown"]) / 2)).sum()
        log(f"Garden excess cost vs two-branch average at Garden's volume ({gq.sum():,} units): ${excess:,.2f} in 6 months")
        log(f"Garden margin at Harbour's +5% price list, same cost: {1 - (1 - bm.loc['Garden', gp] / bm.loc['Garden', rev]) / 1.05:.1%}")

        log("\n[3] DISCOUNTS")
        df["disc_cost"] = df["Quantity"] * df["Unit Price (USD)"] * df["Discount Rate"]
        log(f"Total discount given: ${df['disc_cost'].sum():,.2f}")
        for rate in [0.15, 0.10]:
            s = df[df["Discount Rate"] == rate]
            log(f"  {rate:.0%}: {len(s):,} lines ({len(s) / len(df):.2%} of lines), ${s['disc_cost'].sum():,.2f}; branches "
                f"{sorted(s['Branch'].unique())}; categories {sorted(s['Category'].unique()) if rate == 0.15 else 'all ' + str(s['Category'].nunique())}")
        p15 = df[df["Discount Rate"] == 0.15]
        days15 = sorted(p15["Date"].dt.strftime("%m-%d").unique())
        log(f"  15% dates: {days15[0]}..{days15[7]} and {days15[8]}..{days15[-1]} ({len(days15)} days)")
        window = df[(df["Branch"] == "Harbour") & df["Date"].isin(p15["Date"].unique()) & df["Category"].isin(["Cold Coffee", "Desserts"])]
        log(f"  Share of Harbour Cold Coffee + Desserts lines discounted 15% on those days: {(window['Discount Rate'] == 0.15).mean():.1%}")
        ten = df["Discount Rate"] == 0.10
        log(f"  10% rate on lines with vs without Customer ID: {ten[df['Customer ID'].notna()].mean():.2%} vs {ten[df['Customer ID'].isna()].mean():.2%}")

        log("\n[4] DEMAND PATTERNS")
        daily = tx.groupby("Date").size()
        dow = daily.groupby(daily.index.dayofweek).mean()
        log(f"Average orders/day: Tuesday {dow[1]:.1f}, other days {daily[daily.index.dayofweek != 1].mean():.1f} "
            f"(ratio {dow[1] / daily[daily.index.dayofweek != 1].mean():.2f}), Saturday {dow[5]:.1f}")
        wk = daily.groupby(pd.Grouper(freq="W-SUN"))
        ratios = [w[w.index.dayofweek == 1].mean() / w[w.index.dayofweek != 1].mean() for _, w in wk if len(w) == 7]
        log(f"Tuesday ratio in each complete week: {min(ratios):.2f} to {max(ratios):.2f}, mean {np.mean(ratios):.2f} ({len(ratios)} weeks)")
        log("Orders/day by month: " + ", ".join(f"M{m} {v:.1f}" for m, v in daily.groupby(daily.index.month).mean().items())
            + f" (Jun/Jan {daily[daily.index.month == 6].mean() / daily[daily.index.month == 1].mean() - 1:+.0%})")
        hr = tx.groupby("Hour").size()
        log(f"Busiest hour {hr.idxmax()}:00 ({hr.max():,} orders); 20:00-22:59 share {hr.loc[20:22].sum() / hr.sum():.1%}; "
            f"quietest {hr.idxmin()}:00 ({hr.min():,})")
        log(f"Share of orders from 18:00 on: {(tx['Hour'] >= 18).mean():.1%}; share of gross profit from 18:00 on: "
            f"{df.loc[df['Hour'] >= 18, gp].sum() / df[gp].sum():.1%}")
        log(f"Takeaway share before 18:00 {(tx.loc[tx['Hour'] < 18, 'Type'] == 'Takeaway').mean():.1%}, "
            f"from 18:00 {(tx.loc[tx['Hour'] >= 18, 'Type'] == 'Takeaway').mean():.1%}")

        log("\n[5] BASKETS & ADD-ONS")
        log(f"Orders with 2+ items: {(tx['n'] >= 2).mean():.2%} ({(tx['n'] >= 2).sum():,}); single-item: {(tx['n'] == 1).sum():,}")
        r1 = df[(df["row"] == 1) & (df["n_rows"] >= 2)].set_index("Transaction ID")["Category"]
        r2 = df[df["row"] == 2].set_index("Transaction ID")["Category"]
        pair = pd.DataFrame({"a": r1, "b": r2}).dropna()
        for a, b in [("Matcha & Tea", "Bakery"), ("Sandwiches", "Cold Drinks"), ("Meals", "Cold Drinks")]:
            log(f"  First item {a} -> second item {b}: {(pair.loc[pair['a'] == a, 'b'] == b).mean():.1%}")
        unit_gp = (price - cost)
        for p_ in ["Still Water", "Sparkling Water", "Lemonade", "Fresh Orange Juice", "Chocolate Cookie", "Butter Croissant",
                   "Almond Croissant", "Banana Bread"]:
            log(f"  Unit gross profit {p_:<20} Harbour ${unit_gp.loc[p_, 'Harbour']:.2f} | Garden ${unit_gp.loc[p_, 'Garden']:.2f}")
        jun_single = (tx[(tx["Month"] == 6)]["n"] == 1).sum() / 30 * 365
        log(f"Single-item orders at June pace: {jun_single:,.0f}/yr. Add-on prompt value at ${ADDON_GP} GP per add-on: "
            + ", ".join(f"{c:.0%} -> ${jun_single * c * ADDON_GP:,.0f}/yr" for c in [0.01, 0.03, 0.05]))
        jun = df[df["Month"] == 6]
        extra_lines = jun[(jun["row"] >= 2)]
        bundle_cost = 0.10 * extra_lines[rev].sum() / 30 * 365
        avg_addon_rev = extra_lines[rev].sum() / len(extra_lines)
        avg_addon_gp = extra_lines[gp].sum() / len(extra_lines)
        net_gp_new = avg_addon_gp - 0.10 * avg_addon_rev
        need = bundle_cost / net_gp_new
        log(f"10% bundle discount on every 2nd/3rd item: subsidises existing multi-item orders by ${bundle_cost:,.0f}/yr (June pace); "
            f"a new add-on nets ${net_gp_new:.2f} GP after the discount, so {need:,.0f} extra add-ons/yr "
            f"({need / jun_single:.1%} of single-item orders) are needed just to break even")

        log("\n[6] BRANCH MENU TILT (first item of each order)")
        first = df[df["row"] == 1]
        tilt = pd.crosstab(first["Branch"], first["Category"], normalize="index")
        log("  Sandwiches: " + ", ".join(f"{b} {v:.2%}" for b, v in tilt["Sandwiches"].items()))
        log("  Desserts:   " + ", ".join(f"{b} {v:.2%}" for b, v in tilt["Desserts"].items()))

        log("\n[7] CUSTOMER ID BEHAVIOUR")
        ident = tx.dropna(subset=["Cust"])
        visits = ident.groupby("Cust").size()
        log(f"Visits per ID: mean {visits.mean():.2f}, variance/mean {visits.var() / visits.mean():.2f} (1.00 = pure chance)")
        first_seen = ident.reset_index().sort_values("Date").groupby("Cust")["Month"].first().value_counts().sort_index()
        log("New IDs first seen by month: " + ", ".join(f"M{m} {v:,}" for m, v in first_seen.items()))
        log(f"Average order value with vs without ID: ${ident['rev'].mean():.2f} vs ${tx[tx['Cust'].isna()]['rev'].mean():.2f}")
        log(f"Orders with a Customer ID: {len(ident):,}")
        tx["has_id"] = tx["Cust"].notna()
        tx["size"] = tx["n"].clip(upper=3)
        first_cat = df[df["row"] == 1].set_index("Transaction ID")["Category"]
        tx["first_cat"] = first_cat
        pvals = {c: chi2_contingency(pd.crosstab(tx[c], tx["has_id"]))[1] for c in ["Branch", "Hour", "first_cat", "size"]}
        log("Is a missing Customer ID linked to... (chi-square p-values; high = no link): "
            + ", ".join(f"{k} p={v:.2f}" for k, v in pvals.items()))

        log("\n[8] OTHER CHECKS QUOTED IN THE README")
        df["half"] = np.where(df["Month"] <= 3, "Jan-Mar", "Apr-Jun")
        hot = df["Category"] == "Hot Coffee"
        log(f"Hot Coffee share of items: Jan-Mar {hot[df['half'] == 'Jan-Mar'].mean():.1%}, Apr-Jun {hot[df['half'] == 'Apr-Jun'].mean():.1%} "
            f"(p={chi2_contingency(pd.crosstab(df['half'], hot))[1]:.2f})")
        log(f"Product share by month (do products trend?): chi-square p={chi2_contingency(pd.crosstab(df['Product'], df['Month']))[1]:.2f}")


if __name__ == "__main__":
    run_fact_check()
    print(f"Done. Report: {OUTPUT_DIR / '11_readme_fact_check.txt'}")
