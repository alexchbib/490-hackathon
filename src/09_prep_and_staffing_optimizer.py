import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import PoissonRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------------------
# ASSUMPTIONS (not observed in the data — every one is varied in a sensitivity table below)
# -----------------------------------------------------------------------------------------
PERISHABLE_CATS = ["Bakery", "Breakfast", "Desserts", "Meals", "Sandwiches"]
SALVAGE_SHARES = [0.0, 0.3]          # share of unit cost recovered on unsold items (0 = binned)
ORDERS_PER_STAFF_HOUR = [8, 12, 16]  # front-line throughput per staff member per hour
BASE_THROUGHPUT = 12
WAGES_USD_PER_HOUR = [3, 5, 8]       # fully loaded hourly staff cost
BASE_WAGE = 5
HOURS = list(range(7, 24))
BRANCHES = ["Garden", "Harbour", "Uptown"]
TEST_MONTH = 6
N_BOOT = 2000
RNG = np.random.default_rng(490)

# Reference palette (dataviz skill): categorical slots 1-3, text & grid tokens
C_BRANCH = {"Harbour": "#2a78d6", "Garden": "#eb6834", "Uptown": "#1baf7a"}
C_TEXT, C_TEXT2, C_GRID, C_MUTED, C_SURF = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae", "#fcfcfb"


def one_hot(frame, cols):
    return pd.get_dummies(frame[cols].astype(str), columns=cols, dtype=float)


def fit_poisson(X_train, y_train):
    model = PoissonRegressor(alpha=1e-8, max_iter=3000)
    model.fit(X_train, y_train)
    return model


def bootstrap_mean_ci(x):
    x = np.asarray(x)
    means = [RNG.choice(x, size=len(x), replace=True).mean() for _ in range(N_BOOT)]
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def run_prep_and_staffing():
    out_file = OUTPUT_DIR / "09_prep_and_staffing_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        log("=" * 100)
        log("PHASE 7: DECISION MODELS — PERISHABLE PREP OPTIMIZER & HOURLY STAFFING / CAPACITY FORECAST")
        log("Train: Jan-May 2026 | Test: unseen June 2026 | Model is fit once and NOT refit during June")
        log("=" * 100)

        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month
        df["Hour"] = df["Time"].str[:2].astype(int)
        all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        day0 = all_dates[0]

        # =================================================================================
        # TRACK A: PERISHABLE PREP OPTIMIZER (NEWSVENDOR)
        # =================================================================================
        log("\n" + "=" * 100)
        log("[TRACK A: HOW MANY UNITS OF EACH PERISHABLE SKU SHOULD EACH BRANCH PREP TOMORROW?]")
        log("=" * 100)
        log("Decision rule (newsvendor): prep the q-th quantile of tomorrow's demand, where")
        log("  q = critical ratio = Cu / (Cu + Co),  Cu = unit gross profit lost on a stock-out,")
        log("  Co = unit cost wasted on an unsold item = Unit Cost x (1 - salvage share).")
        log("With zero salvage, q equals the item's gross margin -> Garden's 1.46x cost directly lowers its optimal prep.")

        per = df[df["Category"].isin(PERISHABLE_CATS)]
        skus = per[["Branch", "Product", "Category"]].drop_duplicates()
        log(f"\nScope: {per['Product'].nunique()} perishable SKUs x 3 branches = {len(skus)} branch-SKU series, "
            f"{per['Quantity'].sum():,} units ({per['Quantity'].sum() / df['Quantity'].sum():.1%} of all units).")

        econ = per[per["Month"] < TEST_MONTH].groupby(["Branch", "Product"]).agg(
            rev=("Line Revenue (USD)", "sum"), qty=("Quantity", "sum"), cost=("Unit Cost (USD)", "first"))
        econ["price"] = econ["rev"] / econ["qty"]  # realized price per unit, net of discounts

        daily = per.groupby(["Date", "Branch", "Product"])["Quantity"].sum()
        grid = pd.MultiIndex.from_tuples(
            [(d, b, p) for d in all_dates for b, p in skus[["Branch", "Product"]].itertuples(index=False)],
            names=["Date", "Branch", "Product"])
        demand = daily.reindex(grid, fill_value=0).rename("D").reset_index()
        demand = demand.merge(econ[["price", "cost"]].reset_index(), on=["Branch", "Product"])
        demand["sku"] = demand["Branch"] + "|" + demand["Product"]
        demand["dow"] = demand["Date"].dt.dayofweek
        demand["t"] = (demand["Date"] - day0).dt.days / 30.0
        demand = demand.sort_values(["sku", "Date"]).reset_index(drop=True)
        g = demand.groupby("sku")["D"]
        demand["lag1"] = g.shift(1)
        demand["lag7"] = g.shift(7)
        demand["avg4w"] = sum(g.shift(7 * k) for k in range(1, 5)) / 4.0

        X = pd.concat([one_hot(demand, ["sku", "dow"]), demand[["t"]]], axis=1)
        train = demand["Date"].dt.month < TEST_MONTH
        test = demand["Date"].dt.month == TEST_MONTH
        pm = fit_poisson(X[train], demand.loc[train, "D"])
        demand["mu"] = pm.predict(X)
        te = demand[test].copy()

        log(f"Demand model: Poisson GLM  log(mu) = SKU-at-branch + DayOfWeek + trend   "
            f"(train rows {train.sum():,}, test rows {test.sum():,}; mean June demand {te['D'].mean():.2f} units/SKU/day)")
        log(f"  June forecast MAE: model {np.abs(te['D'] - te['mu']).mean():.3f} | lag-1 {np.abs(te['D'] - te['lag1']).mean():.3f} | "
            f"lag-7 {np.abs(te['D'] - te['lag7']).mean():.3f} | 4-week same-weekday avg {np.abs(te['D'] - te['avg4w']).mean():.3f} units")

        def critical_ratio(frame, salvage):
            cu = frame["price"] - frame["cost"]
            co = frame["cost"] * (1 - salvage)
            return cu / (cu + co)

        def profit(frame, prep, salvage):
            sold = np.minimum(prep, frame["D"])
            unsold = prep - sold
            return (frame["price"] * sold - frame["cost"] * prep + frame["cost"] * salvage * unsold,
                    sold, unsold, frame["D"] - sold)

        pooled_margin = 1 - (econ["cost"] * econ["qty"]).sum() / econ["rev"].sum()

        for salvage in SALVAGE_SHARES:
            te["cr"] = critical_ratio(te, salvage)
            pooled_cr = (pooled_margin) / (pooled_margin + (1 - pooled_margin) * (1 - salvage))
            rules = {
                "R1 Yesterday's sales (lag-1)": te["lag1"],
                "R2 Same weekday last week (lag-7)": te["lag7"],
                "R3 4-week same-weekday average, rounded up": np.ceil(te["avg4w"]),
                "R4 Model mean forecast, rounded": np.round(te["mu"]),
                "R5 'Never run out' (model 95th pct)": stats.poisson.ppf(0.95, te["mu"]),
                f"R6 Model + ONE company-wide ratio ({pooled_cr:.0%})": stats.poisson.ppf(pooled_cr, te["mu"]),
                "R7 Model + per-branch-SKU ratio (RECOMMENDED)": stats.poisson.ppf(te["cr"], te["mu"]),
                "Oracle (perfect foresight, upper bound)": te["D"],
            }
            oracle_gp = profit(te, te["D"], salvage)[0].sum()
            log(f"\n--- June 2026 out-of-sample prep simulation | salvage = {salvage:.0%} of unit cost ---")
            log(f"{'Prep rule':<52}{'June GP $':>11}{'% oracle':>10}{'Waste u':>9}{'Lost u':>8}{'Fill':>7}")
            results = {}
            for name, prep in rules.items():
                gp, sold, unsold, lost = profit(te, prep, salvage)
                results[name] = te.assign(gp=gp).groupby("Date")["gp"].sum()
                log(f"{name:<52}{gp.sum():>11,.0f}{gp.sum() / oracle_gp:>10.1%}{unsold.sum():>9,.0f}"
                    f"{lost.sum():>8,.0f}{sold.sum() / te['D'].sum():>7.1%}")

            names_ = list(rules)
            fc_gain = (results[names_[3]] - results[names_[2]]).mean()
            cr_gain = (results[names_[6]] - results[names_[3]]).mean()
            log(f"  Decomposition of R7's gain over R3: better forecast (R4-R3) ${fc_gain:+,.2f}/day + "
                f"margin-based prep level (R7-R4) ${cr_gain:+,.2f}/day")
            rec = results["R7 Model + per-branch-SKU ratio (RECOMMENDED)"]
            for base in ["R3 4-week same-weekday average, rounded up", "R2 Same weekday last week (lag-7)",
                         list(rules)[5]]:
                m, lo, hi = bootstrap_mean_ci(rec - results[base])
                log(f"  R7 vs {base[:2]}: +${m:,.2f}/day across 66 SKUs (95% bootstrap CI over June days: "
                    f"${lo:,.2f} to ${hi:,.2f})  ->  ~${m * 365:,.0f}/yr at June volume")

            if salvage == 0.0:
                prep_main = rules["R7 Model + per-branch-SKU ratio (RECOMMENDED)"]
                te["hit"] = te["D"] <= prep_main
                by_branch = te.groupby("Branch").agg(target=("cr", "mean"), achieved=("hit", "mean"))
                log("\n  Calibration (target service level = mean critical ratio; achieved = share of June SKU-days with no stock-out):")
                for b, r in by_branch.iterrows():
                    log(f"    {b:<8} target {r['target']:.1%}  achieved {r['achieved']:.1%}")
                calib = by_branch
                prep_results = {k: v.sum() / profit(te, te["D"], 0.0)[0].sum() for k, v in results.items()}
                br_gain = te.assign(gp_r7=profit(te, prep_main, 0)[0],
                                    gp_r3=profit(te, np.ceil(te["avg4w"]), 0)[0]).groupby("Branch")[["gp_r7", "gp_r3"]].sum()
                log("\n  June GP by branch, R7 vs R3 (zero salvage):")
                for b, r in br_gain.iterrows():
                    log(f"    {b:<8} R7 ${r['gp_r7']:,.0f} vs R3 ${r['gp_r3']:,.0f}  (diff ${r['gp_r7'] - r['gp_r3']:+,.0f})")
                g_cr = te[te["Branch"] == "Garden"]["cr"].mean()
                h_cr = te[te["Branch"] == "Harbour"]["cr"].mean()
                log(f"\n  Garden's mean critical ratio is {g_cr:.0%} vs Harbour's {h_cr:.0%}: for identical demand, Garden should")
                log("  prep to a LOWER service level — accepting more sell-outs — because each wasted unit costs it 1.46x more.")

        log("\nAssumptions & limits (Track A):")
        log("  * Recorded sales = true demand. If items sold out in history, demand is under-counted (censored) and every rule under-preps.")
        log("  * No substitution: a customer facing a sold-out SKU buys nothing. In reality many switch to a sibling SKU, which shrinks")
        log("    the cost of stock-outs AND the absolute $ gaps between rules. Treat the $ figures as an upper-range simulation;")
        log("    the ranking of rules and the Garden-vs-Harbour prep difference are the robust results.")
        log("  * Achieved service levels exceed targets because demand is a small integer (~2.5/day): the optimal prep is the")
        log("    smallest whole unit whose cumulative probability reaches the ratio, which overshoots it.")
        log("  * Prep is decided once per day per SKU with no batch-size constraint; real bakery batches (e.g. trays of 6) blunt the gain.")
        log("  * Waste value = Unit Cost from the data. If Garden's 1.46x cost is an accounting allocation (Finding 1), its true waste")
        log("    cost is lower and its critical ratio should be closer to Harbour's — the audit in Finding 1 also sets this parameter.")
        log("  * Practical test: run R7 at one branch for 4 weeks with daily waste and sell-out-time logs; compare waste $ and lost sales to R3.")

        # =================================================================================
        # TRACK B: HOURLY DEMAND, STAFFING & CAPACITY DATE
        # =================================================================================
        log("\n" + "=" * 100)
        log("[TRACK B: HOURLY ORDER FORECAST WITH PREDICTION INTERVALS -> STAFFING SCHEDULE & CAPACITY DATE]")
        log("=" * 100)

        tx = df.drop_duplicates("Transaction ID")
        hourly = tx.groupby(["Date", "Branch", "Hour"]).size()
        hgrid = pd.MultiIndex.from_product([all_dates, BRANCHES, HOURS], names=["Date", "Branch", "Hour"])
        hr = hourly.reindex(hgrid, fill_value=0).rename("n").reset_index()
        hr["bh"] = hr["Branch"] + "|" + hr["Hour"].astype(str)
        hr["dow"] = hr["Date"].dt.dayofweek
        hr["t"] = (hr["Date"] - day0).dt.days / 30.0
        hr = hr.sort_values(["bh", "Date"]).reset_index(drop=True)
        hg = hr.groupby("bh")["n"]
        hr["lag7"] = hg.shift(7)
        hr["avg4w"] = sum(hg.shift(7 * k) for k in range(1, 5)) / 4.0
        disp = hr.groupby([hr["Date"].dt.month, "dow", "bh"])["n"].agg(["mean", "var"])
        log(f"Hourly counts are Poisson-like: mean variance/mean ratio within (month, weekday, branch, hour) cells = "
            f"{(disp['var'] / disp['mean']).mean():.2f} (1.00 = Poisson)")

        HX = pd.concat([one_hot(hr, ["bh", "dow"]), hr[["t"]]], axis=1)
        htrain = hr["Date"].dt.month < TEST_MONTH
        htest = hr["Date"].dt.month == TEST_MONTH
        hm = fit_poisson(HX[htrain], hr.loc[htrain, "n"])
        hr["mu"] = hm.predict(HX)
        hr["static_mean"] = hr.groupby(["bh", "dow"])["n"].transform(lambda s: s[hr.loc[s.index, "Date"].dt.month < TEST_MONTH].mean())
        ht = hr[htest].copy()

        log(f"Model: Poisson GLM  log(mu) = Branch x Hour + DayOfWeek + trend   (train {htrain.sum():,} branch-hours, test {htest.sum():,})")
        log(f"\n{'June hourly forecast':<46}{'MAE':>8}{'RMSE':>8}{'Bias':>8}{'Tue 20-22h MAE':>16}")
        peak_mask = (ht["dow"] == 1) & ht["Hour"].between(20, 22)
        for name, col in [("Baseline: Jan-May mean (branch x weekday x hour)", "static_mean"),
                          ("Baseline: same hour last week (lag-7)", "lag7"),
                          ("Baseline: 4-week same weekday-hour average", "avg4w"),
                          ("Model: Poisson GLM with trend", "mu")]:
            e = ht["n"] - ht[col]
            log(f"{name:<46}{e.abs().mean():>8.2f}{np.sqrt((e ** 2).mean()):>8.2f}{e.mean():>+8.2f}{e[peak_mask].abs().mean():>16.2f}")

        log("\nPrediction-interval coverage on June (does the stated range contain the actual count as often as claimed?):")
        coverage = {}
        for level in [0.80, 0.90]:
            lo = stats.poisson.ppf((1 - level) / 2, ht["mu"])
            hi = stats.poisson.ppf(1 - (1 - level) / 2, ht["mu"])
            cov = ((ht["n"] >= lo) & (ht["n"] <= hi)).mean()
            coverage[level] = cov
            log(f"  {level:.0%} interval -> actual coverage {cov:.1%}  (discrete intervals are slightly conservative by construction)")
        q90_up = stats.poisson.ppf(0.90, ht["mu"])
        log(f"  One-sided 90th percentile (used for staffing): actual <= P90 in {(ht['n'] <= q90_up).mean():.1%} of June branch-hours")

        log("\nStaffing translation (ASSUMPTION-DRIVEN: throughput per staff-hour and wage are not in the data).")
        log("Schedule rule: staff in hour h = max(1, ceil(P90 forecast / throughput)). Compared against a FLAT schedule that")
        log("keeps each branch's daily peak-hour crew on all 17 opening hours (07:00-23:59), a common default.")
        log("A second comparison, 'static profile', sizes each hour from the Jan-May P90 of that branch x weekday x hour with")
        log("no trend: it shows how much of the saving needs no model at all, and what the trend term adds (fewer understaffed hours).")
        log(f"\n{'Throughput':>11}{'Staff-hours in June (flat / static / model)':>46}{'Saved vs flat':>15}{'Understaffed hours (flat / static / model)':>45}")
        staffing_rows = []
        q90_static = stats.poisson.ppf(0.90, ht["static_mean"])
        for thr in ORDERS_PER_STAFF_HOUR:
            ht["staff_model"] = np.maximum(1, np.ceil(q90_up / thr))
            ht["staff_static"] = np.maximum(1, np.ceil(q90_static / thr))
            ht["staff_flat"] = ht.groupby(["Date", "Branch"])["staff_model"].transform("max")
            under = {k: (ht["n"] > ht[k] * thr).sum() for k in ["staff_flat", "staff_static", "staff_model"]}
            hours_ = {k: ht[k].sum() for k in ["staff_flat", "staff_static", "staff_model"]}
            saved = hours_["staff_flat"] - hours_["staff_model"]
            staffing_rows.append((thr, saved))
            log(f"{thr:>11}{hours_['staff_flat']:>22,.0f} / {hours_['staff_static']:,.0f} / {hours_['staff_model']:,.0f}"
                f"{saved:>15,.0f}{under['staff_flat']:>27} / {under['staff_static']} / {under['staff_model']}")
        log("\nAnnualized labour saving vs flat schedule (June staff-hours saved x 365/30 x wage):")
        log(f"{'Throughput':>11}" + "".join(f"{'$' + str(w) + '/h':>10}" for w in WAGES_USD_PER_HOUR))
        for thr, saved in staffing_rows:
            log(f"{thr:>11}" + "".join(f"{saved * 365 / 30 * w:>10,.0f}" for w in WAGES_USD_PER_HOUR))

        gp_per_order = tx.merge(df.groupby("Transaction ID")["Line Gross Profit (USD)"].sum().rename("gp"),
                                left_on="Transaction ID", right_index=True)
        jun = gp_per_order[gp_per_order["Month"] == TEST_MONTH]
        log("\nThe 07:00 opening hour in June (orders and gross profit per branch-hour):")
        for b in BRANCHES:
            s = jun[(jun["Branch"] == b) & (jun["Hour"] == 7)]
            log(f"  {b:<8} {len(s) / 30:.2f} orders/hour, ${s['gp'].sum() / 30:.2f} GP/hour "
                f"(vs ${BASE_WAGE}/h assumed staff cost; some of these orders would shift to 08:00 if opening moved)")
        sat = jun[(jun["Date"].dt.dayofweek == 5) & (jun["Hour"] < 10)]
        log(f"  Saturday 07:00-09:59 across all branches: {len(sat) / (sat['Date'].nunique() * 9):.2f} orders per branch-hour")

        log("\nCapacity outlook — week's busiest hour (Tuesday 21:00), mean and P90 orders/hour by month:")
        log("  Months 1-6 = observed mean over that month's Tuesdays; months 7-12 = model extrapolation IF the Jan-Jun")
        log("  trend continues (the single most fragile assumption here; the 'growth stops' case stays at June's level).")
        coef_t = hm.coef_[list(HX.columns).index("t")]
        log(f"  Fitted trend: {np.exp(coef_t) - 1:+.1%} per 30 days")
        outlook = {}
        for b in BRANCHES:
            obs = hr[(hr["Branch"] == b) & (hr["dow"] == 1) & (hr["Hour"] == 21)].groupby(hr["Date"].dt.month)["n"].mean()
            june_mu = hr[(hr["Branch"] == b) & (hr["dow"] == 1) & (hr["Hour"] == 21) & htest]["mu"].mean()
            row = {"mu": june_mu}
            proj = {m: june_mu * np.exp(coef_t * (m - TEST_MONTH)) for m in range(7, 13)}
            outlook[b] = (obs, proj)
            line = " ".join(f"M{m}:{v:5.1f}" for m, v in obs.items()) + f" | model June {june_mu:.1f} | " + \
                   " ".join(f"M{m}:{v:5.1f}(P90 {stats.poisson.ppf(0.9, v):.0f})" for m, v in proj.items())
            log(f"  {b:<8} {line}")
            for thr in ORDERS_PER_STAFF_HOUR:
                crew_now = max(1, int(np.ceil(stats.poisson.ppf(0.9, row["mu"]) / thr)))
                exceed = [m for m, v in proj.items() if stats.poisson.ppf(0.9, v) > crew_now * thr]
                when = f"month {exceed[0]}" if exceed else "not before December"
                log(f"           at {thr:>2} orders/staff-h: June crew of {crew_now} is exceeded at the P90 in {when}")

        log("\nAssumptions & limits (Track B):")
        log("  * Sales count = demand. If the Tuesday 21:00 queue already turns people away, true peak demand is higher.")
        log("  * Seating is not modelled: 72% of post-18:00 orders are dine-in, so tables may bind before staff does.")
        log("  * Throughput and wage are assumed; the staffing $ scale linearly with wage and the table shows the range.")
        log("  * Practical test: time 2 weeks of Tuesday 20:00-22:00 orders (order-to-hand-off) to measure real throughput, then")
        log("    reset the throughput constant; run the hourly roster at one branch for 4 weeks against its current roster.")

    # =========================================================================================
    # FIGURE 5
    # =========================================================================================
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT2,
                         "xtick.color": C_TEXT2, "ytick.color": C_TEXT2, "axes.titlecolor": C_TEXT,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": C_SURF,
                         "axes.facecolor": C_SURF})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    # A — prep rules, % of perfect-foresight profit
    ax = axes[0, 0]
    names = [k for k in prep_results if not k.startswith("Oracle")]
    vals = [prep_results[k] * 100 for k in names]
    colors = [C_BRANCH["Harbour"] if "RECOMMENDED" in k else C_MUTED for k in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", color=C_TEXT, fontsize=9)
    ax.set_yticks(y, [n.replace(" (RECOMMENDED)", "  ← recommended") for n in names], fontsize=9)
    ax.set_xlim(min(vals) - 5, 100)
    ax.set_xlabel("June gross profit on perishables, % of perfect-foresight prep")
    ax.set_title("A. Perishable prep rules tested on unseen June (zero salvage)", loc="left")
    ax.grid(axis="x", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)

    # B — target vs achieved service level per branch
    ax = axes[0, 1]
    x = np.arange(len(BRANCHES))
    ax.bar(x - 0.18, calib.loc[BRANCHES, "target"] * 100, width=0.34, color=C_MUTED, label="Target (critical ratio)")
    ax.bar(x + 0.18, calib.loc[BRANCHES, "achieved"] * 100, width=0.34,
           color=C_TEXT2, label="Achieved in June (no-stock-out days)")
    for xi, b in zip(x, BRANCHES):
        ax.text(xi - 0.18, calib.loc[b, "target"] * 100 + 1, f"{calib.loc[b, 'target']:.0%}", ha="center", color=C_TEXT, fontsize=9)
        ax.text(xi + 0.18, calib.loc[b, "achieved"] * 100 + 1, f"{calib.loc[b, 'achieved']:.0%}", ha="center", color=C_TEXT, fontsize=9)
    ax.set_xticks(x, BRANCHES)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of SKU-days with no stock-out (%)")
    ax.set_title("B. Garden's higher unit cost → lower optimal service level", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)

    # C — one unseen Tuesday at Harbour: forecast, P10-P90 band, actual
    ax = axes[1, 0]
    last_tue = hr[(hr["dow"] == 1) & htest]["Date"].max()
    d = hr[(hr["Date"] == last_tue) & (hr["Branch"] == "Harbour")].sort_values("Hour")
    ax.fill_between(d["Hour"], stats.poisson.ppf(0.10, d["mu"]), stats.poisson.ppf(0.90, d["mu"]),
                    color=C_BRANCH["Harbour"], alpha=0.15, lw=0, label="80% prediction interval")
    ax.plot(d["Hour"], d["mu"], color=C_BRANCH["Harbour"], lw=2, label="Forecast (trained Jan–May)")
    ax.plot(d["Hour"], d["n"], "o", ms=6, color=C_TEXT, mec=C_SURF, mew=1.5, label="Actual orders")
    ax.set_xticks(HOURS[::2], [f"{h:02d}:00" for h in HOURS[::2]])
    ax.set_ylabel("Orders per hour")
    ax.set_title(f"C. Harbour, {last_tue:%a %d %b} (unseen): hourly forecast vs actual", loc="left")
    ax.text(0.99, 0.98, f"80% interval held {coverage[0.80]:.0%} of June branch-hours", transform=ax.transAxes,
            ha="right", va="top", color=C_TEXT2, fontsize=8.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)

    # D — busiest hour outlook
    ax = axes[1, 1]
    for b in BRANCHES:
        obs, proj = outlook[b]
        ax.plot(list(obs.index), list(obs.values), color=C_BRANCH[b], lw=2, marker="o", ms=5)
        june_mu = list(proj.values())[0] / np.exp(coef_t)
        ax.plot([6] + list(proj), [june_mu] + list(proj.values()), color=C_BRANCH[b], lw=2, ls="--")
        ax.text(12.15, list(proj.values())[-1], b, color=C_TEXT, va="center", fontsize=9)
    for k in [1, 2, 3]:
        ax.axhline(k * BASE_THROUGHPUT, color=C_MUTED, lw=0.8, ls=":")
        ax.text(6.95, k * BASE_THROUGHPUT + 0.4, f"{k} staff @ {BASE_THROUGHPUT} orders/h (assumed)", color=C_TEXT2, fontsize=8)
    ax.axvspan(6.5, 12.5, color=C_GRID, alpha=0.35, lw=0)
    ax.text(9.5, 3.2, "projection if Jan–Jun trend continues", ha="center", va="bottom",
            color=C_TEXT2, fontsize=8.5)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.7, 13)
    ax.set_ylabel("Mean orders in the Tuesday 21:00 hour")
    ax.set_title("D. Busiest hour of the week (Tue 21:00), by month", loc="left")
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_prep_and_staffing.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    run_prep_and_staffing()
    print(f"Done. Report: {OUTPUT_DIR / '09_prep_and_staffing_report.txt'} | Figure: {FIG_DIR / 'fig5_prep_and_staffing.png'}")
