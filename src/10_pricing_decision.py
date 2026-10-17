import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("Project delivirables/stories_transactions.csv")
OUTPUT_DIR = Path("outputs")
FIG_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------------------
# ASSUMPTIONS (not observed in the data)
# -----------------------------------------------------------------------------------------
PRICE_STEPS = [0.02, 0.03, 0.05, 0.08]
MAIN_STEP = 0.05                  # +5% = Garden moves exactly onto Harbour's existing price list
PILOT_BRANCH = "Garden"
PRE_DAYS = 56                     # 8 weeks of pre-period for the pilot comparison
MAX_PILOT_WEEKS = 12
STOP_Z = 0.84                     # stop when >= 80% confident the volume loss exceeds break-even
ELASTICITY_GRID = [0.0, -0.25, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0]
N_SIM = 4000
N_BOOT = 5000
RNG = np.random.default_rng(490)

C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
C_TEXT, C_TEXT2, C_GRID, C_MUTED, C_SURF = "#0b0b0b", "#52514e", "#e4e3df", "#b5b4ae", "#fcfcfb"


def boot_diff(a, b):
    est = a.mean() - b.mean()
    bs = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def residualize(s):
    return s - s.groupby([s.index.month, s.index.dayofweek]).transform("mean")


def gp_change_rate(margin, p, eps):
    """Change in gross profit as a share of current revenue after a +p price rise with elasticity eps."""
    return (1 + eps * p) * (margin + p) - margin


def run_pricing_decision():
    out_file = OUTPUT_DIR / "10_pricing_decision_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        def log(msg=""):
            print(msg, file=f)

        log("=" * 100)
        log("PHASE 8: THE PRICING DECISION — WHAT THE DATA CAN AND CANNOT SAY, AND A STAGED PRICE RISE WITH A STOP RULE")
        log("=" * 100)

        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month

        # =================================================================================
        # STEP 1: WHAT DID HARBOUR'S 15% PROMO REVEAL ABOUT PRICE SENSITIVITY?
        # =================================================================================
        log("\n[STEP 1: THE ONLY PRICE CHANGE IN THE DATA — HARBOUR'S 15% PROMO (16 DAYS, COLD COFFEE + DESSERTS)]")
        promo_days = set(df.loc[df["Discount Rate"] == 0.15, "Date"].unique())
        df["tgt"] = df["Category"].isin(["Cold Coffee", "Desserts"])
        df["H"] = df["Branch"] == "Harbour"
        u = df.groupby(["Date", "H", "tgt"])["Quantity"].sum().unstack(["H", "tgt"])
        item_ddd = np.log(u[(True, True)] / u[(True, False)]) - np.log(u[(False, True)] / u[(False, False)])
        txn = df.drop_duplicates("Transaction ID").groupby(["Date", "H"]).size().unstack()
        traffic_dd = np.log(txn[True] / txn[False])

        log("Promo-item units: Harbour promo categories vs Harbour other categories, relative to the same ratio at Garden+Uptown.")
        log("Traffic: Harbour transactions relative to Garden+Uptown transactions.")
        log(f"{'Outcome':<22}{'Control days':<44}{'Effect':>9}{'95% CI':>22}")
        promo_ci = None
        for name, s in [("Promo-item units", item_ddd), ("Harbour traffic", traffic_dd)]:
            for spec, series, mask in [
                ("Mar & May non-promo days (same months)", s, s.index.month.isin([3, 5])),
                ("All non-promo days, month x weekday adj.", residualize(s), np.ones(len(s), bool)),
            ]:
                is_p = series.index.isin(promo_days)
                est, lo, hi = boot_diff(series[mask & is_p].values, series[mask & ~is_p].values)
                log(f"{name:<22}{spec:<44}{np.exp(est) - 1:>+9.1%}   [{np.exp(lo) - 1:+.1%}, {np.exp(hi) - 1:+.1%}]")
                if name == "Promo-item units" and promo_ci is None:
                    promo_ci = (np.exp(lo) - 1, np.exp(est) - 1, np.exp(hi) - 1)
        eps_ci = [x / -0.15 for x in promo_ci]
        log(f"\nImplied price elasticity of promo items (volume change / -15% price): point {eps_ci[1]:+.2f}, "
            f"95% CI {eps_ci[2]:+.2f} to {eps_ci[0]:+.2f}")
        log("Reading: traffic did not respond (the CI rules out more than about +7%), but the item-level CI is too wide to")
        log("pin down price sensitivity. It cannot distinguish 'customers ignore price' from 'customers are very price-sensitive'.")
        log("Why: 16 days, one branch, two categories, ~20 promo units/day, and day-to-day noise of ~15%.")
        log("CORRECTION to Finding 2: 'zero detectable lift' should read 'no traffic lift; item-level lift inconclusive'.")

        # =================================================================================
        # STEP 2: ECONOMICS OF A PRICE RISE
        # =================================================================================
        log("\n" + "=" * 100)
        log("[STEP 2: WHAT A MENU PRICE RISE IS WORTH, AND HOW PRICE-SENSITIVE CUSTOMERS WOULD HAVE TO BE TO MAKE IT LOSE MONEY]")
        log("=" * 100)
        jun = df[df["Month"] == 6]
        branch = pd.DataFrame({
            "rev_day": jun.groupby("Branch")["Line Revenue (USD)"].sum() / 30,
            "margin": df.groupby("Branch")["Line Gross Profit (USD)"].sum() / df.groupby("Branch")["Line Revenue (USD)"].sum(),
        })
        total_rev_day = branch["rev_day"].sum()
        m_all = df["Line Gross Profit (USD)"].sum() / df["Line Revenue (USD)"].sum()
        log(f"Base: June 2026 run-rate revenue ${total_rev_day:,.0f}/day (${total_rev_day * 365:,.0f}/yr, no further growth assumed); "
            f"gross margin {m_all:.1%}")
        log("Formula: change in GP = Revenue x [(1 + e x p)(m + p) - m], e = elasticity, p = price rise, m = margin.")
        log("Break-even elasticity e* = -1 / (m + p): any customer response milder than e* makes the rise profitable.\n")
        log(f"{'Price rise':>10}{'Break-even vol. loss':>22}{'Break-even e*':>15}"
            + "".join(f"{'GP/yr @ e=' + format(e, '+.1f'):>17}" for e in [0.0, -0.5, -1.0, -2.0]))
        for p in PRICE_STEPS:
            be_loss = p / (m_all + p)
            row = f"{p:>+10.0%}{be_loss:>22.1%}{-1 / (m_all + p):>15.2f}"
            for e in [0.0, -0.5, -1.0, -2.0]:
                gain = sum(r["rev_day"] * 365 * gp_change_rate(r["margin"], p, e) for _, r in branch.iterrows())
                row += f"{gain:>17,.0f}"
            log(row)

        log(f"\nBy branch, +{MAIN_STEP:.0%} (June run-rate):")
        for b, r in branch.iterrows():
            log(f"  {b:<8} revenue ${r['rev_day'] * 365:>9,.0f}/yr, margin {r['margin']:.1%}, break-even e* {-1 / (r['margin'] + MAIN_STEP):+.2f}, "
                f"GP gain @ e=-0.5: ${r['rev_day'] * 365 * gp_change_rate(r['margin'], MAIN_STEP, -0.5):,.0f}/yr")
        log("  Garden has the lowest margin, so it tolerates the MOST price sensitivity before a rise loses money, and")
        log("  +5% at Garden puts it exactly on Harbour's existing price list — a price customers already pay at the main branch.")
        log("\nSupporting (weak, cross-sectional) evidence: Harbour already charges 5.0% more than Garden and 9.4% more than Uptown")
        log("on every item, and its customers buy the same baskets:")
        txb = df.groupby("Transaction ID").agg(b=("Branch", "first"), n=("Product", "size"), q=("Quantity", "sum"))
        log("  items per order " + ", ".join(f"{b} {v:.3f}" for b, v in txb.groupby("b")["n"].mean().items())
            + " | units per order " + ", ".join(f"{b} {v:.3f}" for b, v in txb.groupby("b")["q"].mean().items()))
        log("  This does NOT measure elasticity (each branch has a different location and customer base); it only shows that")
        log("  Stories' customers already accept the Harbour price list at its busiest branch.")

        # =================================================================================
        # STEP 3: STAGED PRICE RISE WITH A STOP RULE — SIMULATED ON STORIES' OWN DAY-TO-DAY NOISE
        # =================================================================================
        log("\n" + "=" * 100)
        log(f"[STEP 3: STAGED +{MAIN_STEP:.0%} RISE — PILOT AT {PILOT_BRANCH.upper()}, THEN ALL BRANCHES — SIMULATED ON REAL NOISE]")
        log("=" * 100)
        units = df.groupby(["Date", "Branch"])["Quantity"].sum().unstack()
        others = [b for b in units.columns if b != PILOT_BRANCH]
        noise = residualize(np.log(units[PILOT_BRANCH] / units[others].sum(axis=1))).values
        sd = noise.std(ddof=1)
        log(f"Metric: daily log(units at {PILOT_BRANCH} / units at {'+'.join(others)}), adjusted for month x weekday.")
        log(f"Its day-to-day noise in Jan-Jun: SD {sd:.3f} (~{sd:.0%}), lag-1 autocorrelation {pd.Series(noise).autocorr(1):+.2f}.")
        be_loss = MAIN_STEP / (branch.loc[PILOT_BRANCH, "margin"] + MAIN_STEP)
        log(f"\nPlan: raise {PILOT_BRANCH} +{MAIN_STEP:.0%}; compare with the 8 weeks before, relative to the other branches; check weekly.")
        log(f"  Stop rule: roll back as soon as we are >=80% confident the volume loss exceeds {PILOT_BRANCH}'s break-even ({be_loss:.1%}).")
        log(f"  At week {MAX_PILOT_WEEKS}: keep {PILOT_BRANCH} if the estimated loss is below its break-even; otherwise roll back.")
        log("  Simulation: daily noise is resampled from the real Jan-Jun residuals; the true effect is injected; repeated "
            f"{N_SIM:,} times per elasticity. 12-month GP is compared with changing nothing.")

        def daily_gain(b, eps):
            return branch.loc[b, "rev_day"] * gp_change_rate(branch.loc[b, "margin"], MAIN_STEP, eps)

        all_day = lambda eps: sum(daily_gain(b, eps) for b in branch.index)
        be_branch = {b: MAIN_STEP / (branch.loc[b, "margin"] + MAIN_STEP) for b in branch.index}
        log("  Extension rule: at week 12 the rise goes to each other branch only if the measured loss is inside THAT branch's")
        log("  break-even (" + ", ".join(f"{b} {v:.1%}" for b, v in be_branch.items()) + "): Garden's low margin makes it the")
        log("  most tolerant branch, so passing at Garden alone is not enough for Harbour or Uptown.")
        log(f"\n{'Elasticity':>10}{'Raise all now':>15}{'Staged plan':>13}{'Staged P5':>11}{'P(keep':>9}{'P(extend':>11}{'Wks to stop':>13}")
        log(f"{'':>10}{'(12-mo GP $)':>15}{'(mean $)':>13}{'(worst 5%)':>11}{'Garden)':>9}{'Harbour)':>11}{'(if stopped)':>13}")
        sim_rows = []
        for eps in ELASTICITY_GRID:
            delta = np.log(1 + eps * MAIN_STEP)
            gains, keeps, stop_weeks = [], 0, []
            ext_counts = Counter()
            for _ in range(N_SIM):
                pre = RNG.choice(noise, PRE_DAYS)
                post = RNG.choice(noise, MAX_PILOT_WEEKS * 7) + delta
                stopped_at = None
                for k in range(1, MAX_PILOT_WEEKS + 1):
                    n = 7 * k
                    est = post[:n].mean() - pre.mean()
                    se = sd * np.sqrt(1 / n + 1 / PRE_DAYS)
                    if est + STOP_Z * se < np.log(1 - be_loss):
                        stopped_at = k
                        break
                final_est = post.mean() - pre.mean()
                if stopped_at is None and final_est < np.log(1 - be_loss):
                    stopped_at = MAX_PILOT_WEEKS
                if stopped_at is None:
                    keeps += 1
                    # extend only to branches whose own break-even the measured loss stays inside
                    kept = [b for b in branch.index if final_est > np.log(1 - be_branch[b])]
                    ext_counts.update(kept)
                    g = daily_gain(PILOT_BRANCH, eps) * MAX_PILOT_WEEKS * 7 + \
                        sum(daily_gain(b, eps) for b in kept) * (365 - MAX_PILOT_WEEKS * 7)
                else:
                    stop_weeks.append(stopped_at)
                    g = daily_gain(PILOT_BRANCH, eps) * stopped_at * 7
                gains.append(g)
            gains = np.array(gains)
            row = dict(eps=eps, now=all_day(eps) * 365, staged=gains.mean(), p5=np.percentile(gains, 5),
                       keep=keeps / N_SIM, wks=np.mean(stop_weeks) if stop_weeks else np.nan,
                       ext_h=ext_counts["Harbour"] / N_SIM)
            sim_rows.append(row)
            wks = f"{row['wks']:.1f}" if stop_weeks and row["keep"] < 0.99 else "-"
            log(f"{eps:>+10.2f}{row['now']:>15,.0f}{row['staged']:>13,.0f}{row['p5']:>11,.0f}{row['keep']:>9.0%}"
                f"{row['ext_h']:>11.0%}{wks:>13}")
        sim = pd.DataFrame(sim_rows)
        worst = sim["staged"].min()
        log(f"\nReading the table:")
        log(f"  * If customers respond like elasticity -0.5, the staged plan earns about ${sim.loc[sim.eps == -0.5, 'staged'].iloc[0]:,.0f} in 12 months.")
        log(f"  * Across every elasticity tested (down to -4, far beyond what any data here suggests), the staged plan's average")
        log(f"    outcome is never worse than a ${-worst:,.0f} loss (worst 5% of runs: ${-sim['p5'].min():,.0f}): the stop rule")
        log(f"    caps the downside at a few weeks of one branch.")
        cost_of_caution = sim.loc[sim.eps == 0.0, "now"].iloc[0] - sim.loc[sim.eps == 0.0, "staged"].iloc[0]
        log(f"  * 'Raise everywhere now' earns ~${cost_of_caution:,.0f} more if customers ignore price (the price of caution), but")
        log(f"    loses ${-sim.loc[sim.eps == -2.0, 'now'].iloc[0]:,.0f}/yr at elasticity -2 and ${-sim.loc[sim.eps == -3.0, 'now'].iloc[0]:,.0f}/yr at -3.")
        log("  * The pilot does NOT measure elasticity precisely; it only answers 'is the loss worse than break-even?',")
        log("    which is the only question the decision needs.")

        log("\nAssumptions & limits:")
        log("  * Revenue base is June's run-rate with no further growth; growth would raise every $ figure proportionally.")
        log("  * Volume response is immediate and stable; slow erosion (customers drifting away over months) would be caught late.")
        log("    Mitigation: keep the weekly comparison running for 6 months after rollout.")
        log("  * The other branches are a valid comparison: no simultaneous promotions, menu changes or local events at Garden.")
        log("  * Customers do not switch branches because of the rise. If Garden customers move to Harbour, Garden's volume")
        log("    falls but chain profit does not; the pilot would read this as price sensitivity and be conservative.")
        log("  * Elasticity is uniform across items. Item-level rises (e.g. only Matcha & Tea, 59-82% margin) need a separate test.")

    # =========================================================================================
    # FIGURE 6
    # =========================================================================================
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT2,
                         "xtick.color": C_TEXT2, "ytick.color": C_TEXT2, "axes.titlecolor": C_TEXT,
                         "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": C_SURF,
                         "axes.facecolor": C_SURF})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)

    # A — value of a price rise vs elasticity, with what the promo can tell us
    ax = axes[0]
    es = np.linspace(-4, 0.5, 200)
    for p, c in zip([0.03, 0.05, 0.08], C_SERIES):
        y = [sum(r["rev_day"] * 365 * gp_change_rate(r["margin"], p, e) for _, r in branch.iterrows()) / 1000 for e in es]
        ax.plot(es, y, color=c, lw=2)
        ax.text(0.55, y[-1], f"+{p:.0%}", color=C_TEXT, va="center", fontsize=9)
        ax.plot([-1 / (m_all + p)], [0], "o", color=c, ms=7, mec=C_SURF, mew=1.5)
    ax.axvspan(eps_ci[2], eps_ci[0], color=C_MUTED, alpha=0.25, lw=0)
    ax.text((eps_ci[2] + eps_ci[0]) / 2, ax.get_ylim()[0] * 0.85, "range the Harbour\npromo cannot rule out",
            ha="center", color=C_TEXT2, fontsize=8.5)
    ax.axhline(0, color=C_TEXT2, lw=0.8)
    ax.set_xlim(-4, 0.9)
    ax.set_xlabel("Customer price elasticity (assumed; not measurable from this data)")
    ax.set_ylabel("Change in annual gross profit ($k, June run-rate)")
    ax.set_title("A. A price rise pays unless customers are very price-sensitive", loc="left")
    ax.text(-3.95, ax.get_ylim()[1] * 0.92, "dots = break-even elasticity", color=C_TEXT2, fontsize=8.5)

    # B — policy comparison
    ax = axes[1]
    ax.plot(sim["eps"], sim["staged"] / 1000, color=C_SERIES[0], lw=2, marker="o", ms=6, mec=C_SURF, mew=1.5,
            label=f"Staged: {PILOT_BRANCH} pilot + stop rule (mean of {N_SIM:,} runs)")
    ax.plot(sim["eps"], sim["now"] / 1000, color=C_SERIES[1], lw=2, marker="o", ms=6, mec=C_SURF, mew=1.5,
            label="Raise +5% everywhere now")
    ax.axhline(0, color=C_TEXT2, lw=1.2, ls="--", label="Do nothing")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlabel("True customer price elasticity")
    ax.set_ylabel("12-month gross profit vs doing nothing ($k)")
    ax.set_title("B. The staged plan keeps the upside and caps the downside", loc="left")

    # C — chance the pilot keeps the price rise
    ax = axes[2]
    ax.plot(sim["eps"], sim["keep"] * 100, color=C_SERIES[0], lw=2, marker="o", ms=6, mec=C_SURF, mew=1.5)
    be_eps = -1 / (branch.loc[PILOT_BRANCH, "margin"] + MAIN_STEP)
    ax.axvline(be_eps, color=C_TEXT2, lw=1, ls=":")
    ax.text(be_eps + 0.05, 50, f"{PILOT_BRANCH} break-even\n(e = {be_eps:.2f})", color=C_TEXT2, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("True customer price elasticity")
    ax.set_ylabel(f"Chance the {MAX_PILOT_WEEKS}-week pilot keeps the rise (%)")
    ax.set_title("C. The pilot keeps good rises and stops bad ones", loc="left")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_pricing_decision.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    run_pricing_decision()
    print(f"Done. Report: {OUTPUT_DIR / '10_pricing_decision_report.txt'} | Figure: {FIG_DIR / 'fig6_pricing_decision.png'}")
