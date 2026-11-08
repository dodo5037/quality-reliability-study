"""Quality + reliability study on the AI4I 2020 predictive maintenance dataset (UCI).

The dataset is SYNTHETIC: UCI describes it as reflecting real predictive-maintenance data
from industry. 10,000 machine runs, 5 failure modes:
  TWF tool wear failure · HDF heat dissipation failure · PWF power failure ·
  OSF overstrain failure · RNF random failure

Steps
  1. Pareto of failure modes and failure rate by product quality (L / M / H)
  2. Root cause: engineered physical features (temperature gap, power, strain) and the
     thresholds where each failure mode starts, read from the data with shallow decision trees
  3. SPC: individuals control chart of torque and a p-chart of failure rate per batch
  4. FMEA: severity and detection are judgment calls (stated), occurrence comes from the data
  5. Prediction: which machine runs will fail? Raw sensor features vs engineered features,
     stratified 5-fold cross-validation
  6. A tool-replacement policy: how many wear-related failures happen past each wear limit
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
T, CH = ROOT / "outputs" / "tables", ROOT / "outputs" / "charts"
T.mkdir(parents=True, exist_ok=True); CH.mkdir(parents=True, exist_ok=True)
NAVY, TEAL, PINK, GREY, AMBER = "#1f3864", "#10b981", "#f43f5e", "#9aa5b1", "#f59e0b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
MODES = {"HDF": "Heat dissipation", "OSF": "Overstrain", "PWF": "Power", "TWF": "Tool wear", "RNF": "Random"}


def load():
    d = pd.read_csv(RAW, encoding="utf-8-sig")
    d = d.rename(columns={"Air temperature [K]": "air_K", "Process temperature [K]": "process_K",
                          "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque_Nm",
                          "Tool wear [min]": "wear_min", "Machine failure": "failure"})
    d["temp_gap_K"] = d["process_K"] - d["air_K"]
    d["power_W"] = d["torque_Nm"] * d["rpm"] * 2 * np.pi / 60
    d["strain_minNm"] = d["wear_min"] * d["torque_Nm"]
    d["type_code"] = d["Type"].map({"L": 0, "M": 1, "H": 2})
    return d


def pareto(d):
    counts = d[list(MODES)].sum().sort_values(ascending=False)
    share = counts / counts.sum()
    out = pd.DataFrame({"mode": counts.index, "name": [MODES[m] for m in counts.index], "count": counts.values,
                        "share": share.values, "cum_share": share.cumsum().values})
    out.to_csv(T / "pareto_failure_modes.csv", index=False)
    by_type = d.groupby("Type")["failure"].agg(["count", "sum", "mean"]).reindex(["L", "M", "H"])
    by_type.to_csv(T / "failure_rate_by_type.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(out["name"], out["count"], color=[PINK, PINK, PINK, GREY, GREY])
    ax2 = ax.twinx(); ax2.plot(out["name"], out["cum_share"] * 100, color=NAVY, marker="o")
    ax2.set_ylim(0, 105); ax2.set_ylabel("Cumulative %"); ax2.spines["top"].set_visible(False)
    for i, (c, cs) in enumerate(zip(out["count"], out["cum_share"])):
        ax.text(i, c + 2, str(c), ha="center", fontsize=9)
    ax.set_ylabel("Failures"); ax.set_title("Pareto of failure modes (10,000 machine runs)")
    fig.tight_layout(); fig.savefig(CH / "01_pareto.png"); plt.close(fig)
    return out, by_type


def root_causes(d):
    """For each mode, fit a depth-2 tree on physical features and report the split thresholds."""
    feats = {"HDF": ["temp_gap_K", "rpm"], "PWF": ["power_W"], "OSF": ["strain_minNm", "type_code"], "TWF": ["wear_min"]}
    rows = []
    for m, f in feats.items():
        tree = DecisionTreeClassifier(max_depth=2 if m != "OSF" else 3, class_weight="balanced", random_state=0).fit(d[f], d[m])
        pred = tree.predict(d[f])
        rows.append({"mode": m, "name": MODES[m], "features": " + ".join(f),
                     "recall": recall_score(d[m], pred), "precision": precision_score(d[m], pred, zero_division=0),
                     "rule": export_text(tree, feature_names=f, decimals=1).replace("\n", " | ")})
    hdf = d[d.HDF == 1]
    pwf = d[d.PWF == 1]["power_W"]
    osf = d[d.OSF == 1].groupby("Type")["strain_minNm"].min()
    twf = d[d.TWF == 1]["wear_min"]
    facts = {
        "HDF": f"All {len(hdf)} heat failures have temperature gap < 8.6 K AND speed < 1,380 rpm (max seen: {hdf.temp_gap_K.max():.2f} K, {hdf.rpm.max()} rpm)",
        "PWF": f"Power failures sit at the extremes: {int((pwf < 3500).sum())} below 3,500 W and {int((pwf > 9000).sum())} above 9,000 W (normal runs average {d.power_W.mean():,.0f} W)",
        "OSF": "Overstrain starts at wear x torque of about " + ", ".join(f"{t}: {v:,.0f}" for t, v in osf.reindex(['L', 'M', 'H']).items()) + " min*Nm: lower-quality products fail sooner",
        "TWF": f"Tool wear failures happen between {twf.min():.0f} and {twf.max():.0f} min of tool wear (median {twf.median():.0f})",
        "RNF": f"{int(d.RNF.sum())} random failures (0.1% chance per run, no sensor pattern): not preventable by monitoring",
    }
    pd.DataFrame(rows).to_csv(T / "root_cause_trees.csv", index=False)
    pd.Series(facts).to_csv(T / "root_cause_facts.csv", header=["finding"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ok = d[d.failure == 0].sample(2500, random_state=0)
    ax = axes[0]
    ax.scatter(ok.rpm, ok.temp_gap_K, s=4, color=GREY, alpha=0.4, label="no failure (sample)")
    ax.scatter(hdf.rpm, hdf.temp_gap_K, s=12, color=PINK, label="heat dissipation failure")
    ax.axhline(8.6, color=NAVY, ls="--", lw=1); ax.axvline(1380, color=NAVY, ls="--", lw=1)
    ax.set_xlabel("Rotational speed (rpm)"); ax.set_ylabel("Process - air temperature (K)"); ax.set_title("HDF: slow AND poorly cooled"); ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    ax.hist(d[d.failure == 0].power_W, bins=60, color=GREY, alpha=0.7, label="no failure")
    ax.hist(d[d.PWF == 1].power_W, bins=30, color=PINK, label="power failure")
    ax.axvline(3500, color=NAVY, ls="--", lw=1); ax.axvline(9000, color=NAVY, ls="--", lw=1)
    ax.set_xlabel("Power = torque x speed (W)"); ax.set_title("PWF: power outside 3.5-9 kW"); ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    for t, c in zip(["L", "M", "H"], [PINK, AMBER, TEAL]):
        g = d[(d.Type == t)]
        ax.scatter(g[g.OSF == 1].wear_min, g[g.OSF == 1].torque_Nm, s=14, color=c, label=f"overstrain, type {t}")
    ax.scatter(ok.wear_min, ok.torque_Nm, s=3, color=GREY, alpha=0.3)
    ax.set_xlabel("Tool wear (min)"); ax.set_ylabel("Torque (Nm)"); ax.set_title("OSF: worn tool x high torque"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(CH / "02_root_causes.png"); plt.close(fig)
    return facts


def spc(d):
    """Individuals (I-MR) chart of torque in run order, and a p-chart of failure rate per batch of 250 runs."""
    x = d["torque_Nm"].values
    mr = np.abs(np.diff(x)); mrbar = mr.mean()
    center = x.mean(); sigma = mrbar / 1.128
    ucl, lcl = center + 3 * sigma, center - 3 * sigma
    out_i = int(((x > ucl) | (x < lcl)).sum())

    batch = 250
    b = d.assign(batch=np.arange(len(d)) // batch).groupby("batch")["failure"].agg(["sum", "count"])
    b["p"] = b["sum"] / b["count"]
    pbar = b["sum"].sum() / b["count"].sum()
    p_ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / batch); p_lcl = max(0.0, pbar - 3 * np.sqrt(pbar * (1 - pbar) / batch))
    out_p = int(((b["p"] > p_ucl) | (b["p"] < p_lcl)).sum())
    res = {"torque_center": center, "torque_ucl": ucl, "torque_lcl": lcl, "torque_points_outside": out_i,
           "torque_share_outside": out_i / len(x), "p_bar": pbar, "p_ucl": p_ucl, "p_lcl": p_lcl,
           "batches": len(b), "batches_outside": out_p}
    pd.Series(res).to_csv(T / "spc_summary.csv", header=["value"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2))
    ax = axes[0]
    ax.plot(x[:1000], color=NAVY, lw=0.6)
    for y, c in [(center, TEAL), (ucl, PINK), (lcl, PINK)]:
        ax.axhline(y, color=c, ls="--", lw=1)
    sel = np.where((x[:1000] > ucl) | (x[:1000] < lcl))[0]
    ax.scatter(sel, x[:1000][sel], color=PINK, zorder=3, s=14)
    ax.set_title(f"Individuals chart: torque, first 1,000 runs (limits {lcl:.1f}-{ucl:.1f} Nm; {out_i} of 10,000 points outside)")
    ax.set_ylabel("Torque (Nm)")
    ax = axes[1]
    ax.plot(b.index, b["p"] * 100, marker="o", color=NAVY, lw=1)
    ax.axhline(pbar * 100, color=TEAL, ls="--"); ax.axhline(p_ucl * 100, color=PINK, ls="--"); ax.axhline(p_lcl * 100, color=PINK, ls="--")
    ax.set_title(f"p-chart: failure rate per batch of {batch} runs ({out_p} of {len(b)} batches outside limits)")
    ax.set_xlabel("Batch"); ax.set_ylabel("Failure rate (%)")
    fig.tight_layout(); fig.savefig(CH / "03_spc.png"); plt.close(fig)
    return res


def special_cause(d, batch=250):
    """Investigate the out-of-control batches on the p-chart: what changed in those runs?"""
    b = d.assign(batch=np.arange(len(d)) // batch)
    pbar = d["failure"].mean()
    ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / batch)
    rate = b.groupby("batch")["failure"].mean()
    flagged = rate[rate > ucl].index.tolist()
    span = b["batch"].between(min(flagged), max(flagged))
    cols = ["air_K", "process_K", "temp_gap_K", "rpm", "torque_Nm"]
    comp = pd.DataFrame({"flagged period": b.loc[span, cols].mean(), "rest": b.loc[~span, cols].mean()})
    modes = pd.DataFrame({"flagged period": b.loc[span, list(MODES)].sum(), "rest": b.loc[~span, list(MODES)].sum()})
    comp.to_csv(T / "special_cause_conditions.csv"); modes.to_csv(T / "special_cause_modes.csv")
    return {"batches": flagged, "runs": (int(span.sum())), "first_run": int(b.loc[span].index.min()) + 1,
            "last_run": int(b.loc[span].index.max()) + 1, "conditions": comp, "modes": modes,
            "hdf_share_in_period": float(modes.loc["HDF", "flagged period"] / modes.loc["HDF"].sum())}


def fmea(d):
    """Severity (S) and detection (D) are engineering judgment on a 1-10 scale (stated assumptions).
    Occurrence (O) is scaled from the observed rate per 10,000 runs."""
    judgment = {
        "HDF": (7, 3, "Overheating stops the machine; scrap risk", "Temperature gap and speed sensors"),
        "OSF": (8, 3, "Tool breaks under load; part scrapped", "Wear x torque tracked per product type"),
        "PWF": (7, 3, "Motor stalls or overloads", "Power = torque x speed monitored"),
        "TWF": (6, 5, "Worn tool fails; part scrapped, tool changed", "Wear counter, but failure time varies 198-253 min"),
        "RNF": (6, 10, "Unexplained failure", "No sensor signal; cannot be detected in advance"),
    }
    rows = []
    for m, (s, det, effect, control) in judgment.items():
        rate = d[m].mean()
        o = int(np.clip(np.ceil(rate * 10_000 / 15), 1, 10))   # 1 per 15 failures per 10k runs, capped at 10
        rows.append({"mode": m, "failure_mode": MODES[m], "effect": effect, "current_control": control,
                     "count": int(d[m].sum()), "S": s, "O": o, "D": det, "RPN": s * o * det})
    f = pd.DataFrame(rows).sort_values("RPN", ascending=False)
    f.to_csv(T / "fmea.csv", index=False)
    return f


def predict(d):
    y = d["failure"].values
    raw = ["type_code", "air_K", "process_K", "rpm", "torque_Nm", "wear_min"]
    eng = raw + ["temp_gap_K", "power_W", "strain_minNm"]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    models = {
        "Logistic regression, raw sensors": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), raw),
        "Logistic regression, + engineered": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), eng),
        "Decision tree (depth 5), raw sensors": (DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0), raw),
        "Decision tree (depth 5), + engineered": (DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0), eng),
    }
    rows = []
    for name, (model, feats) in models.items():
        pred = cross_val_predict(model, d[feats], y, cv=cv)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append({"model": name, "recall": recall_score(y, pred), "precision": precision_score(y, pred),
                     "f1": f1_score(y, pred), "caught": int(tp), "missed": int(fn), "false_alarms": int(fp)})
    r = pd.DataFrame(rows)
    r.to_csv(T / "prediction_cv.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(r)); w = 0.38
    ax.bar(x - w / 2, r["recall"] * 100, w, color=TEAL, label="Recall (failures caught)")
    ax.bar(x + w / 2, r["precision"] * 100, w, color=NAVY, label="Precision (alarms that were real)")
    for i, (rc, pr) in enumerate(zip(r["recall"], r["precision"])):
        ax.text(i - w / 2, rc * 100 + 1, f"{rc:.0%}", ha="center", fontsize=8)
        ax.text(i + w / 2, pr * 100 + 1, f"{pr:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [m.replace(", ", "\n") for m in r["model"]], fontsize=8)
    ax.set_ylim(0, 110); ax.set_title("Predicting failures (5-fold cross-validation)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(CH / "04_prediction.png"); plt.close(fig)
    return r


def tool_policy(d):
    """Share of wear-driven failures (TWF + OSF) that happened at or above each wear limit.
    If tools were replaced at that limit, these failures would not have happened on a worn tool.
    Upper-bound estimate: replacing tools earlier also costs more tools."""
    wear_fail = d[(d.TWF == 1) | (d.OSF == 1)]
    rows = []
    for limit in [150, 175, 200, 225, 250]:
        avoid = int((wear_fail.wear_min >= limit).sum())
        rows.append({"replace_at_min": limit, "wear_failures_at_or_past_limit": avoid,
                     "share_of_wear_failures": avoid / len(wear_fail),
                     "runs_past_limit_share": float((d.wear_min >= limit).mean())})
    p = pd.DataFrame(rows)
    p.to_csv(T / "tool_policy.csv", index=False)
    return p, len(wear_fail)


if __name__ == "__main__":
    d = load()
    print(f"Runs: {len(d):,} | failures: {d.failure.sum()} ({d.failure.mean():.2%})")
    par, by_type = pareto(d)
    print("\nPareto:\n", par.round(3).to_string(index=False))
    print("\nFailure rate by type:\n", by_type.round(4).to_string())
    facts = root_causes(d)
    print("\nRoot causes:"); [print(" -", k, v) for k, v in facts.items()]
    s = spc(d)
    print("\nSPC:", {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()})
    sc = special_cause(d)
    print(f"\nSpecial cause: batches {sc['batches']} = runs {sc['first_run']}-{sc['last_run']}; "
          f"{sc['hdf_share_in_period']:.0%} of heat failures happened there")
    print(sc["conditions"].round(2).to_string()); print(sc["modes"].to_string())
    print("\nFMEA:\n", fmea(d)[["mode", "S", "O", "D", "RPN", "count"]].to_string(index=False))
    print("\nPrediction (5-fold CV):\n", predict(d).round(3).to_string(index=False))
    p, n = tool_policy(d)
    print(f"\nTool policy ({n} wear-driven failures):\n", p.round(3).to_string(index=False))
