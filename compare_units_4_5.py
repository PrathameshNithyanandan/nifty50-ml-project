"""Compares the Unit 4 and Unit 5 algorithms on the same 822 RELIANCE test days, from the notebook's saved results.
Writes results/units_4_5_comparison.md.

    python3 compare_units_4_5.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def num(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main():
    with open(os.path.join(ROOT, "results", "results.json")) as f:
        R = json.load(f)
    u4, u5 = R["u4"], R["u5"]
    fc = u4["forecast"]["reliance"]
    hmm = dict(fc["filtered"], model="HMM, filtered probabilities", unit="4", group="hidden Markov model")
    rows = [hmm] + [dict(r, unit="5" if r["group"] != "baseline" else "baseline") for r in u5["leaderboard"]]
    n, maj = hmm["n"], u5["majority_rate"]
    assert fc["majority_rate"] == maj and all(r["n_test"] == n for r in u5["leaderboard"] if "n_test" in r), "the test sets differ"
    models = [r for r in rows if r["unit"] != "baseline"]
    rank = sorted(rows, key=lambda r: -r["accuracy"])
    best = max(models, key=lambda r: r["accuracy"])
    best_auc = max((r for r in models if r["auc"] is not None), key=lambda r: r["auc"])
    n_sig = sum(r["p_vs_majority"] < 0.05 for r in models)
    n_above = sum(r["accuracy"] > maj for r in models)
    band = (u5["auc_low"], u5["auc_high"])
    kal = u4["kalman"]["forecast"]
    per = next(r for r in kal if r["key"] == "persistence")

    L = ["# Unit 4 and Unit 5 algorithms compared\n",
         f"Written by `compare_units_4_5.py` from `results/results.json`. Every model forecasts the same {n} RELIANCE test days. All of them train on days before the test period only.\n",
         "## Next-day direction\n",
         f"The baseline is always predicting the majority class, which is right {100 * maj:.2f}% of the time. An AUC between {band[0]:.3f} and {band[1]:.3f} is within the chance band.\n",
         "| Rank | Algorithm | Unit | Accuracy | AUC | p against majority |", "|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(rank, 1):
        L.append(f"| {i} | {r['model']} | {r['unit']} | {num(r['accuracy'])} | {num(r['auc'])} | {num(r['p_vs_majority'], 3)} |")
    L += ["", f"**Result:** Algorithms above the majority rate: {n_above} of {len(models)}. Significantly above it at the 5% level: **{n_sig} of {len(models)}**. "
          f"The highest accuracy is {best['model']} (Unit {best['unit']}) at {100 * best['accuracy']:.2f}%, and the highest AUC is {best_auc['model']} (Unit {best_auc['unit']}) at {best_auc['auc']:.4f}. "
          "Neither unit's algorithms show a dependable edge on this stock.\n",
          f"The HMM forecast uses filtered probabilities, which only look at past days. The smoothed version scores {100 * fc['smoothed_wrong']['accuracy']:.2f}%, but it uses future days, so it is left out of the ranking.\n",
          "## Next-day closing price\n",
          "Only Unit 4 forecasts the price: the Unit 5 models predict direction. So the Kalman filter is compared with persistence, which means using today's close as tomorrow's forecast.\n",
          "| Model | RMSE | MAE | Diebold–Mariano p against persistence |", "|---|---:|---:|---:|"]
    for r in kal:
        L.append(f"| {r['model']} | {num(r['rmse'])} | {num(r['mae'])} | {'baseline' if r['dm_p'] is None else num(r['dm_p'], 3)} |")
    ll, llt = (next(r for r in kal if r["key"] == k) for k in ("ll", "llt"))
    L += ["", f"**Result:** neither fitted Kalman model is significantly different from persistence (p = {ll['dm_p']:.3f} and {llt['dm_p']:.3f}; RMSE {ll['rmse']:.3f} and {llt['rmse']:.3f} against {per['rmse']:.3f}). "
          f"Maximum likelihood sets the local-level Kalman gain to {ll['gain']:.4f}, which makes that model almost the same as persistence. The filter with a gain fixed by hand looks smoother but forecasts far worse.\n"]
    with open(os.path.join(ROOT, "results", "units_4_5_comparison.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
