"""Confirms the demo's code reproduces analysis.ipynb: every model's test score for RELIANCE is compared with results.json."""
from . import pipeline as P
from . import models as M
import sys, time
import numpy as np, pandas as pd

ACC_TOL, AUC_TOL, RMSE_TOL = 1e-9, 1e-6, 1e-4


def main():
    t0 = time.time()
    store = P.Store(); R = store.results
    D = store.frame("RELIANCE"); tr, te = M.split(D, P.TEST_START)
    problems = []
    if not (len(tr) == R["split"]["n_train"] and len(te) == R["split"]["n_test"]):
        problems.append(f"row counts {len(tr)} and {len(te)}")
    print(f"rows: train {len(tr)}, test {len(te)}")

    # the features must not change when the data after a date is removed
    worst = 0.0
    for d in pd.to_datetime(np.random.default_rng(P.SEED).choice(te.index.values, 5, replace=False)):
        cut = store.panel[store.panel.Date <= d]
        F_cut = P.build_features(P.clean_panel(cut), "RELIANCE")[0].loc[d, P.ENG].values.astype(float)
        worst = max(worst, float(np.nanmax(np.abs(F_cut - D.loc[d, P.ENG].values.astype(float)))))
    print(f"largest feature difference after removing the future: {worst:.2e}")
    if worst > 1e-9: problems.append(f"feature leak {worst}")

    b = M.fit_bundle(store, "RELIANCE", P.TEST_START, progress=lambda k: print("  fitting", k, f"{time.time() - t0:.1f}s", flush=True))
    x = b["design"].x(te); y = te.y_dir.values.astype(int); close_t = te.close_t.values; truth = te.close_next.values

    saved = {r["key"]: r for r in R["u2_clf"]["table"] + R["u5"]["leaderboard"]}
    print(f"\n{'direction model':30s} {'accuracy':>9s} {'notebook':>9s} {'AUC':>8s} {'notebook':>9s}")
    for key in M.DIRECTION:
        yhat, score, _ = M.predict_direction(key, b["direction"][key], x)
        s = M.score_direction(yhat, score, y); n = saved[key]
        ok = abs(s["accuracy"] - n["accuracy"]) < ACC_TOL and (n.get("auc") is None or s["auc"] is None or abs(s["auc"] - n["auc"]) < AUC_TOL)
        print(f"{key:30s} {s['accuracy']:9.5f} {n['accuracy']:9.5f} {str(None if s['auc'] is None else round(s['auc'], 5)):>8s} {str(None if n.get('auc') is None else round(n['auc'], 5)):>9s} {'' if ok else '  MISMATCH'}")
        if not ok: problems.append(key)

    reg = {r["key"]: r for r in R["u2_reg"]["table"]}
    print(f"\n{'price model':30s} {'RMSE':>9s} {'notebook':>9s}")
    for key in M.PRICE:
        p = M.predict_price(key, b["price"][key], x, close_t)[0]
        e = np.sqrt(np.mean((truth - p) ** 2)); ok = abs(e - reg[key]["rmse"]) < RMSE_TOL
        print(f"{key:30s} {e:9.4f} {reg[key]['rmse']:9.4f} {'' if ok else '  MISMATCH'}")
        if not ok: problems.append(key)

    kf = {r["key"]: r for r in R["u4"]["kalman"]["forecast"]}
    prices = store.series("RELIANCE").Adj
    for key, kind in [("kalman_ll", "ll"), ("kalman_llt", "llt")]:
        fc = pd.Series(M.kalman_price_forecast(b["price"][key], prices.values), index=prices.index).reindex(te.index).values
        e = np.sqrt(np.mean((truth - fc) ** 2)); ok = abs(e - kf[kind]["rmse"]) < RMSE_TOL
        print(f"{key:30s} {e:9.4f} {kf[kind]['rmse']:9.4f} {'' if ok else '  MISMATCH'}")
        if not ok: problems.append(key)

    h = b["hmm"]; rets = store.series("RELIANCE").r.dropna() * 100
    filt = M.hmm_filter(rets.values, h["pi"], h["A"], h["mu"], h["var"])
    p_up = pd.Series(M.hmm_forecast(filt, h["A"], h["mu"], h["var"])[0], index=rets.index).reindex(te.index).values
    acc = float(np.mean((p_up > .5).astype(int) == (te.r_next.values > 0).astype(int))); nb = R["u4"]["forecast"]["reliance"]["filtered"]["accuracy"]
    ok = abs(acc - nb) < ACC_TOL
    print(f"\nHMM filtered accuracy {acc:.5f}, notebook {nb:.5f} {'' if ok else '  MISMATCH'}")
    if not ok: problems.append("hmm")

    M.save_bundle(b)
    print(f"\ntotal {time.time() - t0:.1f}s; cached {M.cache_path('RELIANCE', P.TEST_START).name}, {M.cache_path('RELIANCE', P.TEST_START).stat().st_size / 1e6:.1f} MB")
    if problems:
        print("MISMATCHES:", problems); sys.exit(1)
    print("OK: every model matches the notebook.")


if __name__ == "__main__":
    main()
