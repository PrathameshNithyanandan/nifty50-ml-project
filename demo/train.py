"""Fits and saves the models so the demo answers at once. Without --all it covers the stocks the notebook studied."""
from . import pipeline as P
from . import models as M
import argparse, time

CORE = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="fit every stock, not only the four the notebook studied")
    ap.add_argument("--refit", action="store_true", help="ignore saved fits")
    a = ap.parse_args()
    store = P.Store(); t0 = time.time()
    names = store.tickers if a.all else CORE
    for i, tk in enumerate(names, 1):
        for cutoff in (P.TEST_START, P.ALL_DATA_CUTOFF):
            if not a.refit and M.load_bundle(store, tk, cutoff, fit_if_missing=False) is not None:
                continue
            t1 = time.time()
            b = M.fit_bundle(store, tk, cutoff)
            M.save_bundle(b)
            g = b["direction"]["gam"]
            print(f"[{i}/{len(names)}] {tk}, trained before {cutoff.date()}: {time.time() - t1:.1f}s; GAM λ {None if g is None else g.lam_used}; failed {b.get('failed', {})}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s. Saved fits: {len(list(P.CACHE_DIR.glob('*.joblib')))}.")


if __name__ == "__main__":
    main()
