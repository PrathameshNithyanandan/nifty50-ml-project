"""Local server for the live demo: runs the models on request and serves the built site."""
from . import pipeline as P
from . import models as M
from . import market
import argparse, math, threading, time
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

DIST = P.ROOT / "web" / "dist"
HOST = "127.0.0.1"
HISTORY_DAYS = 60
MIN_TRAIN, MIN_TEST = 500, 100
CUTOFF_RANGE = ("2008-01-01", "2020-06-30")
REG_RANGE = (1e-4, 1e8)
DEPTH_RANGE = (1, 30)
SLIDERS = ["r_lag0", "mkt_r", "oc_ret", "rstd_20", "mom_60", "dist_sma50", "vol_ratio20", "hl_range"]
DOW = {"mon": None, "tue": "dow_tue", "wed": "dow_wed", "thu": "dow_thu", "fri": "dow_fri"}

app = Flask(__name__, static_folder=None)
store = None
_bundles, _lock = {}, threading.RLock()


class BadRequest(Exception):
    pass


@app.errorhandler(BadRequest)
def _bad(e): return jsonify(error=str(e)), 400


@app.errorhandler(RuntimeError)
def _unavailable(e): return jsonify(error=str(e)), 503


@app.errorhandler(Exception)
def _other(e):
    if isinstance(e, HTTPException): return jsonify(error=e.description), e.code
    app.logger.exception(e)
    return jsonify(error="The server could not complete this request. Its terminal window has the details."), 500


def num(x):
    if x is None: return None
    v = float(x)
    return v if math.isfinite(v) else None


def date_str(d): return str(pd.Timestamp(d).date())


def body(): return request.get_json(silent=True) or {}


def ticker_arg(v):
    if v not in store.tickers: raise BadRequest(f"Unknown stock: {v}")
    return v


def bundle(tk, cutoff=P.TEST_START):
    k = (tk, str(pd.Timestamp(cutoff).date()))
    with _lock:
        if k not in _bundles:
            _bundles[k] = M.load_bundle(store, tk, cutoff)
        return _bundles[k]


def test_row(tk, date_):
    D = store.frame(tk)
    try: d = pd.Timestamp(date_)
    except (ValueError, TypeError): raise BadRequest(f"Not a date: {date_}")
    if d < P.TEST_START or d not in D.index: raise BadRequest(f"{tk} has no test day on {date_}. Pick a trading day from {date_str(P.TEST_START)} to {date_str(D.index.max())}.")
    return d, D.loc[[d]]


# ---------- running every model on some rows ----------
def direction_rows(b, x, keys=None):
    out = []
    for key, spec in M.DIRECTION.items():
        if keys and key not in keys or b["direction"][key] is None: continue
        yhat, score, prob = M.predict_direction(key, b["direction"][key], x)
        out.append(dict(key=key, label=spec["label"], group=spec["group"], call="up" if yhat[0] == 1 else "down", prob=num(prob[0]) if prob is not None else None, score=num(score[0]) if score is not None else None))
    return out


def price_rows(b, x, close_t, keys=None):
    out = []
    for key, spec in M.PRICE.items():
        if keys and key not in keys: continue
        p, lo, hi = M.predict_price(key, b["price"][key], x, close_t)
        out.append(dict(key=key, label=spec["label"], group=spec["group"], forecast=num(p[0]), lo=num(lo[0]) if lo is not None else None, hi=num(hi[0]) if hi is not None else None))
    return out


def kalman_forecasts(b, prices):
    return {key: M.kalman_price_forecast(b["price"][key], prices.values) for key in M.KALMAN}


def hmm_block(h, rets, start=None):
    pi = h["pi"] if start is None else start
    filt = M.hmm_filter(rets.values, pi, h["A"], h["mu"], h["var"])
    p_up, mean, sd = M.hmm_forecast(filt, h["A"], h["mu"], h["var"])
    return filt, p_up, mean, sd


def hmm_row(h, filt, p_up, mean, sd, i):
    return dict(names=h["names"], state=[num(v) for v in filt[i]], current=h["names"][int(filt[i].argmax())], p_up=num(p_up[i]), mean=num(mean[i]), sd=num(sd[i]), call="up" if p_up[i] > .5 else "down")


def features_of(row):
    return {c: num(row[c].iloc[0]) for c in P.ENG}


# ---------- routes ----------
@app.get("/api/meta")
def meta():
    R = store.results
    return jsonify(
        stocks=store.tickers, default_stock="RELIANCE", test_start=date_str(P.TEST_START), data_end=date_str(store.panel.Date.max()),
        direction=[dict(key=k, label=v["label"], group=v["group"], slow=v.get("slow", False)) for k, v in M.DIRECTION.items()],
        price=[dict(key=k, label=v["label"], group=v["group"], slow=False) for k, v in {**M.PRICE, **M.KALMAN}.items()],
        sliders=SLIDERS, cutoff_range=CUTOFF_RANGE, reg_range=REG_RANGE, depth_range=DEPTH_RANGE, max_run_days=len(store.frame("RELIANCE")[lambda d: d.index >= P.TEST_START]),
        notebook=dict(majority_rate=R["u2_clf"]["maj_rate"], persistence_rmse=next(r["rmse"] for r in R["u2_reg"]["table"] if r["key"] == "persistence"), n_test=R["split"]["n_test"]))


@app.get("/api/stock")
def stock():
    tk = ticker_arg(request.args.get("ticker"))
    D = store.frame(tk); te = D[D.index >= P.TEST_START]
    b = bundle(tk)
    r = te.r_next
    return jsonify(ticker=tk, dates=[date_str(d) for d in te.index], first=date_str(te.index.min()), last=date_str(te.index.max()), n_train=b["n_train"], train_end=b["train_end"], ranges=b["ranges"],
                   highlights=dict(biggest_fall=date_str(r.idxmin()), biggest_rise=date_str(r.idxmax()), latest=date_str(te.index.max())))


@app.post("/api/replay")
def replay():
    q = body(); tk = ticker_arg(q.get("ticker"))
    d, row = test_row(tk, q.get("date"))
    t0 = time.perf_counter()
    b = bundle(tk); x = b["design"].x(row); close_t = row.close_t.values
    dirs = direction_rows(b, x)
    prices = price_rows(b, x, close_t)
    s = store.series(tk); adj = s.Adj
    i_price = adj.index.get_loc(d)
    kf = kalman_forecasts(b, adj.iloc[:i_price + 1])          # only prices up to the chosen day go in
    for key, spec in M.KALMAN.items():
        prices.append(dict(key=key, label=spec["label"], group=spec["group"], forecast=num(kf[key][-1]), lo=None, hi=None))
    rets = s.r.loc[:d].dropna() * 100
    filt, p_up, mean, sd = hmm_block(b["hmm"], rets)
    hmm = hmm_row(b["hmm"], filt, p_up, mean, sd, -1)
    hist = adj.iloc[max(0, i_price - HISTORY_DAYS + 1):i_price + 1]
    out = dict(ticker=tk, date=date_str(d), close=num(close_t[0]), direction=dirs, price=prices, hmm=hmm, features=features_of(row), history=dict(date=[date_str(i) for i in hist.index], close=[num(v) for v in hist.values]))
    out["failed"] = sorted(b.get("failed", {}))
    out["n_up"] = sum(r["call"] == "up" for r in dirs if r["group"] != "baseline")
    out["n_models"] = sum(r["group"] != "baseline" for r in dirs)
    if q.get("reveal") is True:
        up = bool(row.r_next.iloc[0] > 0); truth = num(row.close_next.iloc[0])
        out["actual"] = dict(date=date_str(adj.index[i_price + 1]), close=truth, r=num(row.r_next.iloc[0]), direction="up" if up else "down")
        for r in dirs: r["correct"] = (r["call"] == "up") == up
        hmm["correct"] = (hmm["call"] == "up") == up
        for r in prices: r["error"] = None if r["forecast"] is None else abs(r["forecast"] - truth)
    out["seconds"] = time.perf_counter() - t0
    return jsonify(out)


@app.post("/api/replay/run")
def replay_run():
    q = body(); tk = ticker_arg(q.get("ticker"))
    d, _ = test_row(tk, q.get("date"))
    try: days = int(q.get("days"))
    except (TypeError, ValueError): raise BadRequest("days must be a whole number")
    D = store.frame(tk); te = D[D.index >= d].iloc[:days]
    if days < 1 or days > len(store.frame(tk)): raise BadRequest("days is out of range")
    t0 = time.perf_counter()
    b = bundle(tk); x = b["design"].x(te); y = te.y_dir.values.astype(int); n = np.arange(1, len(te) + 1)
    series = []
    for key, spec in M.DIRECTION.items():
        if b["direction"][key] is None: continue
        yhat, score, _ = M.predict_direction(key, b["direction"][key], x)
        hit = yhat == y
        series.append(dict(key=key, label=spec["label"], group=spec["group"], running=[num(v) for v in np.cumsum(hit) / n], accuracy=num(hit.mean()), n_correct=int(hit.sum())))
    maj = next(r for r in series if r["key"] == "majority")["accuracy"]
    s = store.series(tk)
    rets = s.r.dropna() * 100; filt, p_up, _, _ = hmm_block(b["hmm"], rets)
    hit = (pd.Series(p_up, index=rets.index).reindex(te.index).values > .5).astype(int) == y
    series.append(dict(key="hmm", label="HMM (filtered)", group="regime", running=[num(v) for v in np.cumsum(hit) / n], accuracy=num(hit.mean()), n_correct=int(hit.sum())))
    for r in series: r["p_vs_majority"] = None if r["group"] == "baseline" else M.binom_p(r["n_correct"], len(te), maj)
    close_t, truth = te.close_t.values, te.close_next.values
    prices = []
    base_err = truth - close_t
    for key, spec in M.PRICE.items():
        p = M.predict_price(key, b["price"][key], x, close_t)[0]; e = truth - p
        prices.append(dict(key=key, label=spec["label"], group=spec["group"], rmse=num(np.sqrt(np.mean(e ** 2))), dm_p=None if key == "persistence" or len(e) < 20 else M.dm_test(e, base_err)))
    adj = s.Adj
    for key, spec in M.KALMAN.items():
        p = pd.Series(M.kalman_price_forecast(b["price"][key], adj.values), index=adj.index).reindex(te.index).values; e = truth - p
        prices.append(dict(key=key, label=spec["label"], group=spec["group"], rmse=num(np.sqrt(np.mean(e ** 2))), dm_p=None if len(e) < 20 else M.dm_test(e, base_err)))
    return jsonify(ticker=tk, dates=[date_str(i) for i in te.index], up=[int(v) for v in y], direction=series, price=prices, majority_rate=maj, n=len(te), seconds=time.perf_counter() - t0)


@app.post("/api/whatif")
def whatif():
    q = body(); tk = ticker_arg(q.get("ticker"))
    d, row = test_row(tk, q.get("date"))
    b = bundle(tk); row = row.copy(); ranges = b["ranges"]
    values = q.get("values") or {}
    if not isinstance(values, dict): raise BadRequest("values must be an object")
    for k, v in values.items():
        if k not in SLIDERS: raise BadRequest(f"Unknown feature: {k}")
        try: v = float(v)
        except (TypeError, ValueError): raise BadRequest(f"{k} must be a number")
        span = ranges[k]["hi"] - ranges[k]["lo"]
        if not (math.isfinite(v) and ranges[k]["lo"] - 5 * span <= v <= ranges[k]["hi"] + 5 * span): raise BadRequest(f"{k} is too far outside the range seen in training")
        row[k] = v
    if "dow" in q:
        if q["dow"] not in DOW: raise BadRequest("Unknown weekday")
        for name, col in DOW.items():
            if col: row[col] = 1.0 if name == q["dow"] else 0.0
    t0 = time.perf_counter()
    x = b["design"].x(row); close_t = row.close_t.values
    keys = [k for k, v in M.PRICE.items() if v["target"] != "A"]
    flags = [k for k in SLIDERS if not ranges[k]["lo"] <= row[k].iloc[0] <= ranges[k]["hi"]]
    dirs = direction_rows(b, x)
    return jsonify(ticker=tk, date=date_str(d), close=num(close_t[0]), direction=dirs, price=price_rows(b, x, close_t, keys), features=features_of(row), outside=flags, n_up=sum(r["call"] == "up" for r in dirs if r["group"] != "baseline"), n_models=sum(r["group"] != "baseline" for r in dirs), seconds=time.perf_counter() - t0)


def settings_with(reg, depth):
    h = M.tuned(store.results); changed = {}
    if reg is not None:
        h["logistic"]["C"] = 1 / reg; h["bayes_logistic"]["alpha"] = reg
        for k in ("ridge_a", "ridge_b_raw", "ridge_b_eng"): h[k]["alpha"] = reg
        changed.update({k: "λ" for k in ("logistic", "bayes_logistic", "ridge_a", "ridge_b_raw", "ridge_b_eng")})
    if depth is not None:
        h["cart"] = dict(ccp_alpha=0.0, max_depth=depth); h["random_forest"]["max_depth"] = depth; h["gradient_boosting"]["max_depth"] = depth
        changed.update({k: "depth" for k in ("cart", "random_forest", "gradient_boosting")})
    return h, changed


def notebook_score(key):
    R = store.results
    for r in R["u2_clf"]["table"] + R["u5"]["leaderboard"]:
        if r["key"] == key: return dict(accuracy=r["accuracy"], auc=r.get("auc"))
    for r in R["u2_reg"]["table"]:
        if r["key"] == key: return dict(rmse=r["rmse"])
    for r in R["u4"]["kalman"]["forecast"]:
        if {"kalman_ll": "ll", "kalman_llt": "llt"}.get(key) == r["key"]: return dict(rmse=r["rmse"])
    if key == "hmm": return dict(accuracy=R["u4"]["forecast"]["reliance"]["filtered"]["accuracy"])
    return None


@app.post("/api/retrain")
def retrain():
    q = body(); tk = ticker_arg(q.get("ticker")); key = q.get("model")
    if key not in M.DIRECTION and key not in M.PRICE and key not in M.KALMAN and key != "hmm": raise BadRequest(f"Unknown model: {key}")
    try: cutoff = pd.Timestamp(q.get("cutoff"))
    except (ValueError, TypeError): raise BadRequest("cutoff must be a date")
    if not pd.Timestamp(CUTOFF_RANGE[0]) <= cutoff <= pd.Timestamp(CUTOFF_RANGE[1]): raise BadRequest(f"Pick a training cutoff from {CUTOFF_RANGE[0]} to {CUTOFF_RANGE[1]}.")
    reg, depth = q.get("reg"), q.get("depth")
    try:
        reg = None if reg is None else float(reg); depth = None if depth is None else int(depth)
    except (TypeError, ValueError): raise BadRequest("reg and depth must be numbers")
    if reg is not None and not REG_RANGE[0] <= reg <= REG_RANGE[1]: raise BadRequest("Regularisation is out of range.")
    if depth is not None and not DEPTH_RANGE[0] <= depth <= DEPTH_RANGE[1]: raise BadRequest("Tree depth is out of range.")
    D = store.frame(tk); tr, te = M.split(D, cutoff)
    if len(tr) < MIN_TRAIN or len(te) < MIN_TEST: raise BadRequest(f"{tk} has {len(tr)} training days and {len(te)} test days at that cutoff. It needs at least {MIN_TRAIN} and {MIN_TEST}.")
    h, changed = settings_with(reg, depth)
    des = M.Design(tr); xtr, xte = des.x(tr), des.x(te)
    ytr, yte = tr.y_dir.values.astype(int), te.y_dir.values.astype(int)
    maj_cls = int(ytr.mean() > .5); maj = float(np.mean(yte == maj_cls))
    close_t, truth = te.close_t.values, te.close_next.values; base_rmse = float(np.sqrt(np.mean((truth - close_t) ** 2)))
    t0 = time.perf_counter()
    out = dict(key=key, ticker=tk, cutoff=date_str(cutoff), n_train=len(tr), n_test=len(te), majority_rate=maj, persistence_rmse=base_rmse, changed=changed.get(key))
    if key in M.DIRECTION:
        try:
            m = M.fit_direction(key, xtr, ytr, h)
        except (M.OptimizationError, np.linalg.LinAlgError, ValueError) as e:
            raise RuntimeError(f"{M.DIRECTION[key]['label']} could not be fitted with these settings.") from e
        out["seconds"] = time.perf_counter() - t0
        yhat, score, _ = M.predict_direction(key, m, xte); s = M.score_direction(yhat, score, yte)
        trh = M.predict_direction(key, m, xtr)[0]
        out.update(kind="direction", label=M.DIRECTION[key]["label"], group=M.DIRECTION[key]["group"], accuracy=s["accuracy"], auc=s["auc"], share_up=s["share_up"], train_accuracy=float(np.mean(trh == ytr)), p_vs_majority=None if M.DIRECTION[key]["group"] == "baseline" else M.binom_p(s["n_correct"], len(yte), maj))
        if key == "cart": out["leaves"] = int(m.get_n_leaves())
        if key == "gam": out["lam_used"] = m.lam_used
    elif key == "hmm":
        s = store.series(tk); rets = s.r.dropna() * 100
        m = M.fit_hmm(rets, cutoff, store.results, tk); out["seconds"] = time.perf_counter() - t0
        _, p_up, _, _ = hmm_block(m, rets)
        hit = (pd.Series(p_up, index=rets.index).reindex(te.index).values > .5).astype(int) == yte
        out.update(kind="direction", label="HMM (filtered)", group="regime", accuracy=float(hit.mean()), auc=None, train_accuracy=None, p_vs_majority=M.binom_p(int(hit.sum()), len(yte), maj))
    else:
        if key in M.PRICE:
            m = M.fit_price(key, xtr, tr, h); out["seconds"] = time.perf_counter() - t0
            p = M.predict_price(key, m, xte, close_t)[0]; spec = M.PRICE[key]
        else:
            adj = store.series(tk).Adj; spec = M.KALMAN[key]
            m = M.fit_kalman(spec["kind"], 100 * np.log(adj[adj.index < cutoff].values), store.results, tk, cutoff); out["seconds"] = time.perf_counter() - t0
            p = pd.Series(M.kalman_price_forecast(m, adj.values), index=adj.index).reindex(te.index).values
        e = truth - p
        out.update(kind="price", label=spec["label"], group=spec["group"], rmse=float(np.sqrt(np.mean(e ** 2))), dm_p=None if key == "persistence" else M.dm_test(e, truth - close_t))
    if tk == "RELIANCE" and cutoff == P.TEST_START and key not in changed:
        out["notebook"] = notebook_score(key)
    return jsonify(out)


@app.get("/api/today")
def today():
    tk = ticker_arg(request.args.get("ticker"))
    m = market.get(store.tickers, force=request.args.get("refresh") == "1")
    if tk not in m["have"]:
        return jsonify(ticker=tk, available=False, missing=m["missing"], have=m["have"], source=m["source"], fetched_at=m["fetched_at"], last_date=m["last_date"])
    t0 = time.perf_counter()
    F, RAW, T = P.build_features(m["panel"], tk, m["mkt_r"])
    frame = pd.concat([F, RAW, T], axis=1).dropna(subset=P.ENG)
    if frame.empty: raise RuntimeError(f"Not enough recent data for {tk}.")
    row = frame.iloc[[-1]]; d = row.index[0]
    b = bundle(tk, P.ALL_DATA_CUTOFF)
    x = b["design"].x(row); close_t = row.close_t.values
    dirs = direction_rows(b, x)
    prices = price_rows(b, x, close_t, [k for k, v in M.PRICE.items() if v["target"] != "A"])
    s = m["panel"][m["panel"].Ticker == tk].set_index("Date").sort_index()
    kf = kalman_forecasts(b, s.Adj)
    for key, spec in M.KALMAN.items():
        prices.append(dict(key=key, label=spec["label"], group=spec["group"], forecast=num(kf[key][-1]), lo=None, hi=None))
    h = b["hmm"]; rets = s.r.dropna() * 100
    filt, p_up, mean, sd = hmm_block(h, rets, M.stationary(h["A"]))
    outside = [k for k in P.ENG if not k.startswith("dow_") and not b["ranges"][k]["lo"] <= row[k].iloc[0] <= b["ranges"][k]["hi"]]
    hist = s.Adj.iloc[-HISTORY_DAYS:]
    return jsonify(ticker=tk, available=True, date=date_str(d), close=num(close_t[0]), direction=dirs, price=prices, hmm=hmm_row(h, filt, p_up, mean, sd, -1), features=features_of(row), outside=outside, n_up=sum(r["call"] == "up" for r in dirs if r["group"] != "baseline"), n_models=sum(r["group"] != "baseline" for r in dirs),
                   history=dict(date=[date_str(i) for i in hist.index], close=[num(v) for v in hist.values]), trained_through=b["train_end"], source=m["source"], fetched_at=m["fetched_at"], note=m["note"], last_date=m["last_date"], missing=m["missing"], n_market=len(m["have"]), seconds=time.perf_counter() - t0)


# ---------- the built site ----------
@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        return "The site has not been built yet. Run npm run build inside web/.", 503
    return send_from_directory(DIST, "index.html")


@app.get("/<path:path>")
def assets(path):
    return send_from_directory(DIST, path)


def main():
    global store
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print("Loading the dataset…", flush=True)
    store = P.Store()
    print("Loading the RELIANCE models…", flush=True)
    bundle("RELIANCE")
    print(f"Ready: open http://localhost:{a.port}/#/live-demo", flush=True)
    app.run(host=HOST, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
