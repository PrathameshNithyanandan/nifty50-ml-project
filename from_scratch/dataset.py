"""Reads the per-stock CSV files, removes corporate actions and builds the features, using only Python itself."""
import csv
import math
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
NOT_STOCKS = {"NIFTY50_all.csv", "stock_metadata.csv"}
TEST_START = "2018-01-01"
ENG = ["r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4", "rmean_10", "rmean_20", "rstd_5", "rstd_10", "rstd_20", "mom_60", "dist_sma50",
       "hl_range", "oc_ret", "vol_ratio20", "dlog_volume", "mkt_r", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
RAWC = ["close_lag0", "close_lag1", "close_lag2", "close_lag3", "close_lag4", "log_volume"]
RAWRET = ["r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4", "dlog_volume"]


def read_stock(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["Date"])
    return dict(date=[r["Date"] for r in rows], open=[float(r["Open"]) for r in rows], high=[float(r["High"]) for r in rows],
                low=[float(r["Low"]) for r in rows], close=[float(r["Close"]) for r in rows], volume=[float(r["Volume"]) for r in rows])


def clean_stock(d):
    """Log-returns with corporate-action days and data holes removed, then a back-adjusted price series."""
    n = len(d["date"])
    days = [date.fromisoformat(x) for x in d["date"]]
    r, ca, hole = [None] * n, [False] * n, [False] * n
    for t in range(1, n):
        cp = d["close"][t - 1]
        ca[t] = d["high"][t] < 0.80 * cp or d["low"][t] > 1.25 * cp or d["open"][t] < 0.75 * cp or d["open"][t] > 1.333 * cp
        hole[t] = (days[t] - days[t - 1]).days > 20
        if not (ca[t] or hole[t]):
            r[t] = math.log(d["close"][t] / cp)
    cum, total = [0.0] * n, 0.0
    for t in range(n):
        total += r[t] or 0.0
        cum[t] = total
    adj = [d["close"][-1] * math.exp(c - total) for c in cum]
    fac = [a / c for a, c in zip(adj, d["close"])]
    d.update(r=r, ca=ca, hole=hole, adj=adj, adj_open=[x * f for x, f in zip(d["open"], fac)], adj_high=[x * f for x, f in zip(d["high"], fac)],
             adj_low=[x * f for x, f in zip(d["low"], fac)], adj_volume=[x / f for x, f in zip(d["volume"], fac)])
    return d


def load_panel():
    panel = {}
    for name in sorted(os.listdir(DATA)):
        if name.endswith(".csv") and name not in NOT_STOCKS:
            d = read_stock(os.path.join(DATA, name))
            if d["date"]:
                panel[name[:-4]] = clean_stock(d)
    return panel


def industries():
    with open(os.path.join(DATA, "stock_metadata.csv"), newline="") as f:
        # the file for M&M is named MM.csv
        return {r["Symbol"].replace("&", ""): r["Industry"] for r in csv.DictReader(f)}


def audit(panel):
    return dict(tickers=len(panel), n_returns=sum(x is not None for d in panel.values() for x in d["r"]), n_ca=sum(sum(d["ca"]) for d in panel.values()),
                n_hole=sum(sum(d["hole"]) for d in panel.values()), n_overlap=sum(a and b for d in panel.values() for a, b in zip(d["ca"], d["hole"])))


def market_return(panel):
    """Equal-weighted mean return per date, over the stocks that have a return that day."""
    total, count = {}, {}
    for d in panel.values():
        for day, x in zip(d["date"], d["r"]):
            if x is not None:
                total[day] = total.get(day, 0.0) + x
                count[day] = count.get(day, 0) + 1
    return {day: total[day] / count[day] for day in total}


def rolling(v, w, min_periods, stat):
    """Rolling statistic over the last w values, skipping missing ones; a window with too few values gives None."""
    out = [None] * len(v)
    for t in range(len(v)):
        win = [x for x in v[max(0, t - w + 1):t + 1] if x is not None]
        if len(win) >= min_periods:
            out[t] = stat(win)
    return out


def _mean(x):
    return sum(x) / len(x)


def _std(x):
    if len(x) < 2:
        return None
    m = _mean(x)
    return math.sqrt(sum((a - m) ** 2 for a in x) / (len(x) - 1))


def shift(v, k):
    return ([None] * k + v[:-k]) if k > 0 else ([*v[-k:], *[None] * -k] if k < 0 else v[:])


def features(panel, tk, mkt=None):
    """Features use day t only; the targets use day t+1."""
    d = panel[tk]
    mkt = market_return(panel) if mkt is None else mkt
    n, r, adj, vol = len(d["date"]), d["r"], d["adj"], d["adj_volume"]
    weekday = [date.fromisoformat(x).weekday() for x in d["date"]]
    sma50, vol20 = rolling(adj, 50, 50, _mean), rolling(vol, 20, 20, _mean)
    F = {f"r_lag{k}": shift(r, k) for k in range(5)}
    for w in (10, 20):
        F[f"rmean_{w}"] = rolling(r, w, int(0.8 * w), _mean)
    for w in (5, 10, 20):
        F[f"rstd_{w}"] = rolling(r, w, int(0.8 * w), _std)
    F["mom_60"] = [None if t < 60 else math.log(adj[t] / adj[t - 60]) for t in range(n)]
    F["dist_sma50"] = [None if sma50[t] is None else math.log(adj[t] / sma50[t]) for t in range(n)]
    F["hl_range"] = [math.log(h / l) for h, l in zip(d["adj_high"], d["adj_low"])]
    F["oc_ret"] = [math.log(a / o) for a, o in zip(adj, d["adj_open"])]
    F["vol_ratio20"] = [None if vol20[t] is None else vol[t] / vol20[t] for t in range(n)]
    F["dlog_volume"] = [None if t == 0 else math.log(vol[t] / vol[t - 1]) for t in range(n)]
    F["mkt_r"] = [mkt.get(x) for x in d["date"]]
    for i, nm in zip(range(1, 5), ["tue", "wed", "thu", "fri"]):
        F[f"dow_{nm}"] = [float(x == i) for x in weekday]
    for k in range(5):
        F[f"close_lag{k}"] = shift(adj, k)
    F["log_volume"] = [math.log(x) for x in vol]
    close_next, r_next = shift(adj, -1), shift(r, -1)
    keep = [t for t in range(n) if close_next[t] is not None and r_next[t] is not None and all(F[c][t] is not None for c in ENG + RAWC)]
    cols = {c: [F[c][t] for t in keep] for c in F}
    return dict(date=[d["date"][t] for t in keep], close_t=[adj[t] for t in keep], close_next=[close_next[t] for t in keep], r_next=[r_next[t] for t in keep],
                y=[1 if r_next[t] > 0 else 0 for t in keep], cols=cols)


def split(D):
    """Chronological split: the rows before TEST_START train the models and the rest test them."""
    cut = sum(x < TEST_START for x in D["date"])
    rows = lambda names, a, b: [list(r) for r in zip(*[D["cols"][c][a:b] for c in names])]
    part = lambda a, b: dict(date=D["date"][a:b], close_t=D["close_t"][a:b], close_next=D["close_next"][a:b], r_next=D["r_next"][a:b], y=D["y"][a:b],
                             ENG=rows(ENG, a, b), RAW=rows(RAWC, a, b), RET=rows(RAWRET, a, b), r_lag0=D["cols"]["r_lag0"][a:b], rstd_20=D["cols"]["rstd_20"][a:b])
    return part(0, cut), part(cut, len(D["date"]))


def time_folds(n, splits=5):
    """Expanding-window folds: each trains on the past and validates on the block after it."""
    size = n // (splits + 1)
    return [(start, start + size) for start in range(n - splits * size, n, size)]
