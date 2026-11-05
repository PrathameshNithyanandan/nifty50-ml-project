"""Data pipeline for the live demo: the same cleaning and feature code as analysis.ipynb."""
import os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import glob, json, pathlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "demo" / "cache"
SEED = 42
TEST_START = pd.Timestamp("2018-01-01")
ALL_DATA_CUTOFF = pd.Timestamp("2021-05-01")   # the day after the dataset ends: used only for forecasts on new market data
RAW_COLS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
ENG = ["r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4", "rmean_10", "rmean_20", "rstd_5", "rstd_10", "rstd_20", "mom_60", "dist_sma50", "hl_range", "oc_ret", "vol_ratio20", "dlog_volume", "mkt_r", "dow_tue", "dow_wed", "dow_thu", "dow_fri"]
RAWC = ["close_lag0", "close_lag1", "close_lag2", "close_lag3", "close_lag4", "log_volume"]
RAWRET = ["r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4", "dlog_volume"]


def load_results():
    with open(ROOT / "results" / "results.json") as f:
        return json.load(f)


def load_raw():
    per = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        name = os.path.basename(f)[:-4]
        if name in ("NIFTY50_all", "stock_metadata"):
            continue
        s = pd.read_csv(f)
        if len(s) == 0:
            continue
        s["Ticker"] = "M&M" if name == "MM" else name
        per.append(s)
    ps = pd.concat(per, ignore_index=True)
    ps["Date"] = pd.to_datetime(ps["Date"])
    return ps


def clean_panel(raw):
    d = raw[RAW_COLS].sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    g = d.groupby("Ticker"); cp = g.Close.shift(1); gap = g.Date.diff().dt.days
    ca = (d.High < 0.80 * cp) | (d.Low > 1.25 * cp) | (d.Open < 0.75 * cp) | (d.Open > 1.333 * cp)
    hole = gap > 20
    d["ca"], d["hole"] = ca, hole
    d["r_raw"] = np.log(d.Close / cp)
    d["r"] = d.r_raw.where(~(ca | hole))
    cum = d.r.fillna(0).groupby(d.Ticker).cumsum()
    d["Adj"] = d.groupby("Ticker").Close.transform("last") * np.exp(cum - cum.groupby(d.Ticker).transform("last"))
    fac = d.Adj / d.Close
    d["AdjOpen"], d["AdjHigh"], d["AdjLow"], d["AdjVolume"] = d.Open * fac, d.High * fac, d.Low * fac, d.Volume / fac
    return d


def build_features(pan, tk, mkt_r=None):
    d = pan[pan.Ticker == tk].set_index("Date").sort_index()
    mkt_r = pan.groupby("Date").r.mean() if mkt_r is None else mkt_r
    r = d.r; F = pd.DataFrame(index=d.index)
    for k in range(5): F[f"r_lag{k}"] = r.shift(k)
    for w in (10, 20): F[f"rmean_{w}"] = r.rolling(w, min_periods=int(0.8 * w)).mean()
    for w in (5, 10, 20): F[f"rstd_{w}"] = r.rolling(w, min_periods=int(0.8 * w)).std()
    F["mom_60"] = np.log(d.Adj / d.Adj.shift(60))
    F["dist_sma50"] = np.log(d.Adj / d.Adj.rolling(50).mean())
    F["hl_range"] = np.log(d.AdjHigh / d.AdjLow)
    F["oc_ret"] = np.log(d.Adj / d.AdjOpen)
    F["vol_ratio20"] = d.AdjVolume / d.AdjVolume.rolling(20).mean()
    F["dlog_volume"] = np.log(d.AdjVolume / d.AdjVolume.shift(1))
    F["mkt_r"] = mkt_r.reindex(d.index)
    for i, nm in zip(range(1, 5), ["tue", "wed", "thu", "fri"]):
        F[f"dow_{nm}"] = (d.index.dayofweek == i).astype(float)
    RAW = pd.DataFrame(index=d.index)
    for k in range(5): RAW[f"close_lag{k}"] = d.Adj.shift(k)
    RAW["log_volume"] = np.log(d.AdjVolume)
    T = pd.DataFrame(index=d.index)
    T["close_t"] = d.Adj; T["close_next"] = d.Adj.shift(-1); T["r_next"] = r.shift(-1)
    T["y_dir"] = np.where(T.r_next.isna(), np.nan, (T.r_next > 0).astype(float))
    T["quoted_close_t"] = d.Close
    return F, RAW, T


def modelling_frame(pan, tk, mkt_r=None):
    """Rows with complete features and a known next-day target, as in the notebook."""
    F, RAW, T = build_features(pan, tk, mkt_r)
    return pd.concat([F, RAW, T], axis=1).dropna(subset=ENG + RAWC + ["close_next", "r_next"])


class Store:
    """The cleaned panel and each stock's frames, loaded once per process."""
    def __init__(self):
        self.panel = clean_panel(load_raw())
        self.mkt_r = self.panel.groupby("Date").r.mean()
        self.tickers = sorted(self.panel.Ticker.unique())
        self.results = load_results()
        self._frames = {}

    def frame(self, tk):
        if tk not in self._frames:
            self._frames[tk] = modelling_frame(self.panel, tk, self.mkt_r)
        return self._frames[tk]

    def series(self, tk):
        return self.panel[self.panel.Ticker == tk].set_index("Date").sort_index()
