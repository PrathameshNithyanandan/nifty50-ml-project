"""Recent NIFTY-50 prices for the today's-market forecast. The last good download is kept on disk as a fallback."""
from . import pipeline as P
import datetime as dt, json, threading, time
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

SAVED = P.CACHE_DIR / "market.csv"
SAVED_INFO = P.CACHE_DIR / "market.json"
IST = ZoneInfo("Asia/Kolkata")
FRESH_SECONDS = 600
MIN_DAYS = 120
_lock = threading.Lock()
_memory = None


def _download(tickers):
    import yfinance as yf
    raw = yf.download([f"{t}.NS" for t in tickers], period="1y", auto_adjust=False, progress=False, threads=True, group_by="ticker", timeout=20)
    rows = []
    for t in tickers:
        sym = f"{t}.NS"
        if sym not in raw.columns.get_level_values(0): continue
        d = raw[sym][["Open", "High", "Low", "Close", "Volume"]].dropna()
        d = d[d.Volume > 0]
        if len(d) < MIN_DAYS: continue
        rows.append(d.assign(Ticker=t).rename_axis("Date").reset_index())
    if not rows: raise RuntimeError("the price feed returned no data")
    return pd.concat(rows, ignore_index=True)


def _drop_unfinished(d, now=None):
    """While the exchange is open, the newest bar is not a finished day, so it is removed."""
    now = now or dt.datetime.now(IST)
    d = d.copy(); d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
    if now.time() < dt.time(15, 45):
        d = d[d.Date < pd.Timestamp(now.date())]
    return d


def _prepare(d, tickers, source, fetched_at, note=None):
    panel = P.clean_panel(d)
    have = sorted(panel.Ticker.unique())
    return dict(panel=panel, mkt_r=panel.groupby("Date").r.mean(), have=have, missing=[t for t in tickers if t not in have], source=source, fetched_at=fetched_at, note=note, last_date=str(panel.Date.max().date()), loaded=time.time())


def get(tickers, force=False):
    """Fresh data when the feed works. Otherwise the last saved download, clearly marked. None if neither exists."""
    global _memory
    with _lock:
        if _memory and not force and _memory["source"] == "live" and time.time() - _memory["loaded"] < FRESH_SECONDS:
            return _memory
        try:
            d = _drop_unfinished(_download(tickers))
            now = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
            P.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            d.to_csv(SAVED, index=False); SAVED_INFO.write_text(json.dumps(dict(fetched_at=now)))
            _memory = _prepare(d, tickers, "live", now)
        except Exception as e:
            if not SAVED.exists():
                _memory = None
                raise RuntimeError(f"The price feed could not be reached and no saved download exists yet ({type(e).__name__}).") from e
            d = pd.read_csv(SAVED, parse_dates=["Date"])
            _memory = _prepare(d, tickers, "saved", json.loads(SAVED_INFO.read_text())["fetched_at"], note=type(e).__name__)
        return _memory
