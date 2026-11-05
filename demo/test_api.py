"""Exercises the running demo server: normal requests, bad requests, and that the outcome stays hidden until reveal."""
import json, sys, time, urllib.error, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
fails = []


def call(path, payload=None):
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.time() - t


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name, detail)
    if not ok: fails.append(name)


s, meta, _ = call("/api/meta")
check("meta", s == 200 and len(meta["stocks"]) == 49 and len(meta["direction"]) == 16 and len(meta["price"]) == 13)
s, st, t = call("/api/stock?ticker=RELIANCE")
check("stock", s == 200 and len(st["dates"]) == meta["notebook"]["n_test"], f"{t:.2f}s")
day = st["highlights"]["biggest_fall"]

s, a, t = call("/api/replay", dict(ticker="RELIANCE", date=day))
text = json.dumps(a)
check("replay runs every model", s == 200 and len(a["direction"]) == 16 and len(a["price"]) == 13, f"{t:.2f}s")
check("outcome hidden before reveal", "actual" not in a and '"correct"' not in text and '"error"' not in text)
check("history stops at the chosen day", a["history"]["date"][-1] == day)
s, b, t = call("/api/replay", dict(ticker="RELIANCE", date=day, reveal=True))
check("reveal gives the outcome", s == 200 and "actual" in b and all("correct" in r for r in b["direction"]), f"{t:.2f}s")
check("reveal does not change the forecasts", [r["call"] for r in a["direction"]] == [r["call"] for r in b["direction"]] and [r["forecast"] for r in a["price"]] == [r["forecast"] for r in b["price"]])

s, run, t = call("/api/replay/run", dict(ticker="RELIANCE", date=st["first"], days=len(st["dates"])))
acc = {r["key"]: r["accuracy"] for r in run["direction"]}
check("a full run scores the notebook's majority rate", s == 200 and abs(run["majority_rate"] - meta["notebook"]["majority_rate"]) < 1e-9, f"{t:.2f}s")
rm = {r["key"]: r["rmse"] for r in run["price"]}
check("a full run scores the notebook's persistence RMSE", abs(rm["persistence"] - meta["notebook"]["persistence_rmse"]) < 1e-6)
s, r50, t = call("/api/replay/run", dict(ticker="RELIANCE", date=st["first"], days=50))
check("50-day run", s == 200 and r50["n"] == 50 and len(r50["direction"][0]["running"]) == 50, f"{t:.2f}s")

s, w0, t = call("/api/whatif", dict(ticker="RELIANCE", date=day, values={}))
check("what-if with no change equals the replay", s == 200 and [r["prob"] for r in w0["direction"]] == [r["prob"] for r in a["direction"]], f"{t:.2f}s")
hi = st["ranges"]["r_lag0"]["hi"]
s, w1, _ = call("/api/whatif", dict(ticker="RELIANCE", date=day, values=dict(r_lag0=hi), dow="fri"))
changed = sum(x["prob"] != y["prob"] for x, y in zip(w0["direction"], w1["direction"]) if x["prob"] is not None)
check("what-if responds to a slider", s == 200 and changed > 3, f"{changed} probabilities changed")
s, w2, _ = call("/api/whatif", dict(ticker="RELIANCE", date=day, values=dict(r_lag0=st["ranges"]["r_lag0"]["hi"] * 2)))
check("what-if flags values outside training", s == 200 and "r_lag0" in w2["outside"])

times = {}
for row in meta["direction"] + meta["price"] + [dict(key="hmm")]:
    s, r, t = call("/api/retrain", dict(ticker="RELIANCE", cutoff="2018-01-01", model=row["key"], reg=None, depth=None))
    times[row["key"]] = round(t, 2)
    nb = r.get("notebook")
    good = s == 200 and (nb is None or all(abs(r[k] - v) < 1e-4 for k, v in nb.items() if v is not None and r.get(k) is not None))
    check(f"retrain {row['key']} matches the notebook", good and nb is not None or row["key"] == "persistence" and good, f"{t:.2f}s")
s, r, _ = call("/api/retrain", dict(ticker="RELIANCE", cutoff="2018-01-01", model="cart", depth=30))
check("deep CART overfits", s == 200 and r["train_accuracy"] > .9 and r["leaves"] > 100 and "notebook" not in r, f"train {r['train_accuracy']:.3f}, test {r['accuracy']:.3f}, leaves {r['leaves']}")
s, r, t = call("/api/retrain", dict(ticker="TCS", cutoff="2012-06-15", model="logistic", reg=10))
check("retrain at another cutoff and stock", s == 200 and r["n_train"] > 500 and r["changed"] == "λ", f"{t:.2f}s")

for name, path, payload in [
    ("unknown stock", "/api/replay", dict(ticker="NOPE", date=day)),
    ("non-date", "/api/replay", dict(ticker="RELIANCE", date="soon")),
    ("training-period day", "/api/replay", dict(ticker="RELIANCE", date="2010-01-04")),
    ("non-trading day", "/api/replay", dict(ticker="RELIANCE", date="2020-03-22")),
    ("no body", "/api/replay", None),
    ("bad days", "/api/replay/run", dict(ticker="RELIANCE", date=day, days="many")),
    ("zero days", "/api/replay/run", dict(ticker="RELIANCE", date=day, days=0)),
    ("unknown feature", "/api/whatif", dict(ticker="RELIANCE", date=day, values=dict(volume=1))),
    ("feature not a number", "/api/whatif", dict(ticker="RELIANCE", date=day, values=dict(r_lag0="x"))),
    ("feature absurd", "/api/whatif", dict(ticker="RELIANCE", date=day, values=dict(r_lag0=1e9))),
    ("bad weekday", "/api/whatif", dict(ticker="RELIANCE", date=day, dow="sun")),
    ("unknown model", "/api/retrain", dict(ticker="RELIANCE", cutoff="2018-01-01", model="magic")),
    ("cutoff too late", "/api/retrain", dict(ticker="RELIANCE", cutoff="2021-04-01", model="lda")),
    ("reg out of range", "/api/retrain", dict(ticker="RELIANCE", cutoff="2018-01-01", model="logistic", reg=-1)),
    ("depth out of range", "/api/retrain", dict(ticker="RELIANCE", cutoff="2018-01-01", model="cart", depth=500)),
    ("today unknown stock", "/api/today?ticker=NOPE", None),
]:
    s, r, _ = call(path, payload) if payload is not None or path.startswith("/api/today") else call(path, {})
    check(f"rejects {name}", s == 400 and "error" in r and "Traceback" not in json.dumps(r), f"{s}: {r.get('error', '')[:70]}")

s, r, t = call("/api/today?ticker=RELIANCE")
if s == 200:
    check("today's forecast", r["available"] and len(r["direction"]) == 16 and r["trained_through"] < "2021-05-01", f"{t:.2f}s, data to {r['date']}, source {r['source']}, {r['n_market']} stocks, missing {r['missing']}, outside {r['outside']}")
else:
    check("today's forecast", False, str(r))
s, r, t = call("/api/today?ticker=HDFC")
check("today for a delisted stock", s == 200 and r["available"] is False, f"{t:.2f}s")

print("\nretrain seconds:", times)
print("FAILED:" if fails else "All API checks passed.", fails if fails else "")
sys.exit(1 if fails else 0)
