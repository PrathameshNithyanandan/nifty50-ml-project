"""Unit 2 with no libraries: next-day closing price and up/down direction for RELIANCE.
Equation numbers refer to MATHEMATICAL_MODEL.md.

    python3 from_scratch/unit2.py              uses the settings saved by the last tuning run
    python3 from_scratch/unit2.py --tune       tunes first, using the training days only (about 2 minutes)
    python3 from_scratch/unit2.py --tune-rbf   also searches the RBF SVM grid (about 4 minutes on 10 cores)
"""
import json
import math
import multiprocessing
import os
import random
import sys
import time
from array import array

from dataset import RESULTS, audit, features, load_panel, split, time_folds
from matrix import add_diag, dot, eigh, gram, identity, inverse, log_det, matvec, mean, solve, standardise, transpose, variance

TUNED = os.path.join(RESULTS, "from_scratch_tuned.json")
Z95 = 1.959963984540054
SEED = 42


# ---------- Measures ----------

def normal_two_sided(t):
    return math.erfc(abs(t) / math.sqrt(2))


def dm_test(e1, e0, lags=5):
    """Diebold–Mariano on squared errors with a Newey–West long-run variance (equation 2.20)."""
    d = [a * a - b * b for a, b in zip(e1, e0)]
    n, m = len(d), mean(d)
    c = [x - m for x in d]
    v = dot(c, c) / n + 2 * sum((1 - l / (lags + 1)) * dot(c[l:], c[:-l]) / n for l in range(1, lags + 1))
    t = m / math.sqrt(v / n)
    return t, normal_two_sided(t)


def auc(y, score):
    """Area under the ROC curve from average ranks, so tied scores share a rank."""
    order = sorted(range(len(y)), key=lambda i: score[i])
    ranks, i = [0.0] * len(y), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    return (sum(r for r, t in zip(ranks, y) if t) - n1 * (n1 + 1) / 2) / (n1 * n0)


def f1(y, yhat):
    tp = sum(a and b for a, b in zip(y, yhat))
    p, q = sum(yhat), sum(y)
    return 2 * tp / (p + q) if p + q else 0.0


def log_loss(y, p):
    p = [min(max(x, 1e-9), 1 - 1e-9) for x in p]
    return -mean([math.log(a) if t else math.log(1 - a) for t, a in zip(y, p)])


def binom_greater(k, n, p):
    """Exact P(X ≥ k) for X ~ Binomial(n, p)."""
    lg = math.lgamma
    term = lambda i: math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * math.log(p) + (n - i) * math.log(1 - p))
    return min(1.0, sum(term(i) for i in range(k, n + 1)))


def tune(score, grid, max_ext=4, factor=100.0, rel_tol=1e-4):
    """Grid search on time-ordered folds. While the winner is on an edge the grid is extended,
    and the extension stops once the score has flattened."""
    grid, seen, prev, plateau = list(grid), {}, None, False
    for _ in range(max_ext + 1):
        for g in grid:
            if g not in seen:
                seen[g] = score(g)
        best = max(grid, key=lambda g: seen[g])
        if best not in (grid[0], grid[-1]):
            break
        if prev is not None and abs(seen[best] - seen[prev]) <= rel_tol * abs(seen[prev]):
            best, plateau = prev, True
            break
        prev = best
        grid = grid + [grid[-1] * factor] if best == grid[-1] else [grid[0] / factor] + grid
    return dict(best=best, plateau=plateau, at_edge=best in (grid[0], grid[-1]), grid=grid, score=[seen[g] for g in grid])


# ---------- Regression ----------

def fit_ridge(X, y, alpha=0.0):
    """Normal equations on centred data (equations 2.3 and 2.8). The intercept is not penalised."""
    C = transpose(X)
    mx, my = [mean(c) for c in C], mean(y)
    Cc = [[v - m for v in c] for c, m in zip(C, mx)]
    t = [v - my for v in y]
    w = solve(add_diag(gram(Cc), alpha), [dot(c, t) for c in Cc])
    return my - dot(mx, w), w


def predict(model, X):
    b, w = model
    return [b + dot(w, r) for r in X]


def mse(y, p):
    return mean([(a - b) ** 2 for a, b in zip(y, p)])


def ols_as_mle(X, y):
    """Maximises the Gaussian log-likelihood by gradient ascent and compares it with the closed form (equation 2.4)."""
    Phi = [[1.0] + r for r in X]
    n, C = len(y), transpose(Phi)
    A = [[v / n for v in r] for r in gram(C)]
    b = [dot(c, y) / n for c in C]
    w_closed = solve(A, b)
    lam = eigh(A, vectors=False)
    step, w, iters = 1 / lam[0], [0.0] * len(b), 0
    for iters in range(1, 200001):
        g = [a - c for a, c in zip(matvec(A, w), b)]
        w = [a - step * c for a, c in zip(w, g)]
        if max(map(abs, g)) < 1e-13:
            break
    res = [t - dot(r, w_closed) for t, r in zip(y, Phi)]
    return dict(mle_diff=max(abs(a - c) for a, c in zip(w, w_closed)), iters=iters, beta_closed=n / dot(res, res), cond=lam[0] / lam[-1])


def fit_huber(X, y, eps, alpha=1e-4, iters=500, tol=1e-9):
    """Huber loss with a jointly estimated scale, by iteratively reweighted least squares (equations 2.6 and 2.7).
    y is standardised first, as in the notebook."""
    my = mean(y)
    sy = math.sqrt(variance(y))
    t = [(v - my) / sy for v in y]
    Phi = [[1.0] + r for r in X]
    C = transpose(Phi)
    d = len(C)
    w, sigma = [0.0] * d, 1.0
    for _ in range(iters):
        res = [a - dot(w, r) for a, r in zip(t, Phi)]
        v = [1.0 if abs(e) <= eps * sigma else eps * sigma / abs(e) for e in res]
        A = gram(C, v)
        for j in range(1, d):
            A[j][j] += sigma * alpha
        w_new = solve(A, [dot(list(map(lambda a, b: a * b, c, v)), t) for c in C])
        res = [a - dot(w_new, r) for a, r in zip(t, Phi)]
        inl = [e * e for e in res if abs(e) <= eps * sigma]
        n_out = len(res) - len(inl)
        sigma_new = math.sqrt(sum(inl) / max(len(res) - eps * eps * n_out, 1e-9))
        change = max(max(abs(a - b) for a, b in zip(w, w_new)), abs(sigma - sigma_new))
        w, sigma = w_new, sigma_new
        if change < tol:
            break
    res = [a - dot(w, r) for a, r in zip(t, Phi)]
    return dict(w=w, sigma=sigma, my=my, sy=sy, outliers=sum(abs(e) > eps * sigma for e in res))


def predict_huber(m, X):
    return [m["my"] + m["sy"] * (m["w"][0] + dot(m["w"][1:], r)) for r in X]


def blr_evidence(X, y, iters=2000, tol=1e-12):
    """Bayesian linear regression. α and β come from maximising the evidence (equations 2.9 to 2.13)."""
    ym = mean(y)
    t = [v - ym for v in y]
    C = transpose(X)
    A = gram(C)
    eig, Q = eigh(A)
    Xt = [dot(c, t) for c in C]
    qt = [dot(q, Xt) for q in Q]
    alpha, beta, used = 1.0, 1 / variance(t), 0
    for used in range(1, iters + 1):
        lam = [beta * e for e in eig]
        coef = [beta * q / (alpha + l) for q, l in zip(qt, lam)]
        m = [sum(c * q[i] for c, q in zip(coef, Q)) for i in range(len(C))]
        gamma = sum(l / (alpha + l) for l in lam)
        res = [a - dot(r, m) for a, r in zip(t, X)]
        a_new, b_new = gamma / dot(m, m), (len(t) - gamma) / dot(res, res)
        conv = abs(math.log(a_new / alpha)) + abs(math.log(b_new / beta))
        alpha, beta = a_new, b_new
        if conv < tol:
            break
    S = inverse(add_diag([[beta * v for v in r] for r in A], alpha))
    return dict(m=[beta * v for v in matvec(S, Xt)], S=S, alpha=alpha, beta=beta, gamma=sum(beta * e / (alpha + beta * e) for e in eig), ym=ym, iters=used)


def blr_predict(f, X):
    mu = [dot(r, f["m"]) + f["ym"] for r in X]
    var = [1 / f["beta"] + dot(r, matvec(f["S"], r)) for r in X]
    return mu, var


# ---------- Classification ----------

def sigmoid(a):
    return 1 / (1 + math.exp(-a)) if a > -700 else 0.0


def fit_lda(X, y):
    """Shared-covariance Gaussian classes give a linear discriminant (equations 2.14 and 2.15)."""
    n1 = sum(y)
    n = len(y)
    C = transpose(X)
    m1 = [sum(v for v, t in zip(c, y) if t) / n1 for c in C]
    m0 = [sum(v for v, t in zip(c, y) if not t) / (n - n1) for c in C]
    Cc = [[v - (a if t else b) for v, t in zip(c, y)] for c, a, b in zip(C, m1, m0)]
    Sw = [[v / n for v in r] for r in gram(Cc)]
    w = solve(Sw, [a - b for a, b in zip(m1, m0)])
    return -0.5 * dot([a + b for a, b in zip(m1, m0)], w) + math.log(n1 / (n - n1)), w


def fit_gnb(X, y):
    """Class-conditional independent Gaussians (equation 2.16)."""
    C = transpose(X)
    smooth = 1e-9 * max(variance(c) for c in C)
    out = {}
    for k in (0, 1):
        part = [[v for v, t in zip(c, y) if t == k] for c in C]
        out[k] = dict(prior=len(part[0]) / len(y), mean=[mean(c) for c in part], var=[variance(c) + smooth for c in part])
    return out


def gnb_prob(f, X):
    def log_joint(k, r):
        g = f[k]
        return math.log(g["prior"]) - 0.5 * sum(math.log(2 * math.pi * s) + (v - m) ** 2 / s for v, m, s in zip(r, g["mean"], g["var"]))
    return [sigmoid(log_joint(1, r) - log_joint(0, r)) for r in X]


def laplace_fit(X, t, alpha, iters=100, tol=1e-10):
    """Finds w_MAP by Newton's method (IRLS), with the bias left out of the prior (equations 2.17 and 2.18)."""
    Phi = [[1.0] + r for r in X]
    C = transpose(Phi)
    d = len(C)
    w, used = [0.0] * d, 0
    for used in range(1, iters + 1):
        y = [sigmoid(dot(r, w)) for r in Phi]
        g = [dot(c, [a - b for a, b in zip(y, t)]) + (alpha * v if j else 0.0) for j, (c, v) in enumerate(zip(C, w))]
        H = gram(C, [a * (1 - a) for a in y])
        for j in range(1, d):
            H[j][j] += alpha
        step = solve(H, g)
        w = [a - b for a, b in zip(w, step)]
        if math.sqrt(dot(step, step)) < tol:
            break
    y = [sigmoid(dot(r, w)) for r in Phi]
    H = gram(C, [a * (1 - a) for a in y])
    for j in range(1, d):
        H[j][j] += alpha
    return dict(w=w, H=H, S=inverse(H), alpha=alpha, iters=used)


def laplace_pred(f, X, moderated=True):
    """Predictive probability with the probit correction of the variance (equation 2.19)."""
    w, S, out = f["w"], f["S"], []
    for r in X:
        p = [1.0] + r
        mu = dot(p, w)
        out.append(sigmoid(mu / math.sqrt(1 + math.pi * dot(p, matvec(S, p)) / 8)) if moderated else sigmoid(mu))
    return out


def laplace_log_evidence(X, t, f):
    Phi = [[1.0] + r for r in X]
    ll = 0.0
    for r, a in zip(Phi, t):
        p = min(max(sigmoid(dot(r, f["w"])), 1e-12), 1 - 1e-12)
        ll += math.log(p) if a else math.log(1 - p)
    w = f["w"][1:]
    return ll + 0.5 * len(w) * math.log(f["alpha"]) - 0.5 * f["alpha"] * dot(w, w) - 0.5 * log_det(f["H"]) + 0.5 * math.log(2 * math.pi)


def fit_linear_svm(X, y, C, seed=SEED, tol=1e-4, epochs=3000):
    """Hinge-loss SVM by dual coordinate descent. The bias is treated as one more feature (equation 2.24)."""
    Phi = [r + [1.0] for r in X]
    s = [1.0 if t else -1.0 for t in y]
    q = [dot(r, r) for r in Phi]
    w, a = [0.0] * len(Phi[0]), [0.0] * len(Phi)
    rng, order = random.Random(seed), list(range(len(Phi)))
    for _ in range(epochs):
        rng.shuffle(order)
        hi, lo = -math.inf, math.inf
        for i in order:
            r = Phi[i]
            g = s[i] * dot(w, r) - 1
            pg = min(g, 0.0) if a[i] == 0 else (max(g, 0.0) if a[i] == C else g)
            hi, lo = max(hi, pg), min(lo, pg)
            if pg != 0:
                new = min(max(a[i] - g / q[i], 0.0), C)
                d = (new - a[i]) * s[i]
                a[i] = new
                w = [x + d * v for x, v in zip(w, r)]
        if hi - lo < tol:
            break
    return w[-1], w[:-1]


def rbf_row(cols, x, gamma):
    d = [0.0] * len(cols[0])
    for col, v in zip(cols, x):
        d = [a + (b - v) * (b - v) for a, b in zip(d, col)]
    return array("d", [math.exp(-gamma * a) for a in d])


def fit_rbf_svm(X, y, C, gamma, tol=1e-3, max_iter=2000000):
    """Kernel SVM dual by sequential minimal optimisation with second-order pair selection (equations 2.21 to 2.23)."""
    n = len(X)
    cols, s = transpose(X), [1.0 if t else -1.0 for t in y]
    a, G, rows = [0.0] * n, [-1.0] * n, {}

    def row(i):
        if i not in rows:
            rows[i] = rbf_row(cols, X[i], gamma)
        return rows[i]

    for used in range(1, max_iter + 1):
        i, gmax = -1, -math.inf
        for t in range(n):
            if (s[t] > 0 and a[t] < C) or (s[t] < 0 and a[t] > 0):
                v = -s[t] * G[t]
                if v > gmax:
                    gmax, i = v, t
        Ki = row(i)
        j, gmax2, best = -1, -math.inf, math.inf
        for t in range(n):
            if (s[t] > 0 and a[t] > 0) or (s[t] < 0 and a[t] < C):
                v = s[t] * G[t]
                if v > gmax2:
                    gmax2 = v
                b = gmax + v
                if b > 0:
                    obj = -b * b / max(2.0 - 2.0 * Ki[t], 1e-12)
                    if obj < best:
                        best, j = obj, t
        if gmax + gmax2 < tol or j < 0:
            break
        Kj = row(j)
        b = -s[i] * G[i] + s[j] * G[j]
        quad = max(2.0 - 2.0 * Ki[j], 1e-12)
        ai, aj = a[i], a[j]
        total = s[i] * ai + s[j] * aj
        a[i] = min(max(ai + s[i] * b / quad, 0.0), C)
        a[j] = min(max(s[j] * (total - s[i] * a[i]), 0.0), C)
        a[i] = s[i] * (total - s[j] * a[j])
        di, dj = s[i] * (a[i] - ai), s[j] * (a[j] - aj)
        G = [g + st * (di * u + dj * w) for g, st, u, w in zip(G, s, Ki, Kj)]
    free = [s[t] * G[t] for t in range(n) if 0 < a[t] < C]
    if free:
        rho = mean(free)
    else:
        ub = min(s[t] * G[t] for t in range(n) if (s[t] > 0 and a[t] == C) or (s[t] < 0 and a[t] == 0))
        lb = max(s[t] * G[t] for t in range(n) if (s[t] > 0 and a[t] == 0) or (s[t] < 0 and a[t] == C))
        rho = (ub + lb) / 2
    sv = [t for t in range(n) if a[t] > 0]
    return dict(sv=[X[t] for t in sv], coef=[a[t] * s[t] for t in sv], rho=rho, gamma=gamma, iters=used)


def rbf_decision(f, X):
    cols = transpose(f["sv"])
    return [dot(f["coef"], rbf_row(cols, r, f["gamma"])) - f["rho"] for r in X]


# ---------- Tuning, on the training days only ----------

def cv(X, y, fit_score):
    return mean([fit_score(X[:a], y[:a], X[a:b], y[a:b]) for a, b in time_folds(len(X))])


def logspace(lo, hi, n):
    return [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def rbf_cell(job):
    X, y, C, gamma = job
    return (C, gamma), cv(X, y, lambda a, b, c, d: auc(d, rbf_decision(fit_rbf_svm(a, b, C, gamma), c)))


def run_tuning(XA, XB, XC, yA, yB, yc, with_rbf, old):
    T, t0 = {}, time.time()
    say = lambda name, v: print(f"  {name}: {v['best'] if isinstance(v, dict) else v}   ({time.time() - t0:.0f}s)", flush=True)
    for key, X, y in [("ridge_a", XA, yA), ("ridge_b_raw", XC, yB), ("ridge_b_eng", XB, yB)]:
        T[key] = tune(lambda al, X=X, y=y: -cv(X, y, lambda a, b, c, d: mse(d, predict(fit_ridge(a, b, al), c))), logspace(-3, 5, 17))
        say(key, T[key])
    for key, X, y in [("huber_a", XA, yA), ("huber_b_eng", XB, yB)]:
        grid = (1.05, 1.15, 1.35, 1.75, 2.5, 5.0)
        scores = {e: cv(X, y, lambda a, b, c, d, e=e: mse(d, predict_huber(fit_huber(a, b, e), c))) for e in grid}
        T[key] = dict(best=min(scores, key=scores.get), grid=list(grid), score=[scores[e] for e in grid])
        say(key, T[key])
    T["logistic"] = tune(lambda c_: cv(XB, yc, lambda a, b, c, d: auc(d, laplace_pred(laplace_fit(a, b, 1 / c_), c, moderated=False))), logspace(-4, 2, 13))
    say("logistic C", T["logistic"])
    grid = [10.0 ** k for k in range(-2, 7)]
    score_a = lambda al: cv(XB, yc, lambda a, b, c, d: auc(d, laplace_pred(laplace_fit(a, b, al), c)))
    scores = {a: score_a(a) for a in grid}
    plateau, top = False, None
    while max(scores, key=scores.get) == max(grid) and max(grid) < 1e10:
        top = max(grid)
        grid.append(top * 100)
        scores[grid[-1]] = score_a(grid[-1])
        if abs(scores[grid[-1]] - scores[top]) <= 1e-4 * scores[top]:
            plateau = True
            break
    T["laplace"] = dict(best=top if plateau else max(scores, key=scores.get), plateau=plateau, grid=grid, score=[scores[a] for a in grid])
    say("Bayesian logistic α", T["laplace"])
    T["linear_svm"] = tune(lambda c_: cv(XB, yc, lambda a, b, c, d: auc(d, predict(fit_linear_svm(a, b, c_), c))), logspace(-4, 1, 6), max_ext=2)
    say("linear SVM C", T["linear_svm"])
    if with_rbf:
        jobs = [(XB, yc, C, g) for C in (0.01, 0.1, 1.0, 10.0) for g in (1e-4, 1e-3, 1e-2, 0.1)]
        with multiprocessing.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            done = {}
            for key, val in pool.imap_unordered(rbf_cell, jobs):
                done[key] = val
                print(f"    RBF C={key[0]} γ={key[1]}: CV AUC {val:.5f}   ({time.time() - t0:.0f}s)", flush=True)
        best = max(sorted(done), key=lambda k: done[k])
        T["rbf"] = dict(C=best[0], gamma=best[1], cv=done[best], searched=True, table=[dict(C=k[0], gamma=k[1], cv_auc=v) for k, v in sorted(done.items())])
    else:
        T["rbf"] = old.get("rbf") or notebook_rbf()
    return T


def notebook_rbf():
    with open(os.path.join(RESULTS, "results.json")) as f:
        r = json.load(f)["u2_clf"]["rbf"]
    return dict(C=r["C"], gamma=r["gamma"], searched=False)


# ---------- The run ----------

def main():
    t0 = time.time()
    panel = load_panel()
    A = audit(panel)
    tr, te = split(features(panel, "RELIANCE"))
    print(f"{A['tickers']} stocks, {A['n_returns']:,} returns, {A['n_ca']} corporate-action days, {A['n_hole']} data holes")
    print(f"Train {tr['date'][0]} → {tr['date'][-1]} ({len(tr['date'])} days) | Test {te['date'][0]} → {te['date'][-1]} ({len(te['date'])} days)")
    assert (A["tickers"], A["n_returns"], A["n_ca"], A["n_hole"]) == (49, 235047, 94, 4) and (len(tr["date"]), len(te["date"])) == (4405, 822)

    XB, XBt = standardise(tr["ENG"], te["ENG"])
    XA, XAt = standardise(tr["RAW"], te["RAW"])
    XC, XCt = standardise(tr["RET"], te["RET"])
    yA, yB, yc, yct = tr["close_next"], tr["r_next"], tr["y"], te["y"]
    c_t, y_true, r_true = te["close_t"], te["close_next"], te["r_next"]

    old = {}
    if os.path.exists(TUNED):
        with open(TUNED) as f:
            old = json.load(f)
    if "--tune" in sys.argv or "--tune-rbf" in sys.argv or not old:
        print("Tuning on the training days:")
        T = run_tuning(XA, XB, XC, yA, yB, yc, "--tune-rbf" in sys.argv, old)
        with open(TUNED, "w") as f:
            json.dump(T, f, indent=1)
    else:
        T = old

    # ----- Next-day closing price -----
    REG, out = {}, {}

    def rec(key, label, target, feats, hat):
        hat = list(hat) if target != "B" else [c * math.exp(p) for c, p in zip(c_t, hat)]
        e = [a - b for a, b in zip(y_true, hat)]
        e0 = [a - b for a, b in zip(y_true, c_t)]
        dm = (None, None) if key == "persistence" else dm_test(e, e0)
        REG[key] = dict(key=key, model=label, target=target, features=feats, rmse=math.sqrt(mean([v * v for v in e])), mae=mean(list(map(abs, e))),
                        mape=100 * mean([abs(v / t) for v, t in zip(e, y_true)]), dm_t=dm[0], dm_p=dm[1])

    rec("persistence", "Persistence", "baseline", "none", c_t)
    for key, X, Xt, tgt, feats in [("ols_a", XA, XAt, "A", "RAW"), ("ols_b_raw", XC, XCt, "B", "RAW-returns"), ("ols_b_eng", XB, XBt, "B", "ENGINEERED")]:
        rec(key, "OLS", tgt, feats, predict(fit_ridge(X, yA if tgt == "A" else yB), Xt))
    out["ols"] = ols_as_mle(XB, yB)
    huber = {}
    for key, X, Xt, tgt, feats in [("huber_a", XA, XAt, "A", "RAW"), ("huber_b_eng", XB, XBt, "B", "ENGINEERED")]:
        m = fit_huber(X, yA if tgt == "A" else yB, T[key]["best"])
        rec(key, "Huber", tgt, feats, predict_huber(m, Xt))
        huber[key] = dict(epsilon=T[key]["best"], outliers=m["outliers"])
    for key, X, Xt, tgt, feats in [("ridge_a", XA, XAt, "A", "RAW"), ("ridge_b_raw", XC, XCt, "B", "RAW-returns"), ("ridge_b_eng", XB, XBt, "B", "ENGINEERED")]:
        rec(key, "Ridge", tgt, feats, predict(fit_ridge(X, yA if tgt == "A" else yB, T[key]["best"]), Xt))
    blr_b, blr_a = blr_evidence(XB, yB), blr_evidence(XA, yA)
    mu_b, var_b = blr_predict(blr_b, XBt)
    mu_a, var_a = blr_predict(blr_a, XAt)
    rec("blr_b_eng", "Bayesian LR", "B", "ENGINEERED", mu_b)
    rec("blr_a", "Bayesian LR", "A", "RAW", mu_a)
    cover_b = mean([float(c * math.exp(m - Z95 * math.sqrt(v)) <= t <= c * math.exp(m + Z95 * math.sqrt(v))) for c, m, v, t in zip(c_t, mu_b, var_b, y_true)])
    cover_a = mean([float(m - Z95 * math.sqrt(v) <= t <= m + Z95 * math.sqrt(v)) for m, v, t in zip(mu_a, var_a, y_true)])
    out["blr"] = dict(b={k: blr_b[k] for k in ("alpha", "beta", "gamma", "iters")}, a={k: blr_a[k] for k in ("alpha", "beta", "gamma", "iters")}, cover_b=cover_b, cover_a=cover_a)
    out["huber"] = huber
    out["ridge"] = {k: dict(best=T[k]["best"], at_edge=T[k]["at_edge"], plateau=T[k]["plateau"]) for k in ("ridge_a", "ridge_b_raw", "ridge_b_eng")}
    out["reg"] = list(REG.values())
    print(f"\n{'Next-day close':<34}{'RMSE':>9}{'MAE':>9}{'MAPE %':>9}{'DM p':>8}")
    for r in sorted(REG.values(), key=lambda v: v["rmse"]):
        print(f"{r['model'] + ' [' + r['target'] + '|' + r['features'] + ']':<34}{r['rmse']:>9.4f}{r['mae']:>9.4f}{r['mape']:>9.4f}{'' if r['dm_p'] is None else format(r['dm_p'], '.3f'):>8}")
    print(f"OLS = MLE: gradient ascent reaches the closed form to {out['ols']['mle_diff']:.1e} in {out['ols']['iters']} steps. BLR: α = {blr_b['alpha']:.1f}, β = {blr_b['beta']:.2f}, γ = {blr_b['gamma']:.3f}. 95% interval covers {100 * cover_b:.1f}% of test days.")

    # ----- Up or down -----
    maj_class = int(mean(yc) > 0.5)
    maj_rate = mean([float(t == maj_class) for t in yct])
    n = len(yct)
    CLF = {}

    def clf(key, label, yhat, score=None, prob=None, cv_auc=None, kind="model"):
        k = sum(a == b for a, b in zip(yhat, yct))
        CLF[key] = dict(key=key, model=label, kind=kind, accuracy=k / n, n_correct=k, f1=f1(yct, yhat), auc=None if score is None else auc(yct, score),
                        log_loss=None if prob is None else log_loss(yct, prob), p_vs_majority=binom_greater(k, n, maj_rate), share_up=mean(yhat), cv_auc=cv_auc)

    clf("majority", "Majority class", [maj_class] * n, prob=[mean(yc)] * n, kind="baseline")
    lag = [float(v > 0) for v in te["r_lag0"]]
    clf("same_as_today", "Same as today", [int(v) for v in lag], score=lag, kind="baseline")
    f = fit_lda(XB, yc)
    d = predict(f, XBt)
    clf("lda", "LDA", [int(v > 0) for v in d], d, [sigmoid(v) for v in d], cv(XB, yc, lambda a, b, c, e: auc(e, predict(fit_lda(a, b), c))))
    f = fit_gnb(XB, yc)
    p = gnb_prob(f, XBt)
    clf("gnb", "Gaussian NB", [int(v > 0.5) for v in p], p, p, cv(XB, yc, lambda a, b, c, e: auc(e, gnb_prob(fit_gnb(a, b), c))))
    f = laplace_fit(XB, yc, 1 / T["logistic"]["best"])
    p = laplace_pred(f, XBt, moderated=False)
    clf("logistic", "Logistic regression", [int(v > 0.5) for v in p], p, p)
    lap = laplace_fit(XB, yc, T["laplace"]["best"])
    p = laplace_pred(lap, XBt)
    clf("bayes_logistic", "Bayesian logistic (Laplace)", [int(v > 0.5) for v in p], p, p)
    eig_H = eigh(lap["H"], vectors=False)
    f = fit_linear_svm(XB, yc, T["linear_svm"]["best"])
    d = predict(f, XBt)
    clf("linear_svm", "Linear SVM", [int(v > 0) for v in d], d)
    print(f"\nFitting the RBF SVM on {len(XB)} days (C = {T['rbf']['C']}, γ = {T['rbf']['gamma']}) ...", flush=True)
    f = fit_rbf_svm(XB, yc, T["rbf"]["C"], T["rbf"]["gamma"])
    d = rbf_decision(f, XBt)
    clf("rbf_svm", "RBF SVM", [int(v > 0) for v in d], d)
    x2, z2 = [1.5, -0.5], [0.5, 2.0]
    phi = lambda v: [v[0] ** 2, math.sqrt(2) * v[0] * v[1], v[1] ** 2]
    out["clf"] = list(CLF.values())
    out["maj_rate"] = maj_rate
    out["laplace"] = dict(alpha=lap["alpha"], iters=lap["iters"], min_eig_H=eig_H[-1], max_eig_H=eig_H[0])
    out["tuned"] = dict(logistic_C=T["logistic"]["best"], linear_svm_C=T["linear_svm"]["best"], rbf_C=T["rbf"]["C"], rbf_gamma=T["rbf"]["gamma"], rbf_searched=T["rbf"].get("searched", False), n_sv=len(f["coef"]), smo_iters=f["iters"])
    out["kernel"] = dict(x=x2, z=z2, lhs=dot(x2, z2) ** 2, rhs=dot(phi(x2), phi(z2)))
    print(f"{'Up or down':<34}{'Accuracy':>10}{'AUC':>9}{'F1':>8}{'p vs majority':>15}")
    for r in CLF.values():
        print(f"{r['model']:<34}{r['accuracy']:>10.4f}{'' if r['auc'] is None else format(r['auc'], '.4f'):>9}{r['f1']:>8.4f}{r['p_vs_majority']:>15.4f}")
    print(f"Majority rate on the test days: {maj_rate:.4f}. Models significantly above it: {sum(r['kind'] == 'model' and r['p_vs_majority'] < 0.05 for r in CLF.values())} of {sum(r['kind'] == 'model' for r in CLF.values())}.")
    print(f"Kernel trick, x = {x2}, z = {z2}: (xᵀz)² = {out['kernel']['lhs']:.6f}, φ(x)ᵀφ(z) = {out['kernel']['rhs']:.6f}")

    out["seconds"] = time.time() - t0
    with open(os.path.join(RESULTS, "from_scratch_unit2.json"), "w") as f_:
        json.dump(out, f_, indent=1)
    print(f"\nDone in {out['seconds']:.0f}s. Saved results/from_scratch_unit2.json")


if __name__ == "__main__":
    main()
