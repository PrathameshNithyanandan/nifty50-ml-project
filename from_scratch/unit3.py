"""Unit 3 with no libraries: clustering NIFTY-50 stocks by return and volatility behaviour, then PCA market factors.
Equation numbers refer to MATHEMATICAL_MODEL.md.

    python3 from_scratch/unit3.py
"""
import json
import math
import os
import random
import time

from dataset import RESULTS, industries, load_panel, market_return
from matrix import add_diag, cholesky, dot, eigh, forward, gram, identity, inverse, log_det, matmul, matvec, mean, standardise, transpose, variance

SEED = 42
FEAT6 = ["ann_return", "ann_vol", "beta", "skew", "ex_kurt", "max_dd"]
FEAT2 = ["ann_return", "ann_vol"]
KS = list(range(2, 9))
COV_TYPES = ["spherical", "diag", "tied", "full"]
TRADING_DAYS = 252


# ---------- Stock features ----------

def stock_features(panel):
    mkt = market_return(panel)
    rows = {}
    for tk, d in panel.items():
        kept = [(day, x) for day, x in zip(d["date"], d["r"]) if x is not None]
        r = [x for _, x in kept]
        m = [mkt[day] for day, _ in kept]
        n, mr, mm = len(r), mean(r), mean(m)
        m2, m3, m4 = (sum((x - mr) ** p for x in r) / n for p in (2, 3, 4))
        peak, dd = -math.inf, 0.0
        for a in d["adj"]:
            peak = max(peak, a)
            dd = max(dd, 1 - a / peak)
        beta = sum((a - mr) * (b - mm) for a, b in zip(r, m)) / (n - 1) / variance(m, 1)
        rows[tk] = dict(ann_return=mr * TRADING_DAYS, ann_vol=math.sqrt(variance(r, 1) * TRADING_DAYS), beta=beta, skew=m3 / m2 ** 1.5, ex_kurt=m4 / m2 ** 2 - 3, max_dd=dd, n=n)
    return rows


def sq_dist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


# ---------- K-means ----------

def kmeans_pp(X, k, rng):
    """Spreads the starting centres: each new centre is drawn with probability proportional to its squared distance."""
    mu = [X[rng.randrange(len(X))]]
    while len(mu) < k:
        d = [min(sq_dist(x, m) for m in mu) for x in X]
        u, total, chosen = rng.random() * sum(d), 0.0, len(X) - 1
        for i, v in enumerate(d):
            total += v
            if total >= u:
                chosen = i
                break
        mu.append(X[chosen])
    return [m[:] for m in mu]


def lloyd(X, mu, iters=300):
    """Alternates the assignment and mean steps of equations 3.2 and 3.3 until nothing changes."""
    labels, inertia = None, math.inf
    for _ in range(iters):
        pairs = [min((sq_dist(x, m), j) for j, m in enumerate(mu)) for x in X]
        new = [j for _, j in pairs]
        inertia = sum(d for d, _ in pairs)
        if new == labels:
            break
        labels = new
        for j in range(len(mu)):
            part = [x for x, l in zip(X, labels) if l == j]
            if part:
                mu[j] = [mean(c) for c in zip(*part)]
    return labels, mu, inertia


def kmeans(X, k, restarts, rng):
    return min((lloyd(X, kmeans_pp(X, k, rng)) for _ in range(restarts)), key=lambda t: t[2])


# ---------- Cluster evaluation ----------

def distance_matrix(X):
    return [[math.sqrt(sq_dist(a, b)) for b in X] for a in X]


def silhouette(D, labels):
    """Equation 3.21. A stock alone in its cluster scores 0."""
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    if len(groups) < 2:
        return None
    total = 0.0
    for i, l in enumerate(labels):
        if len(groups[l]) > 1:
            a = sum(D[i][j] for j in groups[l] if j != i) / (len(groups[l]) - 1)
            b = min(sum(D[i][j] for j in g) / len(g) for c, g in groups.items() if c != l)
            total += (b - a) / max(a, b)
    return total / len(labels)


def davies_bouldin(X, labels):
    """Equation 3.22."""
    groups = {}
    for x, l in zip(X, labels):
        groups.setdefault(l, []).append(x)
    if len(groups) < 2:
        return None
    cen = {l: [mean(c) for c in zip(*g)] for l, g in groups.items()}
    s = {l: mean([math.sqrt(sq_dist(x, cen[l])) for x in g]) for l, g in groups.items()}
    return mean([max((s[a] + s[b]) / math.sqrt(sq_dist(cen[a], cen[b])) for b in groups if b != a) for a in groups])


def adjusted_rand(a, b):
    """Equation 3.23: pair agreement corrected for chance."""
    table = {}
    for x, y in zip(a, b):
        table[(x, y)] = table.get((x, y), 0) + 1
    row, col = {}, {}
    for (x, y), v in table.items():
        row[x] = row.get(x, 0) + v
        col[y] = col.get(y, 0) + v
    c2 = lambda v: v * (v - 1) / 2
    sij, si, sj, tot = sum(map(c2, table.values())), sum(map(c2, row.values())), sum(map(c2, col.values())), c2(len(a))
    exp = si * sj / tot
    high = (si + sj) / 2
    return 1.0 if high == exp else (sij - exp) / (high - exp)


def relabel(labels, vol):
    """Numbers the clusters by mean volatility, so a cluster keeps its number across methods."""
    groups = {}
    for l, v in zip(labels, vol):
        groups.setdefault(l, []).append(v)
    order = sorted(groups, key=lambda l: mean(groups[l]))
    idx = {l: i for i, l in enumerate(order)}
    return [idx[l] for l in labels]


# ---------- Gaussian mixture by EM ----------

def n_params(cov_type, K, D):
    return K * D + K - 1 + dict(spherical=K, diag=K * D, tied=D * (D + 1) // 2, full=K * D * (D + 1) // 2)[cov_type]


def e_step(X, pi, mu, cov):
    """Responsibilities and log-likelihood (equations 3.5 and 3.6). Works in logs to avoid underflow."""
    D, lp = len(X[0]), []
    chol = [cholesky(c) for c in cov]
    half = [sum(math.log(L[i][i]) for i in range(D)) for L in chol]
    for x in X:
        row = []
        for k, L in enumerate(chol):
            z = forward(L, [a - b for a, b in zip(x, mu[k])])
            row.append(math.log(pi[k]) - 0.5 * D * math.log(2 * math.pi) - half[k] - 0.5 * dot(z, z))
        lp.append(row)
    ll, resp = 0.0, []
    for row in lp:
        top = max(row)
        s = top + math.log(sum(math.exp(v - top) for v in row))
        ll += s
        resp.append([math.exp(v - s) for v in row])
    return resp, ll, lp


def m_step(X, resp, cov_type, reg):
    """Equations 3.7 to 3.9, with the covariance restricted by its family."""
    N, D, K = len(X), len(X[0]), len(resp[0])
    Nk = [sum(r[k] for r in resp) + 2.3e-15 for k in range(K)]
    pi = [v / N for v in Nk]
    mu = [[sum(r[k] * x[d] for r, x in zip(resp, X)) / Nk[k] for d in range(D)] for k in range(K)]
    full = []
    for k in range(K):
        S = [[0.0] * D for _ in range(D)]
        for x, r in zip(X, resp):
            e = [a - b for a, b in zip(x, mu[k])]
            w = r[k] / Nk[k]
            for i in range(D):
                for j in range(i + 1):
                    S[i][j] += w * e[i] * e[j]
        for i in range(D):
            for j in range(i):
                S[j][i] = S[i][j]
        full.append(S)
    if cov_type == "full":
        cov = [add_diag(S, reg) for S in full]
    elif cov_type == "tied":
        T = [[sum(Nk[k] * full[k][i][j] for k in range(K)) / N + (reg if i == j else 0.0) for j in range(D)] for i in range(D)]
        cov = [T] * K
    elif cov_type == "diag":
        cov = [[[S[i][i] + reg if i == j else 0.0 for j in range(D)] for i in range(D)] for S in full]
    else:
        cov = [[[sum(S[d][d] for d in range(D)) / D + reg if i == j else 0.0 for j in range(D)] for i in range(D)] for S in full]
    return pi, mu, cov


def em_gmm(X, labels, K, cov_type, reg=1e-3, max_iter=1000, tol=1e-3):
    """EM from a hard starting assignment. tol is on the mean log-likelihood per stock."""
    resp = [[1.0 if l == k else 0.0 for k in range(K)] for l in labels]
    pi, mu, cov = m_step(X, resp, cov_type, reg)
    prev, hist = -math.inf, []
    for _ in range(max_iter):
        resp, ll, lp = e_step(X, pi, mu, cov)
        # with q set to the responsibilities the lower bound touches the log-likelihood (equation 3.10)
        bound = sum(r * (v - math.log(r)) for rr, row in zip(resp, lp) for r, v in zip(rr, row) if r > 0)
        hist.append(dict(ll=ll, gap=ll - bound))
        if abs(ll - prev) / len(X) < tol:
            break
        prev = ll
        pi, mu, cov = m_step(X, resp, cov_type, reg)
    return dict(pi=pi, mu=mu, cov=cov, resp=resp, ll=ll, hist=hist)


def gmm(X, K, cov_type, rng, restarts=100, tol=1e-3):
    best = None
    for _ in range(restarts):
        labels, _, _ = lloyd(X, kmeans_pp(X, K, rng))
        f = em_gmm(X, labels, K, cov_type, tol=tol)
        if best is None or f["ll"] > best["ll"]:
            best = f
    return best


def soft_kmeans(X, mu, eps, iters=5000):
    """A mixture with equal weights and fixed covariance εI: K-means is its limit as ε → 0 (equation 3.11)."""
    resp = None
    for _ in range(iters):
        resp = []
        for x in X:
            lp = [-sq_dist(x, m) / (2 * eps) for m in mu]
            top = max(lp)
            r = [math.exp(v - top) for v in lp]
            s = sum(r)
            resp.append([v / s for v in r])
        new = []
        for k in range(len(mu)):
            s = sum(r[k] for r in resp)
            new.append([sum(r[k] * x[d] for r, x in zip(resp, X)) / s if s > 0 else mu[k][d] for d in range(len(X[0]))])
        done = max(abs(a - b) for m, n_ in zip(mu, new) for a, b in zip(m, n_)) < 1e-12
        mu = new
        if done:
            break
    return resp, mu


# ---------- Hierarchical clustering ----------

def linkage(D, method):
    """Agglomerative merging with the Lance–Williams update (equations 3.13 to 3.15). Returns merges in order of height."""
    n = len(D)
    active = {i: [i] for i in range(n)}
    d = {i: {j: D[i][j] for j in range(n) if j != i} for i in range(n)}
    merges, nxt = [], n
    while len(active) > 1:
        h, a, b = min((d[i][j], i, j) for i in active for j in active if i < j)
        na, nb = len(active[a]), len(active[b])
        new = {}
        for c in active:
            if c in (a, b):
                continue
            dac, dbc, nc = d[a][c], d[b][c], len(active[c])
            if method == "complete":
                new[c] = max(dac, dbc)
            elif method == "average":
                new[c] = (na * dac + nb * dbc) / (na + nb)
            else:
                new[c] = math.sqrt(((na + nc) * dac * dac + (nb + nc) * dbc * dbc - nc * h * h) / (na + nb + nc))
        merges.append((h, active[a], active[b]))
        active[nxt] = active.pop(a) + active.pop(b)
        d[nxt] = new
        for c, v in new.items():
            d[c][nxt] = v
        nxt += 1
    return merges


def cut(merges, n, k):
    """Undoes the k−1 highest merges, which leaves k clusters."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, a, b in merges[:n - k]:
        parent[find(a[0])] = find(b[0])
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def cophenetic(D, merges):
    """Correlation between the original distances and the height at which each pair first joins (equation 3.16)."""
    n, height = len(D), {}
    for h, a, b in merges:
        for i in a:
            for j in b:
                height[(min(i, j), max(i, j))] = h
    x = [D[i][j] for i in range(n) for j in range(i + 1, n)]
    y = [height[(i, j)] for i in range(n) for j in range(i + 1, n)]
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


# ---------- PCA and factor analysis ----------

def covariance_columns(C, ddof):
    n = len(C[0])
    return [[v / (n - ddof) for v in r] for r in gram(C)]


def factor_analysis(S, N, K, iters=20000, tol=1e-6):
    """EM for x = Λz + μ + ε with diagonal noise Ψ (equations 3.19 and 3.20), working from the sample covariance S."""
    P = len(S)
    val, vec = eigh(S)
    Lam = [[vec[k][i] * math.sqrt(val[k]) for k in range(K)] for i in range(P)]
    psi = [max(S[i][i] - dot(Lam[i], Lam[i]), 1e-6) for i in range(P)]

    def loglik(Lam, psi):
        # the matrix determinant lemma and Woodbury identity reduce the work from P×P to K×K
        M = add_diag([[sum(Lam[i][a] * Lam[i][b] / psi[i] for i in range(P)) for b in range(K)] for a in range(K)], 1.0)
        Mi = inverse(M)
        ld = sum(map(math.log, psi)) + log_det(M)
        B = [[Lam[i][a] / psi[i] for a in range(K)] for i in range(P)]
        SB = matmul(S, B)
        tr = sum(S[i][i] / psi[i] for i in range(P)) - sum(Mi[a][b] * sum(B[i][a] * SB[i][b] for i in range(P)) for a in range(K) for b in range(K))
        return -0.5 * N * (P * math.log(2 * math.pi) + ld + tr)

    prev, used = -math.inf, 0
    for used in range(1, iters + 1):
        M = add_diag([[sum(Lam[i][a] * Lam[i][b] / psi[i] for i in range(P)) for b in range(K)] for a in range(K)], 1.0)
        G = inverse(M)
        beta = matmul(G, transpose([[Lam[i][a] / psi[i] for a in range(K)] for i in range(P)]))
        SB = matmul(S, transpose(beta))
        Ezz = [[G[a][b] + sum(beta[a][i] * SB[i][b] for i in range(P)) for b in range(K)] for a in range(K)]
        Lam = matmul(SB, inverse(Ezz))
        psi = [max(S[i][i] - dot(Lam[i], SB[i]), 1e-9) for i in range(P)]
        if used % 10 == 0:
            ll = loglik(Lam, psi)
            if abs(ll - prev) < tol * abs(ll) * 1e-3:
                break
            prev = ll
    return dict(Lam=Lam, psi=psi, ll=loglik(Lam, psi), iters=used)


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    panel = load_panel()
    ind = industries()
    names = sorted(panel)
    SF = stock_features(panel)
    vol = [SF[t]["ann_vol"] for t in names]
    X6 = standardise([[SF[t][c] for c in FEAT6] for t in names])[0]
    X2 = standardise([[SF[t][c] for c in FEAT2] for t in names])[0]
    SPACES = {"6d": X6, "2d": X2}
    DIST = {sp: distance_matrix(X) for sp, X in SPACES.items()}
    IND = [ind[t] for t in names]
    print(f"{len(names)} stocks with features {FEAT6}")
    out = dict(features={t: SF[t] for t in names})

    # ----- K-means -----
    km_rows, LABELS = [], {"kmeans": {}, "gmm": {}, "hier": {}}
    for sp, X in SPACES.items():
        LABELS["kmeans"][sp] = {}
        for k in KS:
            lab, _, inertia = kmeans(X, k, 500, rng)
            lab = relabel(lab, vol)
            LABELS["kmeans"][sp][k] = lab
            sizes = [lab.count(c) for c in set(lab)]
            km_rows.append(dict(space=sp, k=k, inertia=inertia, silhouette=silhouette(DIST[sp], lab), db=davies_bouldin(X, lab), min_size=min(sizes), max_size=max(sizes)))
    chosen = lambda sp: max((r for r in km_rows if r["space"] == sp and r["min_size"] >= 3), key=lambda r: r["silhouette"])["k"]
    K_CHOSEN, K_CHOSEN_2D = chosen("6d"), chosen("2d")
    out.update(kmeans=km_rows, k_chosen=K_CHOSEN, k_chosen_2d=K_CHOSEN_2D)
    print(f"\n{'k':>3}{'inertia':>11}{'silhouette':>12}{'DB':>8}{'smallest':>10}   (K-means, 6 features)")
    for r in km_rows:
        if r["space"] == "6d":
            print(f"{r['k']:>3}{r['inertia']:>11.4f}{r['silhouette']:>12.4f}{r['db']:>8.4f}{r['min_size']:>10}")
    print(f"Silhouette with at least 3 stocks per cluster picks k = {K_CHOSEN}. [{time.time() - t0:.0f}s]", flush=True)

    # ----- GMM: model selection, EM and the K-means limit -----
    gm_rows = []
    for sp, X in SPACES.items():
        for ct in COV_TYPES:
            for k in range(1, 9):
                f = gmm(X, k, ct, rng)
                p = n_params(ct, k, len(X[0]))
                gm_rows.append(dict(space=sp, cov=ct, k=k, ll=f["ll"], n_params=p, bic=-2 * f["ll"] + p * math.log(len(X)), aic=-2 * f["ll"] + 2 * p))
    pick = lambda sp, key: min((r for r in gm_rows if r["space"] == sp), key=lambda r: r[key])
    b6, a6, b2, a2 = pick("6d", "bic"), pick("6d", "aic"), pick("2d", "bic"), pick("2d", "aic")
    for sp, row in (("6d", b6), ("2d", b2)):
        LABELS["gmm"][sp] = {}
        for k in KS:
            f = gmm(SPACES[sp], k, row["cov"], rng)
            LABELS["gmm"][sp][k] = relabel([r.index(max(r)) for r in f["resp"]], vol)
    em_k = max(b6["k"], 2)
    em = gmm(X6, em_k, b6["cov"], rng, restarts=20, tol=1e-12)
    ll_h = [h["ll"] for h in em["hist"]]
    steps = [b - a for a, b in zip(ll_h, ll_h[1:])]
    out["gmm"] = dict(bic_cov=b6["cov"], bic_k=b6["k"], bic=b6["bic"], bic_params=b6["n_params"], aic_cov=a6["cov"], aic_k=a6["k"], aic=a6["aic"], aic_params=a6["n_params"], bic2_cov=b2["cov"], bic2_k=b2["k"], aic2_cov=a2["cov"], aic2_k=a2["k"], em_k=em_k, em_cov=b6["cov"], em_ll=ll_h[-1], em_ll0=ll_h[0], em_iters=len(ll_h), em_monotone=all(s >= -1e-9 for s in steps), em_gap=max(abs(h["gap"]) for h in em["hist"]), rows=gm_rows)
    print(f"\nGMM: BIC picks {b6['cov']} K = {b6['k']} (BIC {b6['bic']:.2f}); AIC picks {a6['cov']} K = {a6['k']} (AIC {a6['aic']:.2f}).")
    print(f"EM on {b6['cov']}, K = {em_k}: {len(ll_h)} iterations, log-likelihood {ll_h[0]:.4f} → {ll_h[-1]:.4f}, never falls: {out['gmm']['em_monotone']}, largest gap to the lower bound {out['gmm']['em_gap']:.1e}. [{time.time() - t0:.0f}s]", flush=True)
    start = kmeans_pp(X6, 3, rng)
    hard_lab, hard_mu, _ = lloyd(X6, [m[:] for m in start], 1000)
    hard = []
    for eps in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]:
        r_, _ = soft_kmeans(X6, [m[:] for m in start], eps)
        hard.append(dict(eps=eps, mean_max_resp=mean([max(r) for r in r_]), ari_vs_kmeans=adjusted_rand(hard_lab, [r.index(max(r)) for r in r_])))
    out["hard_limit"] = hard
    print(f"K-means as the limit: at ε = {hard[-1]['eps']} the mean largest responsibility is {hard[-1]['mean_max_resp']:.4f} and the agreement with K-means is {hard[-1]['ari_vs_kmeans']:.4f}.")

    # ----- Hierarchical clustering -----
    dates = sorted({x for d in panel.values() for x in d["date"]})
    pos = {x: i for i, x in enumerate(dates)}
    W = {t: [None] * len(dates) for t in names}
    for t in names:
        for x, r in zip(panel[t]["date"], panel[t]["r"]):
            W[t][pos[x]] = r

    def corr(a, b):
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        mx, my = mean(xs), mean(ys)
        return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))

    n = len(names)
    Rho = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Rho[i][j] = Rho[j][i] = corr(W[names[i]], W[names[j]])
    Dcorr = [[0.0 if i == j else math.sqrt(max(2 * (1 - Rho[i][j]), 0.0)) for j in range(n)] for i in range(n)]
    hier, Z = [], {}
    for dname, Dm in (("euclidean", DIST["6d"]), ("correlation", Dcorr)):
        for meth in ("ward", "complete", "average"):
            Z[(dname, meth)] = linkage(Dm, meth)
            hier.append(dict(distance=dname, linkage=meth, cophenetic=cophenetic(Dm, Z[(dname, meth)])))
    Z2 = linkage(DIST["2d"], "ward")
    LABELS["hier"]["6d"] = {k: relabel(cut(Z[("euclidean", "ward")], n, k), vol) for k in KS}
    LABELS["hier"]["2d"] = {k: relabel(cut(Z2, n, k), vol) for k in KS}
    out["hier"] = hier
    metrics = []
    for meth in LABELS:
        for sp, X in SPACES.items():
            for k in KS:
                l_ = LABELS[meth][sp][k]
                metrics.append(dict(method=meth, space=sp, k=k, n_clusters=len(set(l_)), silhouette=silhouette(DIST[sp], l_), db=davies_bouldin(X, l_), ari_industry=adjusted_rand(IND, l_)))
    out["metrics"] = metrics
    L_km, L_gm, L_hc = (LABELS[m]["6d"][K_CHOSEN] for m in ("kmeans", "gmm", "hier"))
    L_cw, L_ca = cut(Z[("correlation", "ward")], n, K_CHOSEN), cut(Z[("correlation", "average")], n, K_CHOSEN)
    out["agreement"] = dict(km_gmm=adjusted_rand(L_km, L_gm), km_hier=adjusted_rand(L_km, L_hc), gmm_hier=adjusted_rand(L_gm, L_hc))
    out["ari"] = dict(km_6d=adjusted_rand(IND, L_km), gmm_6d=adjusted_rand(IND, L_gm), hier_6d=adjusted_rand(IND, L_hc), km_2d=adjusted_rand(IND, LABELS["kmeans"]["2d"][K_CHOSEN_2D]), corr_ward=adjusted_rand(IND, L_cw), corr_average=adjusted_rand(IND, L_ca))
    shuffled = IND[:]
    def perm_p(obs, lab, times=5000):
        hits = 0
        for _ in range(times):
            rng.shuffle(shuffled)
            hits += adjusted_rand(shuffled, lab) >= obs
        return (hits + 1) / (times + 1)
    out["ari_p"], out["ari_corr_p"] = perm_p(out["ari"]["km_6d"], L_km), perm_p(out["ari"]["corr_ward"], L_cw)
    out["sil_corr_ward"], out["sil_corr_avg"] = silhouette(Dcorr, L_cw), silhouette(Dcorr, L_ca)
    out["corr_members"] = {str(c): [t for t, l in zip(names, L_cw) if l == c] for c in sorted(set(L_cw))}
    print(f"\nAgreement with industry (ARI): on features {out['ari']['km_6d']:.3f} (p = {out['ari_p']:.3f}); on correlation distance {out['ari']['corr_ward']:.3f} (p = {out['ari_corr_p']:.4f}). [{time.time() - t0:.0f}s]", flush=True)

    # ----- PCA -----
    first = {t: next(x for x, r in zip(panel[t]["date"], panel[t]["r"]) if r is not None) for t in names}
    start_d = max(first.values())
    days = [i for i, x in enumerate(dates) if x >= start_d and all(W[t][i] is not None for t in names)]
    Xp = standardise([[W[t][i] for t in names] for i in days])[0]
    Np, P = len(Xp), n
    C = transpose(Xp)
    S1 = covariance_columns(C, 1)
    ev, vec = eigh(S1)
    total = sum(ev)
    evr = [v / total for v in ev]
    cum, n80, s = [], None, 0.0
    for i, v in enumerate(evr):
        s += v
        if s >= 0.8 and n80 is None:
            n80 = i + 1
    n_kaiser = sum(v > 1 for v in ev)
    print(f"\nPCA on {Np} days and {P} stocks: PC1 {evr[0]:.4f}, PC2 {evr[1]:.4f}, PC3 {evr[2]:.4f}. Parallel analysis: shuffling 200 times ...", flush=True)
    par = []
    for _ in range(200):
        Cs = [rng.sample(c, len(c)) for c in C]
        par.append(eigh(covariance_columns(Cs, 1), vectors=False))
    par95 = []
    for j in range(P):
        vals = sorted(r[j] for r in par)
        q = 0.95 * (len(vals) - 1)
        lo = int(q)
        par95.append(vals[lo] + (q - lo) * (vals[min(lo + 1, len(vals) - 1)] - vals[lo]))
    n_par = next((i for i in range(P) if ev[i] <= par95[i]), P)
    stick = [sum(1 / m for m in range(k, P + 1)) / P for k in range(1, P + 1)]
    n_bs = next((i for i in range(P) if evr[i] <= stick[i]), P)
    flip = 1 if sum(vec[0]) >= 0 else -1
    load1 = [flip * v * math.sqrt(ev[0]) for v in vec[0]]
    score1 = [flip * dot(r, vec[0]) for r in Xp]
    ew = [mean([W[t][i] for t in names]) for i in days]
    mx, my = mean(score1), mean(ew)
    pc1_corr = sum((a - mx) * (b - my) for a, b in zip(score1, ew)) / math.sqrt(sum((a - mx) ** 2 for a in score1) * sum((b - my) ** 2 for b in ew))
    out["pca"] = dict(start=dates[days[0]], end=dates[days[-1]], days=Np, stocks=P, pc1=evr[0], pc2=evr[1], pc3=evr[2], ev1=ev[0], n80=n80, n_kaiser=n_kaiser, n_parallel=n_par, n_broken=n_bs, pc1_corr=pc1_corr, pc1_all_pos=all(v > 0 for v in load1))
    print(f"Components: ≥80% variance {n80}, Kaiser {n_kaiser}, parallel analysis {n_par}, broken stick {n_bs}. Corr(PC1, equal-weighted index) = {pc1_corr:.5f}. [{time.time() - t0:.0f}s]", flush=True)

    # ----- Factor analysis, compared with probabilistic PCA -----
    K_F = n_par
    S0 = covariance_columns(C, 0)
    fa = factor_analysis(S0, Np, K_F)
    sigma2 = mean(ev[K_F:])
    # PPCA log-likelihood at its closed-form maximum (equations 3.17 and 3.18)
    log_det_C = sum(math.log(v) for v in ev[:K_F]) + (P - K_F) * math.log(sigma2)
    prec_quad = sum(dot(matvec(S0, vec[j]), vec[j]) * (1 / ev[j] - 1 / sigma2) for j in range(K_F)) + sum(S0[i][i] for i in range(P)) / sigma2
    ll_ppca = -0.5 * Np * (P * math.log(2 * math.pi) + log_det_C + prec_quad)
    kf, kp = P * K_F - K_F * (K_F - 1) / 2 + P, P * K_F - K_F * (K_F - 1) / 2 + 1
    out["fa"] = dict(k=K_F, iters=fa["iters"], ll_fa=fa["ll"], ll_ppca=ll_ppca, bic_fa=-2 * fa["ll"] + kf * math.log(Np), bic_ppca=-2 * ll_ppca + kp * math.log(Np), sigma2_ppca=sigma2, psi_min=min(fa["psi"]), psi_max=max(fa["psi"]), psi_mean=mean(fa["psi"]), psi_min_t=names[fa["psi"].index(min(fa["psi"]))])
    print(f"Factor analysis, {K_F} factors, {fa['iters']} EM steps: log-likelihood {fa['ll']:.1f} vs PPCA {ll_ppca:.1f}; noise ψ from {min(fa['psi']):.3f} to {max(fa['psi']):.3f}, PPCA σ² = {sigma2:.3f}.")

    out["seconds"] = time.time() - t0
    with open(os.path.join(RESULTS, "from_scratch_unit3.json"), "w") as f_:
        json.dump(out, f_, indent=1)
    print(f"\nDone in {out['seconds']:.0f}s. Saved results/from_scratch_unit3.json")


if __name__ == "__main__":
    main()
