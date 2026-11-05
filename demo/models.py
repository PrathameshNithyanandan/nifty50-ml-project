"""Every model in the live demo. Fitting code follows analysis.ipynb; tuned settings are read from results.json."""
from . import pipeline as P
import itertools, pathlib
from functools import reduce
from types import SimpleNamespace
import joblib
import numpy as np, pandas as pd
from scipy import optimize, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from pygam import LogisticGAM, s as gam_s, l as gam_l
from pygam.utils import OptimizationError
from hmmlearn.hmm import GaussianHMM

SEED, ENG, RAWC, RAWRET = P.SEED, P.ENG, P.RAWC, P.RAWRET
tscv = TimeSeriesSplit(n_splits=5)
Z95 = stats.norm.ppf(0.975)


# ---------- settings from the notebook ----------
def tuned(results):
    u2c, u2r, u5 = results["u2_clf"], results["u2_reg"], results["u5"]
    t = u5["tuning"]
    bag = t["bagging"]["best"]
    return dict(
        logistic=dict(C=u2c["logistic"]["C"]),
        bayes_logistic=dict(alpha=u2c["laplace"]["alpha"]),
        linear_svm=dict(C=u2c["linear_svm"]["C"]),
        rbf_svm=dict(C=u2c["rbf"]["C"], gamma=u2c["rbf"]["gamma"]),
        # a pruning threshold is compared exactly, so the rounded stored value is nudged up to keep the same tree
        cart=dict(ccp_alpha=t["cart"]["best"]["ccp_alpha"] * (1 + 1e-9), max_depth=None),
        bagging=dict(min_samples_leaf=bag["estimator__min_samples_leaf"], max_samples=bag["max_samples"]),
        random_forest=dict(t["random_forest"]["best"]),
        adaboost=dict(t["adaboost"]["best"]),
        gradient_boosting=dict(t["gradient_boosting"]["best"]),
        gam=dict(lam=u5["gam"]["lam"]),
        huber_a=dict(epsilon=u2r["huber"]["huber_a"]["epsilon"]),
        huber_b_eng=dict(epsilon=u2r["huber"]["huber_b_eng"]["epsilon"]),
        ridge_a=dict(alpha=u2r["ridge"]["ridge_a"]["best"]),
        ridge_b_raw=dict(alpha=u2r["ridge"]["ridge_b_raw"]["best"]),
        ridge_b_eng=dict(alpha=u2r["ridge"]["ridge_b_eng"]["best"]),
    )


# ---------- design matrices ----------
class Design:
    """Scalers are fitted on the training rows only, and applied unchanged to anything predicted later."""
    def __init__(self, tr):
        self.scA = StandardScaler().fit(tr[RAWC]); self.scB = StandardScaler().fit(tr[ENG]); self.scC = StandardScaler().fit(tr[RAWRET])

    def x(self, D):
        return SimpleNamespace(A=self.scA.transform(D[RAWC]), B=self.scB.transform(D[ENG]), C=self.scC.transform(D[RAWRET]), R=D[ENG].values, r_lag0=D["r_lag0"].values)


def split(D, cutoff):
    cutoff = pd.Timestamp(cutoff)
    return D[D.index < cutoff], D[D.index >= cutoff]


# ---------- hand-written estimators from the notebook ----------
class ScaledY:
    def __init__(self, base): self.base = base
    def fit(self, X, y):
        self.m, self.s = y.mean(), y.std(); self.base.fit(X, (y - self.m) / self.s); return self
    def predict(self, X): return self.base.predict(X) * self.s + self.m


def blr_evidence(X, y, iters=2000, tol=1e-12):
    ym = y.mean(); t = y - ym
    eig0, Q = np.linalg.eigh(X.T @ X); Xt = X.T @ t
    alpha, beta = 1.0, 1 / t.var()
    for _ in range(iters):
        lam = beta * eig0
        m = beta * Q @ ((Q.T @ Xt) / (alpha + lam))
        gamma = np.sum(lam / (alpha + lam))
        a_new = gamma / (m @ m); res = t - X @ m; b_new = (len(t) - gamma) / (res @ res)
        conv = abs(np.log(a_new / alpha)) + abs(np.log(b_new / beta)); alpha, beta = a_new, b_new
        if conv < tol: break
    lam = beta * eig0
    S = Q @ np.diag(1 / (alpha + lam)) @ Q.T
    return dict(m=beta * S @ Xt, S=S, alpha=alpha, beta=beta, ym=ym)


def blr_predict(fit, X):
    return X @ fit["m"] + fit["ym"], 1 / fit["beta"] + np.einsum("ij,jk,ik->i", X, fit["S"], X)


def sigmoid(a): return 1 / (1 + np.exp(-a))


def laplace_fit(X, t, alpha, iters=100, tol=1e-10):
    Phi = np.c_[np.ones(len(X)), X]; I_ = np.ones(Phi.shape[1]); I_[0] = 0.0
    w = np.zeros(Phi.shape[1])
    for _ in range(iters):
        y = sigmoid(Phi @ w); g = Phi.T @ (y - t) + alpha * I_ * w
        H = (Phi * (y * (1 - y))[:, None]).T @ Phi + np.diag(alpha * I_)
        step = np.linalg.solve(H, g); w = w - step
        if np.linalg.norm(step) < tol: break
    y = sigmoid(Phi @ w); H = (Phi * (y * (1 - y))[:, None]).T @ Phi + np.diag(alpha * I_)
    return dict(w=w, S=np.linalg.inv(H))


def laplace_pred(X, f):
    Phi = np.c_[np.ones(len(X)), X]; mu = Phi @ f["w"]; s2 = np.einsum("ij,jk,ik->i", Phi, f["S"], Phi)
    return sigmoid((1 + np.pi * s2 / 8) ** -.5 * mu)


def irls_fit(Phi, t, iters=25, tol=1e-8):
    w = np.zeros(Phi.shape[1]); w[0] = np.log(t.mean() / (1 - t.mean()))
    for _ in range(iters):
        a = Phi @ w; y = 1 / (1 + np.exp(-a)); W = y * (1 - y)
        step = np.linalg.solve((Phi * W[:, None]).T @ Phi, Phi.T @ (t - y)); w += step
        if np.abs(step).max() < tol: break
    a = Phi @ w
    return w, -np.logaddexp(0, -a) @ t - np.logaddexp(0, a) @ (1 - t)


def bma_fit(X, t, max_size=4, keep=2000):
    N, D_ = X.shape; b1 = np.ones((N, 1)); models = []
    for size in range(0, max_size + 1):
        for c_ in itertools.combinations(range(D_), size):
            Phi = np.c_[b1, X[:, c_]] if size else b1
            w, ll = irls_fit(Phi, t.astype(float)); models.append((c_, w, -2 * ll + (size + 1) * np.log(N)))
    bic = np.array([m[2] for m in models]); wts = np.exp(-.5 * (bic - bic.min())); wts /= wts.sum()
    kept = np.argsort(-wts)[:keep]; mass = wts[kept].sum()
    return dict(kept=[(models[i][0], models[i][1], wts[i]) for i in kept], mass=mass)


def bma_pred(X, f):
    p = np.zeros(len(X)); one = np.ones((len(X), 1))
    for c_, w, wt in f["kept"]:
        Phi = np.c_[one, X[:, c_]] if len(c_) else one
        p += wt / (1 + np.exp(-(Phi @ w)))
    return p / f["mass"]


CONT = [i for i, c in enumerate(ENG) if not c.startswith("dow_")]
DUM = [i for i, c in enumerate(ENG) if c.startswith("dow_")]


def make_gam(lam):
    terms = reduce(lambda a, b: a + b, [gam_s(i, n_splines=8, lam=lam) for i in CONT] + [gam_l(i, lam=lam) for i in DUM])
    return LogisticGAM(terms, max_iter=100, tol=1e-4)


def fit_gam(X, y, lam, tries=6):
    """The tuned penalty is at the edge of where the fit converges, so on some stocks it needs a larger one."""
    for _ in range(tries):
        try:
            g = make_gam(lam).fit(X, y); g.lam_used = lam
            return g
        except OptimizationError:
            lam *= 10
    raise OptimizationError("no penalty in the range converged")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def stacking_bases(h):
    rf, ada, gb = h["random_forest"], h["adaboost"], h["gradient_boosting"]
    return [lambda: DecisionTreeClassifier(random_state=SEED, ccp_alpha=h["cart"]["ccp_alpha"]),
            lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **rf),
            lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=SEED, **ada),
            lambda: GradientBoostingClassifier(subsample=.8, random_state=SEED, **gb)]


def stacking_fit(Xr, y, h):
    makers = stacking_bases(h)
    oof = np.full((len(y), len(makers)), np.nan)
    for tr_i, va_i in tscv.split(Xr):
        for j, mk in enumerate(makers):
            oof[va_i, j] = mk().fit(Xr[tr_i], y[tr_i]).predict_proba(Xr[va_i])[:, 1]
    ok = ~np.isnan(oof).any(1)
    sc = StandardScaler().fit(logit(oof[ok])); lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(logit(oof[ok])), y[ok])
    bases = [mk().fit(Xr, y) for mk in makers]; bases[1].n_jobs = 1
    return dict(bases=bases, sc=sc, lr=lr)


def stacking_pred(Xr, f):
    return f["lr"].predict_proba(f["sc"].transform(logit(np.column_stack([b.predict_proba(Xr)[:, 1] for b in f["bases"]]))))[:, 1]


# ---------- direction models ----------
def _proba(m, X): return m.predict_proba(X)[:, 1]


def _hard(p): return (p > .5).astype(int)


DIRECTION = {
    "majority": dict(label="Majority class", group="baseline"),
    "same_as_today": dict(label="Same as today", group="baseline"),
    "lda": dict(label="LDA", group="linear"),
    "gnb": dict(label="Gaussian NB", group="linear"),
    "logistic": dict(label="Logistic regression", group="linear"),
    "bayes_logistic": dict(label="Bayesian logistic (Laplace)", group="linear"),
    "linear_svm": dict(label="Linear SVM", group="kernel"),
    "rbf_svm": dict(label="RBF SVM", group="kernel"),
    "cart": dict(label="CART (pruned)", group="tree"),
    "bagging": dict(label="Bagging", group="ensemble"),
    "random_forest": dict(label="Random Forest", group="ensemble"),
    "adaboost": dict(label="AdaBoost", group="ensemble"),
    "gradient_boosting": dict(label="Gradient Boosting", group="ensemble"),
    "stacking": dict(label="Stacking (logistic meta)", group="ensemble", slow=True),
    "gam": dict(label="GAM (LogisticGAM)", group="additive"),
    "bma": dict(label="BMA (logistic subsets)", group="bayesian"),
}


def fit_direction(key, x, y, h):
    p = h.get(key, {})
    if key == "majority": return dict(cls=int(y.mean() > .5), p=float(y.mean()))
    if key == "same_as_today": return {}
    if key == "lda": return LinearDiscriminantAnalysis(solver="lsqr").fit(x.B, y)
    if key == "gnb": return GaussianNB().fit(x.B, y)
    if key == "logistic": return LogisticRegression(C=p["C"], max_iter=5000).fit(x.B, y)
    if key == "bayes_logistic": return laplace_fit(x.B, y.astype(float), p["alpha"])
    if key == "linear_svm": return LinearSVC(loss="hinge", dual=True, max_iter=100000, random_state=SEED, C=p["C"]).fit(x.B, y)
    if key == "rbf_svm": return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"]).fit(x.B, y)
    if key == "cart": return DecisionTreeClassifier(random_state=SEED, ccp_alpha=p["ccp_alpha"], max_depth=p.get("max_depth")).fit(x.R, y)
    if key == "bagging": return BaggingClassifier(DecisionTreeClassifier(random_state=SEED, min_samples_leaf=p["min_samples_leaf"]), n_estimators=150, max_samples=p["max_samples"], random_state=SEED).fit(x.R, y)
    if key == "random_forest":
        m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **p).fit(x.R, y); m.n_jobs = 1   # one row at a time needs no thread pool
        return m
    if key == "adaboost": return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=SEED, **p).fit(x.R, y)
    if key == "gradient_boosting": return GradientBoostingClassifier(subsample=0.8, random_state=SEED, **p).fit(x.R, y)
    if key == "stacking": return stacking_fit(x.R, y, h)
    if key == "gam": return fit_gam(x.R, y, p["lam"])
    if key == "bma": return bma_fit(x.B, y)
    raise KeyError(key)


def predict_direction(key, m, x):
    """Returns (predicted class, ranking score or None, probability of up or None)."""
    n = len(x.R)
    if key == "majority": return np.full(n, m["cls"]), None, np.full(n, m["p"])
    if key == "same_as_today":
        s = (x.r_lag0 > 0).astype(float); return s.astype(int), s, None
    if key == "lda": return m.predict(x.B), m.decision_function(x.B), _proba(m, x.B)
    if key == "gnb":
        q = _proba(m, x.B); return m.predict(x.B), q, q
    if key == "logistic": return m.predict(x.B), m.decision_function(x.B), _proba(m, x.B)
    if key == "bayes_logistic":
        q = laplace_pred(x.B, m); return _hard(q), q, q
    if key in ("linear_svm", "rbf_svm"): return m.predict(x.B), m.decision_function(x.B), None
    if key in ("cart", "bagging", "random_forest", "adaboost", "gradient_boosting"):
        q = _proba(m, x.R); return _hard(q), q, q
    if key == "stacking":
        q = stacking_pred(x.R, m); return _hard(q), q, q
    if key == "gam":
        q = m.predict_proba(x.R); return _hard(q), q, q
    if key == "bma":
        q = bma_pred(x.B, m); return _hard(q), q, q
    raise KeyError(key)


# ---------- price models ----------
PRICE = {
    "persistence": dict(label="Persistence", group="baseline", target="baseline", features="none"),
    "ols_a": dict(label="OLS, price level", group="linear", target="A", features="A"),
    "ols_b_raw": dict(label="OLS, past returns", group="linear", target="B", features="C"),
    "ols_b_eng": dict(label="OLS, all features", group="linear", target="B", features="B"),
    "huber_a": dict(label="Huber, price level", group="robust", target="A", features="A"),
    "huber_b_eng": dict(label="Huber, all features", group="robust", target="B", features="B"),
    "ridge_a": dict(label="Ridge, price level", group="ridge", target="A", features="A"),
    "ridge_b_raw": dict(label="Ridge, past returns", group="ridge", target="B", features="C"),
    "ridge_b_eng": dict(label="Ridge, all features", group="ridge", target="B", features="B"),
    "blr_a": dict(label="Bayesian LR, price level", group="bayesian", target="A", features="A"),
    "blr_b_eng": dict(label="Bayesian LR, all features", group="bayesian", target="B", features="B"),
}


def fit_price(key, x, tr, h):
    spec = PRICE[key]
    if key == "persistence": return {}
    X = getattr(x, spec["features"])
    y = (tr.close_next if spec["target"] == "A" else tr.r_next).values
    if key.startswith("ols"): return LinearRegression().fit(X, y)
    if key.startswith("huber"): return ScaledY(HuberRegressor(epsilon=h[key]["epsilon"], alpha=1e-4, max_iter=5000)).fit(X, y)
    if key.startswith("ridge"): return Ridge(alpha=h[key]["alpha"]).fit(X, y)
    if key.startswith("blr"): return blr_evidence(X, y)
    raise KeyError(key)


def predict_price(key, m, x, close_t):
    """Returns (forecast, lower 95%, upper 95%), the band only for Bayesian LR."""
    spec = PRICE[key]
    if key == "persistence": return close_t, None, None
    X = getattr(x, spec["features"])
    if key.startswith("blr"):
        mu, var = blr_predict(m, X); sd = np.sqrt(var)
        if spec["target"] == "A": return mu, mu - Z95 * sd, mu + Z95 * sd
        return close_t * np.exp(mu), close_t * np.exp(mu - Z95 * sd), close_t * np.exp(mu + Z95 * sd)
    p = m.predict(X)
    return (p if spec["target"] == "A" else close_t * np.exp(p)), None, None


# ---------- hidden Markov model ----------
def log_emis(x, mu, var):
    return -.5 * (np.log(2 * np.pi * var)[None, :] + (x[:, None] - mu[None, :]) ** 2 / var[None, :])


def hmm_filter(x, pi, A, mu, var):
    """Filtered P(state on day t | returns up to t). Nothing later than t is used."""
    T, K = len(x), len(pi); le = log_emis(x, mu, var); b = np.exp(le - le.max(1, keepdims=True))
    al = np.zeros((T, K)); al[0] = pi * b[0]; al[0] /= al[0].sum()
    for t in range(1, T):
        al[t] = (al[t - 1] @ A) * b[t]; al[t] /= al[t].sum()
    return al


def hmm_forecast(filt, A, mu, var):
    nxt = filt @ A
    p_up = nxt @ stats.norm.sf(0, mu, np.sqrt(var))
    mean = nxt @ mu
    sd = np.sqrt(nxt @ (var + mu ** 2) - mean ** 2)
    return p_up, mean, sd


def rand_init(x, K, rng):
    q = np.quantile(x, rng.uniform(.05, .95, K)); v = x.var() * rng.uniform(.3, 2.0, K)
    A = rng.dirichlet(np.ones(K) * 2, K) * .15 + np.eye(K) * .85
    return np.ones(K) / K, A / A.sum(1, keepdims=True), q, v


def fit_hmm(rets, cutoff, results=None, ticker=None, n_restart=8):
    """RELIANCE at the notebook's split uses the notebook's parameters. Anything else is fitted with K = 3 on its own training returns."""
    xtr = rets[rets.index < pd.Timestamp(cutoff)].values
    if results is not None and ticker == "RELIANCE" and pd.Timestamp(cutoff) == P.TEST_START:
        s = results["u4"]["reliance"]
        return dict(pi=np.array(s["pi"]), A=np.array(s["A"]), mu=np.array(s["mean"]), var=np.array(s["sd"]) ** 2, names=s["names"])
    rng = np.random.default_rng(P.SEED); best = None
    for _ in range(n_restart):
        ini = rand_init(xtr, 3, rng)
        h = GaussianHMM(3, "diag", n_iter=500, tol=1e-4, covars_prior=0, covars_weight=0, init_params="", params="stmc", implementation="scaling")
        h.startprob_, h.transmat_, h.means_, h.covars_ = ini[0], ini[1], ini[2][:, None], ini[3][:, None]
        h.fit(xtr[:, None]); ll = h.score(xtr[:, None])
        if best is None or ll > best[0]: best = (ll, h)
    h = best[1]; o = np.argsort(h.means_[:, 0])
    return dict(pi=h.startprob_[o], A=h.transmat_[o][:, o], mu=h.means_[o, 0], var=h.covars_[o, 0, 0], names=["bear", "sideways", "bull"])


def stationary(A):
    w, v = np.linalg.eig(A.T); st = np.real(v[:, np.argmin(np.abs(w - 1))]); return st / st.sum()


# ---------- Kalman filter ----------
KALMAN = {"kalman_ll": dict(label="Kalman, local level", group="kalman", kind="ll"), "kalman_llt": dict(label="Kalman, local linear trend", group="kalman", kind="llt")}


def kf_matrices(kind, theta):
    v = np.exp(theta)
    if kind == "ll": return np.array([[1.0]]), np.array([[1.0]]), np.array([[v[1]]]), v[0]
    return np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[1.0, 0.0]]), np.diag([v[1], v[2]]), v[0]


def kalman(y, TT, Z, Q, R):
    """One-step forecasts: out[t] is the forecast of y[t + 1] from data up to t."""
    n, k = len(y), TT.shape[0]; a = np.zeros(k); P = np.eye(k) * 1e6; ll = 0.0; Zr = Z[0]; out = np.zeros(n)
    for t in range(n):
        v = y[t] - Zr @ a; PZ = P @ Zr; F = Zr @ PZ + R; K_ = PZ / F
        a = a + K_ * v; P = P - np.outer(K_, PZ)
        if t >= k: ll += -.5 * (np.log(2 * np.pi) + np.log(F) + v * v / F)
        a = TT @ a; P = TT @ P @ TT.T + Q
        out[t] = Zr @ a
    return out, ll


def fit_kalman(kind, y_tr, results=None, ticker=None, cutoff=None):
    if results is not None and ticker == "RELIANCE" and pd.Timestamp(cutoff) == P.TEST_START:
        s = results["u4"]["kalman"][kind]
        Q = np.diag(s["Q"]) if kind == "llt" else np.array([s["Q"]])
        TT, Z, _, _ = kf_matrices(kind, np.log([1.0, 1.0] if kind == "ll" else [1.0, 1.0, 1.0]))
        return dict(kind=kind, TT=TT, Z=Z, Q=Q, R=s["R"])
    starts = [np.log([1.0, 3.0]), np.log([.01, 4.0])] if kind == "ll" else [np.log([1.0, 3.0, 1e-4]), np.log([.01, 4.0, 1e-3])]
    def nll(th):
        TT, Z, Q, R = kf_matrices(kind, th); return -kalman(y_tr, TT, Z, Q, R)[1]
    best = None
    for st in starts:
        o_ = optimize.minimize(nll, st, method="L-BFGS-B", bounds=[(-25, 8)] * len(st), options=dict(maxiter=300, ftol=1e-11))
        if best is None or o_.fun < best.fun: best = o_
    TT, Z, Q, R = kf_matrices(kind, best.x)
    return dict(kind=kind, TT=TT, Z=Z, Q=Q, R=R)


def kalman_price_forecast(params, prices):
    """Forecast of the next price, for each day in the series."""
    out, _ = kalman(100 * np.log(prices), params["TT"], params["Z"], params["Q"], params["R"])
    return np.exp(out / 100)


# ---------- metrics ----------
def binom_p(k, n, rate):
    return float(stats.binomtest(int(k), int(n), rate, alternative="greater").pvalue)


def dm_test(e1, e0, lags=5):
    d = e1 ** 2 - e0 ** 2; n = len(d); dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - l / (lags + 1)) * (dc[l:] @ dc[:-l]) / n for l in range(1, lags + 1))
    if v <= 0: return None
    return float(2 * stats.norm.sf(abs(d.mean() / np.sqrt(v / n))))


def score_direction(yhat, score, y):
    return dict(accuracy=float(np.mean(yhat == y)), n_correct=int((yhat == y).sum()), n_test=len(y), auc=None if score is None or len(np.unique(score)) == 1 else float(roc_auc_score(y, score)), share_up=float(np.mean(yhat)))


# ---------- a complete fitted set for one stock and one training cutoff ----------
def cache_path(ticker, cutoff):
    return P.CACHE_DIR / f"{ticker.replace('&', '')}_{pd.Timestamp(cutoff).date()}.joblib"


def fit_bundle(store, ticker, cutoff, progress=None):
    D = store.frame(ticker); tr, _ = split(D, cutoff)
    h = tuned(store.results)
    des = Design(tr); x = des.x(tr); y = tr.y_dir.values.astype(int)
    b = dict(ticker=ticker, cutoff=str(pd.Timestamp(cutoff).date()), design=des, direction={}, price={}, n_train=len(tr), train_end=str(tr.index.max().date()), settings=h)
    for key in DIRECTION:
        if progress: progress(key)
        try:
            b["direction"][key] = fit_direction(key, x, y, h)
        except (OptimizationError, np.linalg.LinAlgError, ValueError) as e:
            # one model that cannot be fitted on one stock should not take the others down with it
            b["direction"][key] = None; b.setdefault("failed", {})[key] = type(e).__name__
    for key in PRICE:
        if progress: progress(key)
        b["price"][key] = fit_price(key, x, tr, h)
    s = store.series(ticker)
    prices = s.Adj; y_tr = 100 * np.log(prices[prices.index < pd.Timestamp(cutoff)].values)
    for key, spec in KALMAN.items():
        if progress: progress(key)
        b["price"][key] = fit_kalman(spec["kind"], y_tr, store.results, ticker, cutoff)
    if progress: progress("hmm")
    b["hmm"] = fit_hmm(s.r.dropna() * 100, cutoff, store.results, ticker)
    q = tr[ENG]
    b["ranges"] = {c: dict(lo=float(q[c].quantile(.01)), hi=float(q[c].quantile(.99)), mean=float(q[c].mean())) for c in ENG}
    return b


def load_bundle(store, ticker, cutoff=P.TEST_START, fit_if_missing=True, progress=None):
    f = cache_path(ticker, cutoff)
    if f.exists():
        b = joblib.load(f)
        # a rerun of the notebook that changes the tuned settings makes the saved fits out of date
        if b.get("settings") == tuned(store.results): return b
    if not fit_if_missing: return None
    b = fit_bundle(store, ticker, cutoff, progress)
    save_bundle(b)
    return b


def save_bundle(b):
    P.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = cache_path(b["ticker"], b["cutoff"]); tmp = pathlib.Path(str(f) + ".tmp")
    joblib.dump(b, tmp, compress=3); tmp.replace(f)
