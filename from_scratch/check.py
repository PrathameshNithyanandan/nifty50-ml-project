"""Compares the no-library results with the notebook, which uses scikit-learn, and writes from_scratch/COMPARISON.md.

    python3 from_scratch/check.py
"""
import ast
import json
import os
import sys

from dataset import RESULTS, ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
ALLOWED = {"array", "ast", "csv", "datetime", "json", "math", "multiprocessing", "operator", "os", "random", "sys", "time"}
OWN = {"dataset", "matrix"}


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def imports_used():
    found = {}
    for name in sorted(os.listdir(HERE)):
        if name.endswith(".py"):
            with open(os.path.join(HERE, name)) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
                for m in mods:
                    found.setdefault(m.split(".")[0], set()).add(name)
    return found


def main():
    R, U2, U3 = load("results.json"), load("from_scratch_unit2.json"), load("from_scratch_unit3.json")
    rows, lines = [], []

    def add(group, label, nb, own, tol):
        """tol is the largest difference that still counts as the same result; None compares exactly."""
        if tol is None:
            same = nb == own
            diff = "" if same else "differs"
        else:
            d = abs(nb - own)
            same = d <= tol
            diff = f"{d:.1e}"
        rows.append(dict(group=group, label=label, nb=nb, own=own, diff=diff, same=same))

    # Unit 2, price
    reg_nb = {r["key"]: r for r in R["u2_reg"]["table"]}
    for r in U2["reg"]:
        n = reg_nb[r["key"]]
        tag = f"{r['model']} [{r['target']}|{r['features']}]"
        for m in ("rmse", "mae", "mape"):
            add("Unit 2: next-day close", f"{tag} {m.upper()}", n[m], r[m], 1e-3)
    blr = R["u2_reg"]["blr"]["b"]
    add("Unit 2: next-day close", "Bayesian LR α", blr["alpha"], U2["blr"]["b"]["alpha"], blr["alpha"] * 1e-6)
    add("Unit 2: next-day close", "Bayesian LR β", blr["beta"], U2["blr"]["b"]["beta"], blr["beta"] * 1e-6)
    add("Unit 2: next-day close", "Bayesian LR effective parameters γ", blr["gamma"], U2["blr"]["b"]["gamma"], 1e-6)
    add("Unit 2: next-day close", "OLS noise precision β", R["u2_reg"]["ols"]["beta_closed"], U2["ols"]["beta_closed"], 1e-3)
    for k, v in R["u2_reg"]["huber"].items():
        add("Unit 2: next-day close", f"{k} tuned ε", v["epsilon"], U2["huber"][k]["epsilon"], None)
        add("Unit 2: next-day close", f"{k} outlier days", v["outliers"], U2["huber"][k]["outliers"], 2)
    for k, v in R["u2_reg"]["ridge"].items():
        add("Unit 2: next-day close", f"{k} tuned α", v["best"], U2["ridge"][k]["best"], v["best"] * 1e-3)

    # Unit 2, direction
    clf_nb = {r["key"]: r for r in R["u2_clf"]["table"]}
    for r in U2["clf"]:
        n = clf_nb[r["key"]]
        add("Unit 2: up or down", f"{r['model']} accuracy", n["accuracy"], r["accuracy"], 2 / 822)
        add("Unit 2: up or down", f"{r['model']} F1", n["f1"], r["f1"], 0.01)
        if r["auc"] is not None:
            add("Unit 2: up or down", f"{r['model']} AUC", n["auc"], r["auc"], 1e-3)
        if r["log_loss"] is not None:
            add("Unit 2: up or down", f"{r['model']} log-loss", n["log_loss"], r["log_loss"], 1e-3)
    c = R["u2_clf"]
    add("Unit 2: up or down", "Logistic tuned C", c["logistic"]["C"], U2["tuned"]["logistic_C"], c["logistic"]["C"] * 1e-3)
    add("Unit 2: up or down", "Bayesian logistic tuned α", c["laplace"]["alpha"], U2["laplace"]["alpha"], c["laplace"]["alpha"] * 1e-3)
    add("Unit 2: up or down", "Linear SVM tuned C", c["linear_svm"]["C"], U2["tuned"]["linear_svm_C"], c["linear_svm"]["C"] * 1e-3)
    add("Unit 2: up or down", "RBF SVM C", c["rbf"]["C"], U2["tuned"]["rbf_C"], None)
    add("Unit 2: up or down", "RBF SVM γ", c["rbf"]["gamma"], U2["tuned"]["rbf_gamma"], None)
    add("Unit 2: up or down", "RBF SVM support vectors", c["rbf"]["n_sv"], U2["tuned"]["n_sv"], 5)
    add("Unit 2: up or down", "Models significantly above the majority rate", c["n_beat_sig"], sum(r["kind"] == "model" and r["p_vs_majority"] < 0.05 for r in U2["clf"]), None)

    # Unit 3
    u3, g = R["u3"], R["u3"]["gmm"]
    km = {(r["space"], r["k"]): r for r in U3["kmeans"]}
    for r in u3["kmeans"]:
        o = km[(r["space"], r["k"])]
        tag = f"K-means {r['space']} k={r['k']}"
        add("Unit 3: K-means", f"{tag} inertia", r["inertia"], o["inertia"], 1e-4)
        add("Unit 3: K-means", f"{tag} silhouette", r["silhouette"], o["silhouette"], 1e-6)
        add("Unit 3: K-means", f"{tag} Davies–Bouldin", r["db"], o["db"], 1e-6)
    add("Unit 3: K-means", "Chosen k, 6 features", u3["k_chosen"], U3["k_chosen"], None)
    add("Unit 3: K-means", "Chosen k, 2 features", u3["k_chosen_2d"], U3["k_chosen_2d"], None)
    ug = U3["gmm"]
    for key in ("bic_cov", "bic_k", "aic_cov", "aic_k", "bic2_cov", "bic2_k", "aic2_cov", "aic2_k"):
        add("Unit 3: GMM selection", key.replace("_", " ").upper(), g[key], ug[key], None)
    add("Unit 3: GMM selection", "BIC of the chosen model", g["bic"], ug["bic"], 1e-3)
    add("Unit 3: GMM selection", "EM log-likelihood never falls", g["em_monotone"], ug["em_monotone"], None)
    hier = {(r["distance"], r["linkage"]): r for r in U3["hier"]}
    for r in u3["hier"]:
        add("Unit 3: hierarchical", f"{r['distance']} {r['linkage']} cophenetic", r["cophenetic"], hier[(r["distance"], r["linkage"])]["cophenetic"], 1e-6)
    met = {(r["method"], r["space"], r["k"]): r for r in U3["metrics"]}
    for r in u3["metrics"]:
        if r["method"] == "hier":
            o = met[("hier", r["space"], r["k"])]
            add("Unit 3: hierarchical", f"Ward {r['space']} k={r['k']} silhouette", r["silhouette"], o["silhouette"], 1e-6)
            add("Unit 3: hierarchical", f"Ward {r['space']} k={r['k']} ARI with industry", r["ari_industry"], o["ari_industry"], 1e-6)
    for key in ("km_6d", "hier_6d", "km_2d", "corr_ward", "corr_average"):
        add("Unit 3: hierarchical", f"ARI with industry, {key}", u3["ari"][key], U3["ari"][key], 1e-6)
    p, q = R["u3_pca"], U3["pca"]
    for key in ("days", "stocks", "n80", "n_kaiser", "n_parallel", "n_broken", "start", "end"):
        add("Unit 3: PCA and factor analysis", key, p[key], q[key], None)
    for key in ("pc1", "pc2", "pc3", "ev1", "pc1_corr"):
        add("Unit 3: PCA and factor analysis", key, p[key], q[key], 1e-6)
    for key, tol in (("ll_fa", 0.5), ("ll_ppca", 1e-2), ("sigma2_ppca", 1e-6), ("psi_min", 1e-3), ("psi_max", 1e-3), ("psi_mean", 1e-3)):
        add("Unit 3: PCA and factor analysis", key, p["fa"][key], U3["fa"][key], tol)

    # differences found in practice, kept visible and not counted as agreement
    open_items = [r for r in rows if not r["same"]]
    used = imports_used()
    outside = sorted(set(used) - ALLOWED - OWN)

    fmt = lambda v: f"{v:.6g}" if isinstance(v, float) else str(v)
    lines.append("# No-library results compared with the notebook\n")
    lines.append("Written by `from_scratch/check.py`. The notebook uses scikit-learn and SciPy; the scripts here use neither, nor NumPy or pandas.\n")
    lines.append(f"**{len(rows) - len(open_items)} of {len(rows)} quantities agree.** Modules imported: " + ", ".join(f"`{m}`" for m in sorted(set(used) - OWN)) + ". All come with Python.\n")
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    lines.append("| Part | Quantities | Agree |\n|---|---:|---:|")
    for g_ in groups:
        part = [r for r in rows if r["group"] == g_]
        lines.append(f"| {g_} | {len(part)} | {sum(r['same'] for r in part)} |")
    lines.append("")
    if open_items:
        lines.append("## Where they differ\n")
        lines.append("K-means and EM start from random points and can stop at different local optima, so runs with different random numbers can end in different clusterings. "
                     "This shows up when 49 stocks are split into many clusters, or when AIC compares models with many parameters. A lower inertia or AIC means that run found the better optimum, and either run can be the one that does. "
                     "The closed-form and convex models have no such randomness.\n")
        lines.append("| Quantity | Notebook | No library | Difference |\n|---|---:|---:|---:|")
        for r in open_items:
            lines.append(f"| {r['label']} | {fmt(r['nb'])} | {fmt(r['own'])} | {r['diff']} |")
        lines.append("")
    for g_ in groups:
        lines.append(f"## {g_}\n\n| Quantity | Notebook | No library | Difference |\n|---|---:|---:|---:|")
        for r in rows:
            if r["group"] == g_:
                lines.append(f"| {r['label']} | {fmt(r['nb'])} | {fmt(r['own'])} | {r['diff']} |")
        lines.append("")
    with open(os.path.join(HERE, "COMPARISON.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"{len(rows) - len(open_items)} of {len(rows)} quantities agree.")
    for g_ in groups:
        part = [r for r in rows if r["group"] == g_]
        print(f"  {g_:<34}{sum(r['same'] for r in part):>4} of {len(part)}")
    for r in open_items:
        print(f"  differs: {r['label']}: notebook {fmt(r['nb'])}, no library {fmt(r['own'])}")
    print("Modules imported:", ", ".join(sorted(set(used) - OWN)))
    if outside:
        print("NOT ALLOWED:", outside)
        sys.exit(1)
    print("Wrote from_scratch/COMPARISON.md")


if __name__ == "__main__":
    main()
