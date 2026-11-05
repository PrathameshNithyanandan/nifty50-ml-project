import { useState } from "react";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { FEATURE_HELP } from "../lib/features";
import { usePalette } from "../lib/colors";
import { useNarrow } from "../lib/useNarrow";
import { Card, Controls, Kpi, KpiRow, Note, PageHead, Segmented, T, Takeaways } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Dot, Table } from "../components/Table";

function Leaderboard() {
  const R = useR(); const U = R.u5; const pal = usePalette();
  return (
    <Card id="leaderboard" title="Model leaderboard against baselines" q="Do trees and ensembles find a direction signal that linear models miss?" m={<>Same RELIANCE split, features and target as Unit 2. Report accuracy, <T k="f1">F1</T>, <T k="auc">AUC</T>, <T k="logloss">log-loss</T> and a one-sided <T k="binom">binomial test</T> against the majority rate.</>} tags={["Practice 2: ensemble classifiers", "Ensemble learning"]}
      meaning={<><strong>No model beats the baseline.</strong> {U.n_above} of {U.n_models} exceed the majority rate of {f.n(U.majority_rate, 4)}, {U.n_sig} are significant, and {U.n_auc_outside} have an AUC outside the chance band of {f.n(U.auc_low, 3)} to {f.n(U.auc_high, 3)}. The most accurate, {U.best_acc_label}, only matches the baseline because it predicts up almost every day. Only {U.n_ll_better} of {U.n_models} improves on a constant probability by log-loss, and only slightly.</>}>
      <Table caption="Unit 5 leaderboard" initialSort={{ key: "auc", dir: -1 }} highlight={(r: any) => r.group === "baseline"} cols={[
        { key: "model", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.model}</> }, { key: "accuracy", label: "Accuracy", render: (r: any) => f.n(r.accuracy, 4) }, { key: "f1", label: "F1", render: (r: any) => f.n(r.f1, 3) }, { key: "auc", label: "AUC", render: (r: any) => f.n(r.auc, 3) }, { key: "cv_auc", label: "CV AUC", render: (r: any) => f.n(r.cv_auc, 3) }, { key: "log_loss", label: "Log-loss", render: (r: any) => f.n(r.log_loss, 4) },
        { key: "p_vs_majority", label: "p vs majority", render: (r: any) => (r.key === "majority" ? "—" : f.p(r.p_vs_majority)) }, { key: "share_up", label: "Predicts up", render: (r: any) => f.pct(r.share_up, 1) },
      ]} rows={U.leaderboard} />
      <Note>Highlighted rows are the baselines. CV AUC is the time-ordered tuning score on training data. It is left blank for stacking because its figure ({f.n(U.stacking.oof_auc, 3)}) is in-sample for the meta-learner, and for BMA, which has no tuning. Test AUCs fall below CV AUCs, as expected when a best setting is selected from noisy scores.</Note>
    </Card>
  );
}

function TreeSvg({ nodes }: { nodes: any[] }) {
  const pal = usePalette();
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const W = 210, H = 86, GX = 24, GY = 54;
  const pos = new Map<number, { x: number; y: number }>();
  let leaf = 0;
  const place = (id: number, depth: number): number => {
    const n = byId.get(id);
    if (n.left < 0) { const x = leaf++ * (W + GX); pos.set(id, { x, y: depth * (H + GY) }); return x; }
    const a = place(n.left, depth + 1), b = place(n.right, depth + 1); const x = (a + b) / 2;
    pos.set(id, { x, y: depth * (H + GY) }); return x;
  };
  place(0, 0);
  const maxX = Math.max(...Array.from(pos.values()).map((p) => p.x)) + W, maxY = Math.max(...Array.from(pos.values()).map((p) => p.y)) + H;
  return (
    <svg viewBox={`0 0 ${maxX} ${maxY}`} className="w-full max-w-3xl mx-auto block" role="img" aria-label="Pruned decision tree">
      {nodes.filter((n) => n.left >= 0).flatMap((n) => [n.left, n.right].map((c: number, i: number) => {
        const a = pos.get(n.id)!, b = pos.get(c)!;
        const mx = (a.x + b.x + W) / 2, my = (a.y + H + b.y) / 2;
        return <g key={`${n.id}-${c}`}>
          <line x1={a.x + W / 2} y1={a.y + H} x2={b.x + W / 2} y2={b.y} stroke={pal.axis} strokeWidth={1.5} />
          <text x={mx} y={my} textAnchor="middle" fontSize={13} fill={pal.ink2} stroke={pal.surface} strokeWidth={4} paintOrder="stroke">{i === 0 ? "yes" : "no"}</text>
        </g>;
      }))}
      {nodes.map((n) => {
        const p = pos.get(n.id)!; const isLeaf = n.left < 0; const up = n.p_up > 0.5;
        return <g key={n.id} transform={`translate(${p.x},${p.y})`}>
          <rect width={W} height={H} rx={6} fill={pal.surface} stroke={isLeaf ? (up ? pal.s.green : pal.s.red) : pal.axis} strokeWidth={isLeaf ? 2 : 1} />
          <text x={W / 2} y={24} textAnchor="middle" fontSize={15} fontWeight={600} fill={pal.ink}>{isLeaf ? (up ? "Predict up" : "Predict not up") : `${n.feature} ≤ ${n.threshold}`}</text>
          <text x={W / 2} y={46} textAnchor="middle" fontSize={13} fill={pal.ink2}>{f.i(n.samples)} days</text>
          <text x={W / 2} y={66} textAnchor="middle" fontSize={13} fill={pal.ink2}>P(up) = {f.n(n.p_up, 3)}, Gini {f.n(n.gini, 3)}</text>
        </g>;
      })}
    </svg>
  );
}

/** The same tree as nested text, so it stays readable on a phone. */
function TreeList({ nodes, id = 0, branch }: { nodes: any[]; id?: number; branch?: string }) {
  const pal = usePalette();
  const n = nodes.find((x) => x.id === id);
  const leaf = n.left < 0; const up = n.p_up > 0.5;
  return (
    <div className="text-sm">
      <div className="rounded border px-2.5 py-1.5 bg-surface" style={{ borderColor: leaf ? (up ? pal.s.green : pal.s.red) : pal.axis }}>
        {branch && <span className="text-ink2">{branch}: </span>}
        <strong>{leaf ? (up ? "Predict up" : "Predict not up") : `${n.feature} ≤ ${n.threshold}`}</strong>
        <div className="text-xs text-ink2 tnum">{f.i(n.samples)} days, P(up) = {f.n(n.p_up, 3)}, Gini {f.n(n.gini, 3)}</div>
      </div>
      {!leaf && <div className="ml-3 pl-3 border-l-2 border-line mt-2 grid gap-2">
        <TreeList nodes={nodes} id={n.left} branch="Yes" />
        <TreeList nodes={nodes} id={n.right} branch="No" />
      </div>}
    </div>
  );
}

function Cart() {
  const R = useR(); const C = R.u5.cart; const c = useChart<any>("u5_tree"); const pal = usePalette(); const narrow = useNarrow();
  const row = R.u5.leaderboard.find((r: any) => r.key === "cart");
  const features: string[] = c ? Array.from(new Set(c.nodes.filter((n: any) => n.feature).map((n: any) => n.feature))) : [];
  return (
    <Card id="cart" title="Pruned CART" q="What rule does one readable tree find?" m={<><T k="cart" /> grows by reducing <T k="gini">Gini impurity</T>. <T k="pruning">Cost-complexity pruning</T> minimises R(T) + α|T|, with α chosen on time-ordered folds.</>} tags={["Practice 1: CART for categorization", "CART"]}
      meaning={<>The unpruned tree has {C.full_leaves} leaves and memorises training data (accuracy {f.n(C.full_train_acc, 3)}), yet scores {f.n(C.full_test_acc, 3)} on test. Pruning leaves {C.n_leaves} leaves, but its test accuracy of {f.n(C.test_acc, 3)} is below the majority rate (p = {f.p(row.p_vs_majority)}), and its CV AUC of {f.n(C.cv_auc, 3)} is about what a coin flip gives. <strong>A single tree finds no usable rule.</strong> It is shown because it can be read, not because it predicts well.</>}>
      <KpiRow>
        <Kpi label="Leaves: unpruned → pruned" value={`${C.full_leaves} → ${C.n_leaves}`} />
        <Kpi label="Train accuracy" value={f.n(C.train_acc, 3)} note={`unpruned ${f.n(C.full_train_acc, 3)}`} />
        <Kpi label="Test accuracy" value={f.n(C.test_acc, 3)} note={`unpruned ${f.n(C.full_test_acc, 3)}`} />
        <Kpi label="α from CV" value={f.n(C.alpha, 5)} note={`${C.n_grid} values tried`} />
      </KpiRow>
      {c ? (narrow ? <TreeList nodes={c.nodes} /> : <TreeSvg nodes={c.nodes} />) : <Skeleton />}
      <Note>{features.map((x) => `${x}: ${FEATURE_HELP[x] ?? x}`).join(". ")}. Thresholds are in the feature's own units.</Note>
      {c && <div className="mt-3"><Chart ariaLabel="CV AUC and number of leaves across the pruning path" height={300} legend={false} x="leaves in the pruned tree (log scale)" y="CV AUC" layout={{ xaxis: { type: "log" }, shapes: [{ type: "line", x0: C.n_leaves, x1: C.n_leaves, yref: "paper", y0: 0, y1: 1, line: { color: pal.ink2, dash: "dash", width: 1 } }] }} data={[{ x: c.prune.leaves, y: c.prune.cv_auc, type: "scatter", mode: "markers", marker: { color: pal.model("cart"), size: 8 }, customdata: c.prune.alpha, hovertemplate: "%{x} leaves<br>CV AUC %{y:.4f}<br>α = %{customdata:.5f}<extra></extra>" }]} note="Dashed line: the chosen tree. No tree size gets far from one half, so the choice among them is noisy." /></div>}
    </Card>
  );
}

function Ensembles() {
  const R = useR(); const U = R.u5; const Tn = U.tuning; const pal = usePalette();
  const rows = [
    { key: "bagging", name: "Bagging", t: Tn.bagging, how: <><T k="bagging" />: averages deep trees on bootstrap samples, which lowers variance.</> },
    { key: "random_forest", name: "Random Forest", t: Tn.random_forest, how: <><T k="rf" />: also samples features at each split, which decorrelates trees.</> },
    { key: "adaboost", name: "AdaBoost", t: Tn.adaboost, how: <><T k="adaboost" />: fits stumps in sequence, reweighting mistakes.</> },
    { key: "gradient_boosting", name: "Gradient Boosting", t: Tn.gradient_boosting, how: <><T k="gb" />: fits each tree to the gradient of the log-loss.</> },
  ];
  const paramText = (b: any) => Object.entries(b).map(([k, v]) => `${k.replace("estimator__", "")} = ${f.g(typeof v === "number" ? v : Number.isNaN(Number(v)) ? null : v) === "—" ? v : f.g(v)}`).join(", ");
  const edges = rows.filter((r) => Object.keys(r.t.at_edge).length > 0);
  return (
    <Card id="ensembles" title="Ensembles and tuning" q="Can combining many weak or noisy trees find a signal?" m={<>Tune each model on <T k="tscv" />. <T k="stacking" /> trains a logistic meta-learner on out-of-fold predictions from the same expanding-window folds, since ordinary K-fold stacking would leak future data.</>} tags={["Bagging", "Random Forest", "Boosting", "Stacking"]}
      meaning={<>CV keeps favouring weak, heavily restricted learners, because extra flexibility mostly fits noise, and CV AUC moves very little between settings. Averaging reduces variance, but there is little signal for it to reveal. Bagging's max_samples at its upper limit is a natural boundary. The other edges mean CV would go further towards weak models if allowed, which is itself the finding.</>}>
      <Table sortable={false} caption="Ensemble tuning" cols={[
        { key: "name", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.name}</> }, { key: "how", label: "How it works", align: "left", wrap: true, render: (r: any) => <span className="block min-w-48">{r.how}</span> }, { key: "p", label: "Chosen settings", align: "left", wrap: true, render: (r: any) => <span className="block min-w-48">{paramText(r.t.best)}</span> }, { key: "cv", label: "CV AUC", render: (r: any) => f.n(r.t.cv_auc, 3) }, { key: "e", label: "At grid edge", align: "left", wrap: true, render: (r: any) => Object.keys(r.t.at_edge).map((k) => k.replace("estimator__", "")).join(", ") || "none" },
      ]} rows={rows} />
      <Note>{edges.length} of {rows.length} models have a setting at a grid edge. Stacking uses {f.i(U.stacking.n_oof)} of {f.i(U.stacking.n_train)} training rows, since rows before the first validation fold have no out-of-fold prediction. Its meta-learner weights on standardised base logits: {Object.entries<number>(U.stacking.coef).map(([k, v]) => `${k.replace("_", " ")} ${f.s(v, 3)}`).join(", ")}.</Note>
    </Card>
  );
}

function Basis() {
  const R = useR(); const S = R.u5.staged; const c = useChart<any>("u5_staged"); const pal = usePalette();
  return (
    <Card id="basis" title="Adaptive basis functions" q="What do trees, boosting and GAMs have in common?" m={<>All fit f(x) = w₀ + Σ wₘφₘ(x; vₘ). In an <T k="abf">adaptive basis function model</T>, the basis itself is learned. Follow an illustrative boosting run (learning rate 0.05, depth 2) as it adds trees.</>} tags={["Adaptive basis function models", "Boosting"]}
      meaning={<>Each tree added is a new learned basis function. Train log-loss keeps falling, from {f.n(S.train_first, 4)} to {f.n(S.train_last, 4)}, but test log-loss rises from {f.n(S.test_first, 4)} to {f.n(S.test_last, 4)}, worse than the constant prediction. More flexible basis functions fit noise when there is no stable signal, which is why CV chose such a low learning rate. These settings are for illustration, not the tuned model.</>}>
      {c ? <Chart ariaLabel="Train and test log-loss as boosting rounds are added" height={340} x="boosting rounds (log scale)" y="log-loss" hover="x" layout={{ xaxis: { type: "log" } }} data={[
        { x: c.rounds, y: c.train_ll, type: "scatter", mode: "lines", name: "Train", line: { color: pal.s.blue, width: 2 }, hovertemplate: "Train %{y:.4f}<extra></extra>" },
        { x: c.rounds, y: c.test_ll, type: "scatter", mode: "lines", name: "Test", line: { color: pal.s.orange, width: 2 }, hovertemplate: "Test %{y:.4f}<extra></extra>" },
        { x: [c.rounds[0], c.rounds[c.rounds.length - 1]], y: [c.constant_ll, c.constant_ll], type: "scatter", mode: "lines", name: "Constant probability (test)", line: { color: pal.model("majority"), width: 1.5, dash: "dash" }, hovertemplate: "Constant %{y:.4f}<extra></extra>" },
      ]} /> : <Skeleton />}
    </Card>
  );
}

function Importance() {
  const R = useR(); const I = R.u5.importance; const c = useChart<any>("u5_importance"); const pal = usePalette();
  const [m, setM] = useState<"rf" | "gb">("rf");
  const color = pal.model(m === "rf" ? "random_forest" : "gradient_boosting");
  const n = c?.features.length ?? 0; const idx = c ? [...Array(n).keys()].sort((a, b) => c[`${m}_impurity`][a] - c[`${m}_impurity`][b]) : [];
  return (
    <Card id="importance" title="Feature importance: impurity vs permutation" q="Which features matter, and do they matter out of sample?" m={<><T k="impurity" /> comes from the training fit. <T k="permutation" /> is the fall in test AUC when a feature is shuffled.</>} tags={["Random Forest", "Boosting"]}
      meaning={<>The two tell different stories (rank correlation {f.n(I.spearman, 2)}). Impurity ranks {I.top_impurity} first, but shuffling it changes test AUC by only {f.n(Math.abs(I.top_impurity_perm), 4)}, which is within shuffle noise. {I.clear_rf.length === 0 ? "No feature" : `${I.clear_rf.length} feature(s)`} clearly exceed 0 by two standard deviations. Impurity shows what a model used in training, and that does not mean it helped on new data.</>}>
      <Controls><Segmented label="Model" value={m} onChange={setM} options={[{ value: "rf", label: "Random Forest" }, { value: "gb", label: "Gradient Boosting" }]} /></Controls>
      {c ? <div className="grid gap-4 lg:grid-cols-2 min-w-0">
        <Chart ariaLabel="Impurity importance" height={480} heightNarrow={480} legend={false} x="impurity importance (train)" layout={{ yaxis: { dtick: 1, tickfont: { size: 11 } }, margin: { l: 90 } }} data={[{ type: "bar", orientation: "h", y: idx.map((i) => c.features[i]), x: idx.map((i) => c[`${m}_impurity`][i]), marker: { color }, text: idx.map((i) => FEATURE_HELP[c.features[i]] ?? ""), hovertemplate: "%{y}: %{text}<br>importance %{x:.4f}<extra></extra>", textposition: "none" }]} />
        <Chart ariaLabel="Permutation importance with error bars" height={480} heightNarrow={480} legend={false} x="fall in test AUC when shuffled" layout={{ yaxis: { dtick: 1, tickfont: { size: 11 } }, margin: { l: 90 } }} data={[{ type: "scatter", mode: "markers", y: idx.map((i) => c.features[i]), x: idx.map((i) => c[`${m}_perm`][i]), marker: { color, size: 8 }, error_x: { type: "data", array: idx.map((i) => 2 * c[`${m}_perm_sd`][i]), color: pal.muted, thickness: 1.5, width: 3 }, hovertemplate: "%{y}<br>fall in AUC %{x:.4f}<extra></extra>" }]} note="Bars show ±2 standard deviations over the shuffles. Both panels use the same feature order." />
      </div> : <Skeleton h={480} />}
    </Card>
  );
}

function Gam() {
  const R = useR(); const G = R.u5.gam; const c = useChart<any>("u5_gam"); const pal = usePalette(); const narrow = useNarrow();
  const color = pal.model("gam");
  const [fName, setFName] = useState<string | null>(null);
  const shown = c ? (fName ? c.curves.filter((x: any) => x.feature === fName) : c.curves.slice(0, narrow ? 3 : 6)) : [];
  const fillc = pal.mode === "dark" ? "rgba(213,81,129,0.28)" : "rgba(232,123,164,0.3)";
  return (
    <Card id="gam" title="Generalized additive model" q="Is any single feature's effect visibly non-flat?" m={<>A <T k="gam">GAM</T> (LogisticGAM) with one penalised spline per feature. λ is chosen on time-ordered folds. <T k="partial">Partial dependence</T> shows each feature's effect on the log-odds, centred to mean 0.</>} tags={["Generalized additive models"]}
      meaning={<>For {G.n_flat_curves} of {G.n_curves} features, a flat line stays inside the 95% band, so those effects cannot be told apart from no effect. The chosen λ = {f.g(G.lam)} sits at the edge of the range where the fit converges ({G.diverged.length > 0 ? `it diverges at ${f.g(G.diverged[0])}` : "no tested value diverged"}), so these curves should be read as rough shapes. Explained deviance is only {f.n(G.explained_deviance, 4)}, even in training.</>}>
      <Controls>
        <label className="text-sm text-ink2 flex items-center gap-2">Feature
          <select value={fName ?? ""} onChange={(e) => setFName(e.target.value || null)} className="rounded border border-line bg-surface text-ink px-2 py-1 text-sm max-w-[240px]">
            <option value="">Largest effects</option>
            {c?.curves.map((x: any) => <option key={x.feature} value={x.feature}>{x.feature}</option>)}
          </select>
        </label>
      </Controls>
      {c ? <div className={`grid gap-4 min-w-0 ${shown.length > 1 ? "sm:grid-cols-2 lg:grid-cols-3" : ""}`}>
        {shown.map((d: any) => (
          <Chart key={d.feature} ariaLabel={`Partial dependence for ${d.feature}`} height={shown.length > 1 ? 260 : 360} heightNarrow={260} legend={false} x={d.feature} y="centred log-odds" hover="x" layout={{ margin: { t: 8 }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: pal.muted, width: 1, dash: "dot" } }] }} data={[
            { x: d.x, y: d.hi, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
            { x: d.x, y: d.lo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: fillc, name: "95% band", customdata: d.hi, hovertemplate: "95% band: %{y:.2f} to %{customdata:.2f}<extra></extra>" },
            { x: d.x, y: d.y, type: "scatter", mode: "lines", line: { color, width: 2 }, name: "Effect", hovertemplate: "effect %{y:.3f}<extra></extra>" },
          ]} />
        ))}
      </div> : <Skeleton h={260} />}
      <Note>The x-axes cover each feature's 1st to 99th percentile, because the spline has almost no data beyond that, so they are zoomed. Effective degrees of freedom: {f.n(G.edof, 1)}. CV AUC by λ: {Object.entries<number>(G.cv).map(([k, v]) => `${f.g(Number(k))}: ${f.n(v, 3)}`).join(", ")}.</Note>
    </Card>
  );
}

function Bma() {
  const R = useR(); const B = R.u5.bma; const c = useChart<any>("u5_bma"); const pal = usePalette();
  const row = R.u5.leaderboard.find((r: any) => r.key === "bma");
  const incl = c ? Object.entries<number>(c.inclusion).sort((a, b) => a[1] - b[1]).slice(-10) : [];
  const sizes = Object.entries<number>(B.size_mass);
  return (
    <Card id="bma" title="Bayesian model averaging" q="Instead of picking one model, what if we average over all of them?" m={<><T k="bma">BMA</T> over logistic models on all {f.i(B.n_models)} feature subsets of up to four features, with weights ∝ exp(−BIC/2).</>} tags={["Bayesian model averaging"]}
      meaning={<>The data barely prefers any model. The top model has weight {f.n(B.top_weight, 3)}, and the intercept-only model comes close behind ({f.n(B.top[1].weight, 3)}), meaning that "no features at all" is a serious candidate. The effective number of models is {f.n(B.effective_models, 1)}. BMA accuracy is {f.n(row.accuracy, 4)} against a majority rate of {f.n(R.u5.majority_rate, 4)}, so averaging is honest about the uncertainty, but it does not create a signal.</>}>
      <div className="grid gap-4 lg:grid-cols-2 min-w-0">
        <div className="min-w-0">
          <Table sortable={false} caption="Top models by posterior weight" cols={[{ key: "rank", label: "#", render: (r: any) => r.rank }, { key: "features", label: "Features", align: "left", wrap: true, render: (r: any) => r.features.join(", ") }, { key: "bic", label: "BIC", render: (r: any) => f.n(r.bic, 1) }, { key: "weight", label: "Weight", render: (r: any) => f.n(r.weight, 3) }, { key: "accuracy", label: "Test acc.", render: (r: any) => f.n(r.accuracy, 4) }]} rows={B.top} />
        </div>
        {c ? <Chart ariaLabel="Posterior inclusion probability by feature" height={380} legend={false} x="posterior inclusion probability" layout={{ yaxis: { dtick: 1, tickfont: { size: 11 } }, margin: { l: 90 } }} data={[{ type: "bar", orientation: "h", y: incl.map((x) => x[0]), x: incl.map((x) => x[1]), marker: { color: pal.model("bma") }, hovertemplate: "%{y}<br>inclusion %{x:.3f}<extra></extra>" }]} note="The ten features with the highest inclusion probability." /> : <Skeleton />}
      </div>
      <Note>Mass by subset size: {sizes.map(([k, v]) => `${k} features ${f.pct(v, 1)}`).join(", ")}. So the limit of four features costs little. The {f.i(B.n_kept)} models kept for prediction hold {f.pct(B.kept_mass, 2)} of the mass.</Note>
    </Card>
  );
}

function Robust() {
  const R = useR(); const U = R.u5; const pal = usePalette();
  const models = [["cart", "CART"], ["bagging", "Bagging"], ["random_forest", "Random Forest"], ["adaboost", "AdaBoost"], ["gradient_boosting", "Gradient Boosting"], ["gam", "GAM"], ["bma", "BMA"]];
  const [v, setV] = useState<"acc" | "auc">("acc");
  return (
    <Card id="robust" title="Robustness: TCS, HDFCBANK, INFY" q="Do the same settings find anything on other stocks?" m="Fit the same models with RELIANCE's hyperparameters, with no re-tuning." tags={["Robustness check"]}
      meaning={<>The same picture holds. {U.robust_n_sig} of {U.robust_n_tests} model-and-stock tests have p below 0.05, against about {f.n(U.robust_expected, 1)} expected by chance. The smallest p is {f.n(U.robust_min_p, 3)}, and AUCs run from {f.n(U.robust_auc_min, 3)} to {f.n(U.robust_auc_max, 3)}.</>}>
      <Controls><Segmented label="Metric" value={v} onChange={setV} options={[{ value: "acc", label: "Accuracy" }, { value: "auc", label: "AUC" }]} /></Controls>
      <Table caption="Robustness for ensembles" cols={[
        { key: "name", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.name}</> },
        ...U.robust.map((s: any) => ({ key: s.ticker, label: s.ticker, value: (r: any) => (r.key === "majority" ? s.majority : s[`${r.key}_${v}`]), render: (r: any) => f.n(r.key === "majority" ? s.majority : s[`${r.key}_${v}`], v === "acc" ? 4 : 3) })),
      ]} rows={[...(v === "acc" ? [{ key: "majority", name: "Majority rate" }] : []), ...models.map(([key, name]) => ({ key, name }))]} highlight={(r: any) => r.key === "majority"} />
      {v === "acc" && <Note>Majority rates: {U.robust.map((s: any) => `${s.ticker} ${f.n(s.majority, 4)}`).join(", ")}. Each stock has {U.robust[0].n_test} test rows.</Note>}
    </Card>
  );
}

export default function Unit5() {
  const R = useR(); const U = R.u5; const Cu = U.cart;
  return (
    <>
      <PageHead title="Unit 5: Ensembles" assigned="Ensemble prediction of daily price direction using CART, Random Forest, AdaBoost & Gradient Boosting"
        mapping="CART, boosting (AdaBoost / Gradient Boosting), adaptive basis function models, generalized additive models, ensemble learning (Bagging / Random Forest / Stacking), Bayesian model averaging → daily direction."
        practice="1: CART for categorization. 2: ensemble classifiers." />
      <Leaderboard />
      <Cart />
      <Ensembles />
      <Basis />
      <Importance />
      <Gam />
      <Bma />
      <Robust />
      <Takeaways items={[
        <>{U.n_sig} of {U.n_models} tree, ensemble, GAM and BMA models beat the majority rate of {f.n(U.majority_rate, 4)}, and every AUC is inside the chance band.</>,
        <>An unpruned tree memorises training data (accuracy {f.n(Cu.full_train_acc, 3)}), and test accuracy stays near a coin flip whether it is pruned or not.</>,
        <>Boosting's train loss keeps falling while its test loss rises, so CV keeps choosing weak learners.</>,
        <>Impurity importance does not show out-of-sample value: permutation on test finds no feature clearly above zero.</>,
        <>BMA gives the intercept-only model substantial weight ({f.n(U.bma.top[1].weight, 3)}), which is a Bayesian way of saying that the features add little.</>,
      ]} />
    </>
  );
}
