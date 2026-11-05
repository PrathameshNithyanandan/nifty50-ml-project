import { useMemo, useState } from "react";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { usePalette } from "../lib/colors";
import { Card, Controls, Formula, Kpi, KpiRow, Note, PageHead, Segmented, T, Takeaways } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Dot, Table } from "../components/Table";

const label = (r: any) => (r.target === "baseline" ? r.model : `${r.model} [${r.target} | ${r.features}]`);

function RegTable() {
  const R = useR(); const U = R.u2_reg; const pal = usePalette();
  const pers = U.table[0];
  return (
    <Card id="reg-table" title="Next-day Close: model comparison" q="Can any linear model forecast tomorrow's closing price better than today's price?"
      m={<>Target A is Close at t+1 from raw lags. Target B is the return at t+1, converted back to a price. Compare against <T k="persistence" /> with <T k="dm">Diebold–Mariano</T> tests.</>} tags={["Practice 1: linear regression", "MLE / least squares", "Robust regression", "Ridge", "Bayesian linear regression"]}
      meaning={<><strong>No model beats persistence.</strong> {U.n_below} of {U.n_models} have a slightly lower RMSE, and {U.n_sig_better} are significantly better. The price R² of {f.n(U.persistence_r2, 4)} means nothing: persistence itself scores that, because tomorrow's price is close to today's. The return R² shows the real skill, and it is negative for every model.</>}>
      <Table caption="Regression models on the test set" initialSort={{ key: "rmse", dir: 1 }} highlight={(r: any) => r.key === "persistence"} cols={[
        { key: "model", label: "Model [target | features]", align: "left", value: label, render: (r: any) => <><Dot color={pal.model(r.key)} />{label(r)}</> },
        { key: "rmse", label: "RMSE", render: (r: any) => f.n(r.rmse, 3) }, { key: "mae", label: "MAE", render: (r: any) => f.n(r.mae, 3) }, { key: "mape", label: "MAPE %", render: (r: any) => f.n(r.mape, 3) },
        { key: "r2_ret_zero", label: "OOS R² (b=0)", render: (r: any) => f.n(r.r2_ret_zero, 4) }, { key: "r2_ret_mean", label: "OOS R² (b=mean)", render: (r: any) => f.n(r.r2_ret_mean, 4) }, { key: "r2_price", label: "Price R²", render: (r: any) => f.n(r.r2_price, 4) }, { key: "dm_p", label: "DM p", render: (r: any) => f.p(r.dm_p) },
      ]} rows={U.table} />
      <Note>Highlighted row: the baseline. RMSE, MAE and MAPE are on the price scale, in INR. <T k="r2oos">Out-of-sample R²</T> on returns is 1 − Σ(r − r̂)² / Σ(r − b)². Benchmark b = 0 is the zero-return forecast, which is exactly persistence. Benchmark b = mean uses the training mean return ({f.n(U.r_train_mean, 6)}). Target A's return forecast is implied by its price forecast. The RMSE of persistence is {f.n(pers.rmse, 3)} INR.</Note>
    </Card>
  );
}

function RidgeCard() {
  const R = useR(); const U = R.u2_reg; const c = useChart<any>("u2_ridge"); const pal = usePalette();
  const rb = U.ridge.ridge_b_eng, ra = U.ridge.ridge_a, hb = U.huber.huber_b_eng;
  const hues = [pal.s.blue, pal.s.orange, pal.s.aqua, pal.s.violet, pal.s.magenta, pal.s.green, pal.s.yellow, pal.s.red];
  return (
    <Card id="ridge" title="Huber and ridge" q="Do robust loss or shrinkage rescue the forecast?" m={<><T k="huber">Huber</T> limits the pull of large errors. <T k="ridge">Ridge</T> shrinks weights, with α chosen on time-ordered folds.</>} tags={["Robust linear regression", "Ridge regression"]}
      meaning={<>CV picks α = {f.g(rb.best)} on engineered features, and {rb.at_edge ? "this is at the edge of the grid" : "this is inside the grid"}. It shrinks the weight norm from {f.n(rb.ols_coef_norm, 5)} to {f.n(rb.coef_norm, 5)}, so CV prefers little signal over noisy signal. On target A, CV's preference goes flat towards ordinary least squares (α ≈ {f.g(ra.best)}), so ridge and OLS match there. Neither model beats persistence.</>}>
      <KpiRow>
        <Kpi label="Huber ε (target B)" value={hb.epsilon} />
        <Kpi label="Days treated as outliers" value={f.i(hb.outliers)} note={`of ${f.i(hb.n)} training days`} />
        <Kpi label="Ridge α (target B)" value={f.g(rb.best)} />
        <Kpi label="Ridge α (target A)" value={f.g(ra.best)} note="flat towards OLS: reported, not chased" />
      </KpiRow>
      {c ? <Chart ariaLabel="Ridge coefficient path" height={340} legend={false} x="α (log scale)" y="weight on standardised feature" hover="closest" layout={{ xaxis: { type: "log" }, shapes: [{ type: "line", x0: c.chosen, x1: c.chosen, yref: "paper", y0: 0, y1: 1, line: { color: pal.ink2, dash: "dash", width: 1 } }], annotations: [{ x: Math.log10(c.chosen), y: 1, yref: "paper", text: "α from CV", showarrow: false, xanchor: "left", font: { size: 12, color: pal.ink2 } }] }}
        data={c.features.map((n: string, j: number) => ({ x: c.alphas, y: c.path.map((row: number[]) => row[j]), type: "scatter", mode: "lines", name: n, line: { width: 1.5, color: hues[j % hues.length] }, hovertemplate: `${n}<br>α = %{x:.3g}<br>weight %{y:.6f}<extra></extra>` }))} note="Each line is one of the features; hover to identify it. All weights shrink to zero as α grows." /> : <Skeleton />}
    </Card>
  );
}

function BandCard() {
  const R = useR(); const C = R.u2_reg.coverage; const c = useChart<any>("u2_pred"); const pal = usePalette();
  const [m, setM] = useState<"b" | "a">("b");
  const data = useMemo(() => {
    if (!c) return [];
    const lo = m === "b" ? c.lo : c.lo_a, hi = m === "b" ? c.hi : c.hi_a, fc = m === "b" ? c.blr : c.blr_a;
    return [
      { x: c.date_target, y: hi, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
      { x: c.date_target, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: pal.mode === "dark" ? "rgba(144,133,233,0.28)" : "rgba(98,80,214,0.18)", name: "95% predictive interval", customdata: hi, hovertemplate: "95% interval: %{y:.1f} to %{customdata:.1f}<extra></extra>" },
      { x: c.date_target, y: c.actual, type: "scatter", mode: "lines", name: "Actual Close", line: { color: pal.s.grey, width: 1.5 }, hovertemplate: "Actual: %{y:.1f}<extra></extra>" },
      { x: c.date_target, y: fc, type: "scatter", mode: "lines", name: "Forecast", line: { color: pal.model("blr_b_eng"), width: 1.5 }, hovertemplate: "Forecast: %{y:.1f}<extra></extra>" },
    ];
  }, [c, m, pal]);
  return (
    <Card id="band" title="Predicted vs actual, with the Bayesian predictive band" q="Is the model honest about its uncertainty?" m={<>Compare forecasts with actual Close on the test period, and measure 95% <T k="coverage">interval coverage</T> overall and on high-volatility days (20-day volatility above the training upper quartile).</>} tags={["Bayesian linear regression"]}
      meaning={<>On returns, the interval covers {f.pct(C.b_all)} overall but only {f.pct(C.b_high)} on the {C.n_high} high-volatility days, because a constant noise variance cannot follow volatility clustering. The price-level model covers just {f.pct(C.a_all)}: it learned its noise when prices were far lower, and errors grow with the price. Modelling returns gives the better uncertainty, not a better forecast.</>}>
      <Controls><Segmented label="Model" value={m} onChange={setM} options={[{ value: "b", label: "Target B: return" }, { value: "a", label: "Target A: price level" }]} /></Controls>
      <KpiRow>
        <Kpi label="Coverage: overall" value={f.pct(m === "b" ? C.b_all : C.a_all)} note="nominal 95%" />
        <Kpi label="Coverage: high-volatility days" value={f.pct(m === "b" ? C.b_high : C.a_high)} note={`${C.n_high} days`} />
        <Kpi label="Coverage: other days" value={f.pct(m === "b" ? C.b_low : C.a_low)} note={`${C.n_low} days`} />
      </KpiRow>
      {c ? <Chart ariaLabel="Actual and forecast Close with a 95 percent interval" height={400} x="date" y="Close at t+1 (INR)" hover="x unified" data={data} /> : <Skeleton />}
    </Card>
  );
}

function ClfTable() {
  const R = useR(); const U = R.u2_clf; const pal = usePalette();
  return (
    <Card id="clf-table" title="Up/down movement: model comparison" q="Can any classifier predict tomorrow's direction better than always saying up?"
      m={<><T k="lda">LDA</T> (<T k="discriminant">discriminant function</T>), <T k="nb">Gaussian NB</T> (<T k="generative">generative</T>), <T k="logistic">logistic</T> (<T k="discriminative">discriminative</T>), Bayesian logistic and <T k="svm">SVMs</T>. A one-sided <T k="binom">binomial test</T> compares each with the majority rate.</>} tags={["Practice 2: Bayesian logistic & SVM", "Discriminant functions", "Generative & discriminative"]}
      meaning={<><strong>No classifier beats the baseline.</strong> {U.n_above_majority} of {U.n_models} has higher accuracy than {f.n(U.maj_rate, 4)}, {U.n_beat_sig} are significant, and {U.n_auc_outside} have an <T k="auc">AUC</T> outside the chance band of {f.n(U.auc_low, 3)} to {f.n(U.auc_high, 3)}. Several models simply predict up almost every day, so they end up matching the baseline exactly.</>}>
      <Table caption="Classifiers on the test set" highlight={(r: any) => r.kind === "baseline"} cols={[
        { key: "model", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.model}</> }, { key: "accuracy", label: "Accuracy", render: (r: any) => f.n(r.accuracy, 4) }, { key: "f1", label: "F1", render: (r: any) => f.n(r.f1, 3) }, { key: "auc", label: "AUC", render: (r: any) => f.n(r.auc, 3) }, { key: "cv_auc", label: "CV AUC", render: (r: any) => f.n(r.cv_auc, 3) },
        { key: "p_vs_majority", label: "p vs majority", render: (r: any) => (r.key === "majority" ? "—" : f.p(r.p_vs_majority)) }, { key: "share_up", label: "Predicts up", render: (r: any) => f.pct(r.share_up, 1) },
      ]} rows={U.table} />
      <Note>Test P(up) = {f.n(U.p_up_test, 4)} ({U.n1} up days, {U.n0} other days). Training P(up) = {f.n(U.p_up_train, 4)}, so the majority class is "up". Highlighted rows are the baselines.</Note>
    </Card>
  );
}

function Roc() {
  const R = useR(); const c = useChart<any>("u2_roc"); const pal = usePalette();
  return (
    <Card id="roc" title="ROC curves" q="Does any model rank up-days above down-days?" m={<>Plot the <T k="roc">ROC curve</T> for each model that gives a score, and compare AUC with the <T k="auc_band" />.</>} tags={["Evaluation"]}
      meaning={<>All curves lie on the diagonal, which is what guessing looks like. The band is {f.n(R.u2_clf.auc_low, 3)} to {f.n(R.u2_clf.auc_high, 3)}, and every AUC is inside it.</>}>
      {c ? <Chart ariaLabel="ROC curves" height={460} heightNarrow={460} x="false positive rate" y="true positive rate" layout={{ xaxis: { range: [0, 1] }, yaxis: { range: [0, 1] } }} data={[
        ...Object.entries<any>(c.curves).map(([k, v]) => ({ x: v.fpr, y: v.tpr, type: "scatter", mode: "lines", name: `${v.label} (${f.n(v.auc, 3)})`, line: { color: pal.model(k), width: 2 }, hovertemplate: `${v.label}<br>FPR %{x:.3f}, TPR %{y:.3f}<extra></extra>` })),
        { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Chance", line: { color: pal.muted, dash: "dash", width: 1 }, hoverinfo: "skip" },
      ]} /> : <Skeleton h={460} />}
    </Card>
  );
}

function Laplace() {
  const R = useR(); const L = R.u2_clf.laplace; const row = R.u2_clf.table.find((r: any) => r.key === "bayes_logistic");
  return (
    <Card id="laplace" title="The Laplace approximation" q="How do we get a posterior for logistic regression, when none exists in closed form?" m={<>Find the peak with <T k="irls">IRLS</T>, take the curvature there as the posterior precision, then average predictions with the <T k="probit">probit approximation</T>. Written from scratch and checked against scikit-learn.</>} tags={["Laplace approximation", "Bayesian logistic regression"]}
      meaning={<>The fit converges in {L.iters} Newton steps and matches scikit-learn to {f.sci(L.sk_diff, 1)}. Under this heavy prior, the posterior variance of the activation is tiny ({f.sci(L.s2_mean, 1)}), so κ ≈ {f.n(L.kappa_mean, 4)} and the probit step changes predictions by only {f.sci(L.mean_map_probit_diff, 1)}. Predictions stay within {f.n(L.max_abs_p, 3)} of one half and the model predicts up on {f.pct(row.share_up, 0)} of days, so it reduces to the base rate.</>}>
      <ol className="pl-5 m-0 grid gap-1.5 list-decimal">
        <li>Prior: w ~ N(0, α⁻¹I′). I′ is the identity with a zero for the bias, so the bias is not penalised.</li>
        <li><T k="hessian">Hessian</T> of the negative log-posterior: H = ΦᵀRΦ + αI′, with R = diag(yₙ(1−yₙ)).</li>
        <li>Posterior: q(w) = N(w_MAP, S_N), with S_N = H⁻¹.</li>
        <li>Predictive: p(up|x) ≈ σ(κ(σₐ²)·μₐ), with μₐ = w_MAPᵀφ, σₐ² = φᵀS_Nφ and κ(σ²) = (1 + πσ²/8)^(−½).</li>
      </ol>
      <div className="mt-3"><KpiRow>
        <Kpi label="α chosen by CV" value={f.g(L.alpha)} note={L.alpha_plateau ? "CV AUC is flat beyond this" : "inside the grid"} />
        <Kpi label="α with best evidence" value={f.g(L.alpha_evidence)} note={`test AUC ${f.n(L.ev_auc, 3)}`} />
        <Kpi label="Smallest eigenvalue of H" value={f.n(L.min_eig_H, 1)} note="above 0, so the MAP is unique" />
        <Kpi label="Largest eigenvalue of H" value={f.g(L.max_eig_H)} />
      </KpiRow></div>
      <Note>The smallest eigenvalue, {f.n(L.min_eig_H, 1)}, equals H₀₀ = {f.n(L.H00, 1)}, the bias entry, which carries no α because I′ excludes it. That is why it does not grow with α. The data term ΦᵀRΦ alone has a smallest eigenvalue of {f.n(L.min_eig_data, 1)}. The AUC depends only on the direction of w, which stops changing under heavy shrinkage, so CV goes flat and stronger priors change nothing.</Note>
      <div className="mt-3"><Table sortable={false} caption="Newton iterations" cols={[{ key: "it", label: "Iteration", render: (r: any) => r.it }, { key: "psi", label: "Ψ(w)", render: (r: any) => f.n(r.psi, 6) }, { key: "grad_norm", label: "‖gradient‖", render: (r: any) => f.sci(r.grad_norm, 2) }, { key: "step_norm", label: "‖step‖", render: (r: any) => f.sci(r.step_norm, 2) }]} rows={L.log} /></div>
    </Card>
  );
}

function Kernel() {
  const R = useR(); const K = R.u2_clf.kernel; const U = R.u2_clf;
  const vec = (v: number[]) => `(${v.map((x) => f.n(x, 4)).join(", ")})`;
  return (
    <Card id="kernel" title="The kernel trick" q="How can an SVM work in a huge feature space without ever building it?" m={<>Check that a <T k="kernel" /> equals an inner product in feature space, check that the <T k="rbf">RBF</T> <T k="gram">Gram matrix</T> is <T k="psd">PSD</T>, then tune linear and RBF SVMs.</>} tags={["Kernel functions", "Kernel trick", "SVM"]}
      meaning={<>The kernel gives the same number as the {K.phi_dim}-dimensional inner product with only {K.d} inputs (largest gap {f.sci(K.diff, 1)} over {K.n_pairs} pairs). That is the trick. But more flexibility does not find a signal: the RBF SVM keeps {f.i(U.rbf.n_sv)} of {f.i(U.rbf.n_train)} training days as support vectors and predicts up on nearly every test day.</>}>
      <h3 className="font-sans text-sm font-semibold mt-0 mb-1">Worked example in two dimensions: φ(x) = (x₁², √2·x₁x₂, x₂²)</h3>
      <Formula>x = {vec(K.x)}, z = {vec(K.z)}<br />(xᵀz)² = ({f.n(K.xz, 4)})² = {f.n(K.lhs, 4)}<br />φ(x) = {vec(K.phi_x)}, φ(z) = {vec(K.phi_z)}<br />φ(x)ᵀφ(z) = {f.n(K.rhs, 4)}</Formula>
      <KpiRow>
        <Kpi label="Identity gap" value={f.sci(K.diff, 1)} note={`${K.n_pairs} pairs in ${K.d} dimensions`} />
        <Kpi label="Gram: smallest eigenvalue" value={f.sci(K.gram_min_eig, 1)} note={`${K.gram_n}×${K.gram_n} matrix`} />
        <Kpi label="RBF SVM" value={<span className="text-lg">C = {U.rbf.C}, γ = {U.rbf.gamma}</span>} note="inside the grid" />
        <Kpi label="Linear SVM" value={<span className="text-lg">C = {f.g(U.linear_svm.C)}</span>} note="at the edge: CV flat beyond" />
      </KpiRow>
    </Card>
  );
}

export default function Unit2() {
  const R = useR(); const U = R.u2_reg, V = R.u2_clf, B = U.blr, O = U.ols, Q = R.u2_robust; const pal = usePalette();
  return (
    <>
      <PageHead title="Unit 2: Regression & classification" assigned="Next-day closing price prediction with ridge & Bayesian linear regression; up/down movement classification with Bayesian logistic regression & SVM"
        mapping="MLE / least squares, robust linear regression, ridge & Bayesian linear regression → next-day Close; discriminant functions, probabilistic generative & discriminative models, Laplace approximation & Bayesian logistic regression, kernel functions / kernel trick, SVM → up/down direction."
        practice="1: linear regression for prediction. 2: Bayesian logistic regression & SVM." />
      <RegTable />
      <Card id="mle" title="Ordinary least squares equals maximum likelihood" q="Why is minimising squared error a probabilistic method?" m={<>Under Gaussian noise, the log-likelihood is a constant minus (β/2)Σ(t − wᵀφ)². Compare the closed form with a numerical optimiser that maximises the full likelihood.</>} tags={["MLE", "Least squares"]}
        meaning={<>The two routes agree to {f.sci(O.mle_diff, 1)} in every weight, so <T k="ols">OLS</T> is <T k="mle">MLE</T> under Gaussian noise. With fat-tailed returns the Gaussian assumption is wrong, which is why Huber is worth trying.</>}>
        <KpiRow>
          <Kpi label="max |w closed − w numerical|" value={f.sci(O.mle_diff, 1)} />
          <Kpi label="max |w closed − w sklearn|" value={f.sci(O.sk_diff, 1)} />
          <Kpi label="β closed / numerical" value={<span className="text-lg">{f.n(O.beta_closed, 3)} / {f.n(O.beta_mle, 3)}</span>} />
          <Kpi label="Noise σ" value={f.n(O.sigma_ml, 4)} note="daily log-return" />
        </KpiRow>
      </Card>
      <RidgeCard />
      <Card id="blr" title="Bayesian linear regression from scratch" q="Can the data choose the amount of regularisation, without a validation set?" m={<>Prior w ~ N(0, α⁻¹I). <T k="evidence">Evidence maximisation</T> iterates γ = Σλᵢ/(α+λᵢ), α = γ/mᵀm and 1/β = ‖t − Φm‖²/(N − γ).</>} tags={["Bayesian linear regression"]}
        meaning={<>On returns, the evidence keeps only γ = {f.n(B.b.gamma, 2)} <T k="gamma">effective parameters</T> of {B.b.n_params}, a Bayesian way of saying that most features carry little. It matches scikit-learn to {f.sci(B.sk_diff_w, 1)}. On target A, γ = {f.n(B.a.gamma, 2)} of {B.a.n_params}, since price lags are strongly informative about the next price.</>}>
        <Table sortable={false} caption="Evidence-maximised hyperparameters" cols={[{ key: "t", label: "Target", align: "left" }, { key: "a", label: "α", render: (r: any) => f.g(r.v.alpha) }, { key: "b", label: "β", render: (r: any) => f.g(r.v.beta) }, { key: "g", label: "γ", render: (r: any) => f.n(r.v.gamma, 3) }, { key: "p", label: "Features", render: (r: any) => r.v.n_params }, { key: "i", label: "Iterations", render: (r: any) => r.v.iters }]}
          rows={[{ t: "B: next-day return", v: B.b }, { t: "A: next-day Close", v: B.a }]} />
        <Note>Against sklearn's BayesianRidge with its hyperpriors switched off: relative difference in α {f.sci(B.sk_diff_alpha, 1)}, and in β {f.sci(B.sk_diff_beta, 1)}.</Note>
      </Card>
      <BandCard />
      <ClfTable />
      <Roc />
      <Laplace />
      <Kernel />
      <Card id="robust" title="Robustness: TCS, HDFCBANK, INFY" q="Is RELIANCE a special case?" m="Apply the same pipeline, split date and RELIANCE's hyperparameters to three other stocks, with no re-tuning." tags={["Robustness check"]}
        meaning={<>The conclusion carries over. Ridge and Bayesian LR stay within {f.pct(Q.max_ratio_dev, 2)} of the persistence RMSE. {Q.n_dm_sig} Diebold–Mariano test is significant ({Q.dm_sig_label}), and it shows the model doing worse than persistence, not better. The smallest p-value for any classifier against its majority rate is {f.n(Q.min_p, 3)}.</>}>
        <Table sortable={false} caption="Robustness results" cols={[
          { key: "ticker", label: "Stock", align: "left" }, { key: "n_test", label: "Test rows" }, { key: "c", label: "Corp. actions in test", render: (r: any) => r.corp_actions_in_test }, { key: "a", label: <><Dot color={pal.model("persistence")} />Persistence RMSE</>, render: (r: any) => f.n(r.persistence_rmse, 3) }, { key: "b", label: "Ridge / persistence", render: (r: any) => f.n(r.ridge_ratio, 4) }, { key: "d", label: "Bayesian LR / persistence", render: (r: any) => f.n(r.blr_ratio, 4) }, { key: "e", label: "Bayesian LR DM p", render: (r: any) => f.p(r.blr_dm_p) }, { key: "g", label: "Coverage", render: (r: any) => f.pct(r.blr_coverage) },
          { key: "m", label: "Majority acc.", render: (r: any) => f.n(r.majority_acc, 4) }, { key: "l", label: "Laplace acc.", render: (r: any) => f.n(r.laplace_acc, 4) }, { key: "s", label: "Linear SVM acc.", render: (r: any) => f.n(r.linear_svm_acc, 4) }, { key: "r", label: "RBF SVM acc.", render: (r: any) => f.n(r.rbf_acc, 4) },
        ]} rows={Q.rows} />
        <Note>These stocks have corporate actions inside the test window, so RMSE is in adjusted INR and the ratio is what matters. Rows that touch a corporate-action day drop out, which is why they have fewer test rows than RELIANCE.</Note>
      </Card>
      <Takeaways items={[
        <>{U.n_sig_better} of {U.n_models} regression models beat persistence, and {V.n_beat_sig} of {V.n_models} classifiers beat the majority rate. These are honest near-coin-flip results.</>,
        <>A price R² of {f.n(U.persistence_r2, 3)} is not skill, since persistence scores it too. Report R² on returns against a stated benchmark.</>,
        <>Bayesian methods do add honest uncertainty: {f.pct(U.coverage.b_all)} coverage overall, though {f.pct(U.coverage.b_high)} on volatile days.</>,
        <>Every from-scratch method matches its library version: OLS and MLE, evidence-maximised Bayesian LR and Laplace logistic regression.</>,
        <>CV keeps choosing heavy shrinkage. When there is no signal, the best model is close to the base rate.</>,
      ]} />
    </>
  );
}
