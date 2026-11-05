import { useMemo, useState } from "react";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { usePalette } from "../lib/colors";
import { useNarrow } from "../lib/useNarrow";
import { Card, Controls, Formula, Kpi, KpiRow, Note, PageHead, Segmented, T, Takeaways } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Table } from "../components/Table";

function Distributions() {
  const R = useR(); const c = useChart<any>("u1_distributions"); const pal = usePalette(); const narrow = useNarrow();
  const [name, setName] = useState("RELIANCE");
  const [logY, setLogY] = useState<"linear" | "log">("linear");
  const d = c?.stocks[name];
  const p = R.u1_dist;
  return (
    <Card id="distributions" title="Return distributions" q={<>What do daily <T k="logret">log-returns</T> look like, and is a Gaussian a fair model?</>}
      m={<>Compare a histogram with Normal and <T k="studentt">Student-t</T> fits, read the <T k="qq">Q-Q plot</T>, and count <T k="sigma4">4σ days</T>. Test normality with <T k="jb">Jarque–Bera</T>.</>} tags={["Continuous RVs", "Probability densities", "Practice 2: summary & statistics"]}
      meaning={<>The mean is close to zero, but the tails are far heavier than a Gaussian's. All {p.stocks_jb_reject} of {p.n_stocks} stocks reject normality, and a Student-t beats the Normal by AIC on {p.n_t_better_aic} of them. Any risk figure based on a Gaussian will understate extreme days.</>}>
      <Controls>
        <label className="text-sm text-ink2 flex items-center gap-2">Series
          <select value={name} onChange={(e) => setName(e.target.value)} className="rounded border border-line bg-surface text-ink px-2 py-1 text-sm max-w-[220px]">
            {c?.names.map((n: string) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <Segmented label="Density axis" value={logY} onChange={setLogY} options={[{ value: "linear", label: "Linear" }, { value: "log", label: "Log (shows tails)" }]} />
      </Controls>
      {d ? (
        <>
          <KpiRow>
            <Kpi label="Mean / std" value={<span className="text-lg">{f.n(d.stats.mean, 5)} / {f.n(d.stats.std, 4)}</span>} note={`${f.i(d.stats.n)} returns`} />
            <Kpi label="Skew / excess kurtosis" value={<span className="text-lg">{f.n(d.stats.skew, 3)} / {f.n(d.stats.ex_kurt, 2)}</span>} note="Gaussian: 0 / 0" />
            <Kpi label="Student-t ν" value={f.n(d.fit.t_df, 2)} note={`log-lik ${f.i(d.fit.ll_t)} vs Normal ${f.i(d.fit.ll_norm)}`} />
            <Kpi label="4σ days" value={f.i(d.tail4.k)} note={`${f.n(d.tail4.expected, 2)} expected: ${f.n(d.tail4.ratio, 0)}× a Gaussian`} />
          </KpiRow>
          <div className="grid gap-4 lg:grid-cols-2 min-w-0">
            <Chart ariaLabel="Histogram of returns with Normal and Student-t fits" height={360} x="daily log-return (axis cut at the plot edges)" y={logY === "log" ? "density (log scale)" : "density"} layout={{ yaxis: { type: logY, ...(logY === "log" ? { range: [-2, 1.7] } : {}) } }} data={[
              { x: d.bin_centers, y: d.density, type: "bar", name: "Observed", marker: { color: pal.s.grey2 }, hovertemplate: "r = %{x:.4f}<br>density %{y:.3f}<extra>Observed</extra>" },
              { x: d.grid, y: d.pdf_norm, type: "scatter", mode: "lines", name: "Normal fit", line: { color: pal.s.orange, width: 2 }, hovertemplate: "r = %{x:.4f}<br>density %{y:.3f}<extra>Normal</extra>" },
              { x: d.grid, y: d.pdf_t, type: "scatter", mode: "lines", name: "Student-t fit", line: { color: pal.s.blue, width: 2 }, hovertemplate: "r = %{x:.4f}<br>density %{y:.3f}<extra>Student-t</extra>" },
            ]} />
            <Chart ariaLabel="Normal Q-Q plot" height={360} x="Normal quantile (z)" y="return quantile" data={[
              { x: d.qq_z, y: d.qq_line, type: "scatter", mode: "lines", name: "Gaussian line", line: { color: pal.s.orange, width: 2 }, hoverinfo: "skip" },
              { x: d.qq_z, y: d.qq_r, type: "scatter", mode: "markers", name: "Observed quantiles", marker: { color: pal.s.blue, size: narrow ? 5 : 6 }, hovertemplate: "z = %{x:.2f}<br>r = %{y:.4f}<extra></extra>" },
            ]} />
          </div>
          <Note>{d.outside > 0 ? `The x-axis is cut at the histogram's range, leaving out ${d.outside} returns beyond it. ` : "No returns fall outside the x-axis range. "}Points leaving the line at both ends of the Q-Q plot show the fat tails.</Note>
        </>
      ) : <Skeleton h={420} />}
    </Card>
  );
}

function VarCard() {
  const R = useR(); const P = R.u1_dist.pooled, Q = R.u1_dist.reliance, V = R.u1_var;
  const rows = [
    { m: "Historical VaR 95%", a: P.hist_var95, b: Q.hist_var95 }, { m: "Gaussian VaR 95%", a: P.gauss_var95, b: Q.gauss_var95 }, { m: "Expected Shortfall 95%", a: P.es95, b: Q.es95 },
    { m: "Historical VaR 99%", a: P.hist_var99, b: Q.hist_var99 }, { m: "Gaussian VaR 99%", a: P.gauss_var99, b: Q.gauss_var99 }, { m: "Expected Shortfall 99%", a: P.es99, b: Q.es99 },
    { m: "Annualised volatility (σ√252)", a: P.ann_vol, b: Q.ann_vol },
  ];
  const b = V.backtest;
  return (
    <Card id="var" title="Value-at-Risk and Expected Shortfall" q="How much can be lost on a bad day, and does an estimate from the past hold up in the future?"
      m={<>Compute <T k="var">VaR</T> as a historical quantile and from a Gaussian, plus <T k="es" />. Back-test on the test period with train-only estimates and a <T k="kupiec">Kupiec</T> test.</>} tags={["Quantiles", "Volatility", "Mean & variance"]}
      meaning={<>At 99%, Gaussian VaR ({f.n(Q.gauss_var99, 4)}) understates the historical figure ({f.n(Q.hist_var99, 4)}) on RELIANCE, because of the fat tails. The historical estimate held up: {b[2].exceedances} exceedances against {f.n(b[2].expected, 2)} expected. But {V.exceed99_march2020} of those {b[2].exceedances} fell in March 2020, and the Kupiec test counts them without noticing that they cluster.</>}>
      <div className="grid gap-4 lg:grid-cols-2 min-w-0">
        <div className="min-w-0">
          <h3 className="font-sans text-sm font-semibold mt-0 mb-1">Daily loss as a positive log-return</h3>
          <Table sortable={false} caption="VaR, ES and volatility" cols={[{ key: "m", label: "Measure", align: "left", wrap: true }, { key: "a", label: "Pooled", render: (r: any) => f.n(r.a, 4) }, { key: "b", label: "RELIANCE", render: (r: any) => f.n(r.b, 4) }]} rows={rows} />
        </div>
        <div className="min-w-0">
          <h3 className="font-sans text-sm font-semibold mt-0 mb-1">Back-test on {V.n_test_returns} test days</h3>
          <Table sortable={false} caption="VaR back-test" cols={[
            { key: "l", label: "Level", align: "left", render: (r: any) => `${r.level}%` }, { key: "method", label: "Method", align: "left" }, { key: "v", label: "VaR", render: (r: any) => f.n(r.var_train, 4) },
            { key: "e", label: "Hits", render: (r: any) => r.exceedances }, { key: "x", label: "Expected", render: (r: any) => f.n(r.expected, 2) }, { key: "p", label: "Kupiec p", render: (r: any) => f.p(r.kupiec_p) },
          ]} rows={b} highlight={(r: any) => r.kupiec_p < 0.05} />
          <Note>Highlighted: rejected at 5% ({V.rejected.join(", ")}). It has too few hits, so it was conservative in this period. The back-test uses {V.n_test_returns} return days, one more than the {R.split.n_test} target rows of the prediction units, because a return exists for the last day but a next-day target does not.</Note>
        </div>
      </div>
    </Card>
  );
}

function BayesCard() {
  const R = useR(); const b = R.u1_bayes;
  const steps = [
    { t: "Count the pairs", body: <>Over {f.i(b.N)} pairs of consecutive days (previous → today): down→down {f.i(b.n00)}, down→up {f.i(b.n01)}, up→down {f.i(b.n10)}, up→up {f.i(b.n11)}.</> },
    { t: "P(up today)", body: <Formula>({f.i(b.n01)} + {f.i(b.n11)}) / {f.i(b.N)} = {f.n(b.P_up, 8)}</Formula> },
    { t: "P(up yesterday)", body: <Formula>({f.i(b.n10)} + {f.i(b.n11)}) / {f.i(b.N)} = {f.n(b.P_prev, 8)}</Formula> },
    { t: "P(up yesterday | up today)", body: <Formula>{f.i(b.n11)} / {f.i(b.n01 + b.n11)} = {f.n(b.P_prev_given_up, 8)}</Formula> },
    { t: "Bayes rule", body: <Formula>P(up | prev up) = {f.n(b.P_prev_given_up, 8)} × {f.n(b.P_up, 8)} / {f.n(b.P_prev, 8)} = {f.n(b.bayes, 8)}</Formula> },
    { t: "Check against a direct count", body: <Formula>{f.i(b.n11)} / {f.i(b.n10 + b.n11)} = {f.n(b.direct, 8)}</Formula> },
  ];
  return (
    <Card id="bayes" title="Bayes rule, step by step" q="If a stock rose yesterday, how likely is it to rise today?" m={<>Define U_t = 1 if r_t &gt; 0, form pairs within each stock, and apply <T k="bayes" /> using the sum and product rules.</>} tags={["Discrete RVs", "Fundamental rules", "Bayes rule"]}
      meaning={<>Bayes rule matches the direct count, as it must. P(up | prev up) = {f.n(b.direct, 4)} looks above one half, and a test against 0.5 gives p = {f.p(b.p_vs_half)}. But that test is misleading, because P(up) = {f.n(b.P_up, 4)} is already above 0.5. Against P(up) itself the same test gives p = {f.p(b.p_vs_pup)}. The fair test is independence, below.</>}>
      <ol className="list-none m-0 p-0 grid gap-2">
        {steps.map((s, i) => (
          <li key={s.t} className="grid grid-cols-[1.75rem_1fr] gap-2 items-baseline">
            <span className="tnum font-semibold text-muted">{i + 1}.</span>
            <div className="min-w-0"><div className="font-medium">{s.t}</div>{s.body}</div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

function IndependenceCard() {
  const R = useR(); const b = R.u1_bayes; const c = useChart<any>("u1_per_stock_bayes"); const pal = usePalette();
  return (
    <Card id="independence" title="Independence" q="Is today's direction independent of yesterday's?" m={<>A <T k="chi2">χ² test</T> on the pooled table, with <T k="cramer" /> as effect size. Then one test per stock with a <T k="bonferroni">Bonferroni</T> correction.</>} tags={["Independence", "Discrete RVs"]}
      meaning={<>Independence is not rejected on pooled data, and the effect is tiny. {b.n_sig5} of {b.n_stocks} stocks look significant at 5% (about {f.n(b.chance_sig, 1)} would by chance), but only {b.n_bonf} survives Bonferroni ({b.bonf_stocks.join(", ")}). The differences go both ways: {b.n_pos_diff} stocks are positive and the rest negative. So there is no general day-to-day momentum in direction.</>}>
      <KpiRow>
        <Kpi label="P(up | prev up)" value={f.n(b.direct, 4)} />
        <Kpi label="P(up | prev down)" value={f.n(b.direct_dn, 4)} />
        <Kpi label="χ² (p-value)" value={<>{f.n(b.chi2, 3)} <span className="text-base">({f.p(b.chi2_p)})</span></>} />
        <Kpi label="Cramér's V" value={f.n(b.V, 4)} />
      </KpiRow>
      {c ? <Chart ariaLabel="Per-stock difference between P(up given previous up) and P(up given previous down)" height={780} heightNarrow={780} legend={false} x="P(up | prev up) − P(up | prev down)" layout={{ xaxis: {}, yaxis: { tickfont: { size: 10 }, dtick: 1, automargin: true }, margin: { l: 90 } }} data={[{
        type: "bar", orientation: "h", y: c.rows.map((r: any) => r.ticker), x: c.rows.map((r: any) => r.diff),
        marker: { color: c.rows.map((r: any) => (r.sig_bonf ? pal.s.red : r.sig5 ? pal.s.orange : pal.s.grey2)) },
        customdata: c.rows.map((r: any) => [r.p, r.V]), hovertemplate: "%{y}<br>difference %{x:.4f}<br>p = %{customdata[0]:.4f}, V = %{customdata[1]:.4f}<extra></extra>",
      }]} note="Grey: not significant. Orange: p below 0.05 uncorrected. Red: survives Bonferroni." /> : <Skeleton />}
    </Card>
  );
}

function ConditionalCard() {
  const R = useR(); const c = R.u1_cond; const b = R.u1_bayes;
  const [z, setZ] = useState("Yesterday's market direction");
  const m = c.summary.find((s: any) => s.Z === "Yesterday's market direction");
  const ind = c.summary.find((s: any) => s.Z === "Industry");
  const vol = c.summary.find((s: any) => s.Z === "Volatility regime");
  return (
    <Card id="conditional" title="Conditional independence and the reversal" q="Does the answer change once we hold something else fixed?"
      m={<>Stratify by industry, volatility regime and yesterday's market direction. Use <T k="cmh">CMH</T> for a common <T k="oddsratio">odds ratio</T>, and <T k="bd">Breslow–Day</T> for equal odds ratios.</>} tags={["Conditional independence"]}
      meaning={<>This is a <T k="simpson" />. The pooled odds ratio is {f.n(b.or_marginal, 4)}, above 1, because stocks co-move: a stock that rose yesterday usually means the market rose ({f.n(c.p_prev_up_by_m["market up yesterday"], 3)} vs {f.n(c.p_prev_up_by_m["market not up yesterday"], 3)}), and an up market yesterday goes with more rises today ({f.n(c.p_up_by_m["market up yesterday"], 3)} vs {f.n(c.p_up_by_m["market not up yesterday"], 3)}). Holding the market fixed, the common odds ratio is {f.n(m.or_common, 4)}, below 1. All effects are tiny (largest V = {f.n(c.max_V, 3)}), and pooled p-values are optimistic because same-day pairs are correlated.</>}>
      <Table sortable={false} caption="Stratified test summary" cols={[
        { key: "Z", label: "Held fixed", align: "left" }, { key: "s", label: "Strata significant", render: (r: any) => `${r.strata_sig} of ${r.n_strata}` }, { key: "m", label: "Pooled OR", render: (r: any) => f.n(r.or_marginal, 4) }, { key: "c", label: "Common OR", render: (r: any) => f.n(r.or_common, 4) },
        { key: "p", label: "CMH p", render: (r: any) => f.p(r.cmh_p) }, { key: "b", label: "Breslow–Day p", render: (r: any) => f.p(r.bd_p) },
      ]} rows={c.summary} highlight={(r: any) => r.Z === z} />
      <Note>Industry: {ind.strata_sig} of {ind.n_strata} industries are significant but with opposite signs. CMH finds no common effect (p = {f.p(ind.cmh_p)}), while Breslow–Day rejects equal odds ratios (p = {f.p(ind.bd_p)}), so the effects cancel when pooled. Volatility regime: {vol.strata_sig} of {vol.n_strata} strata are significant, which is consistent with conditional independence.</Note>
      <div className="mt-3">
        <Controls><Segmented label="Strata for" value={z} onChange={setZ} options={c.summary.map((s: any) => ({ value: s.Z, label: s.Z === "Yesterday's market direction" ? "Market" : s.Z === "Volatility regime" ? "Volatility" : "Industry" }))} /></Controls>
        <Table caption="Per-stratum results" cols={[
          { key: "stratum", label: "Stratum", align: "left" }, { key: "n", label: "Pairs", render: (r: any) => f.i(r.n) }, { key: "p_up_up", label: "P(up|prev up)", render: (r: any) => f.n(r.p_up_up, 4) }, { key: "p_up_dn", label: "P(up|prev down)", render: (r: any) => f.n(r.p_up_dn, 4) },
          { key: "odds_ratio", label: "OR", render: (r: any) => f.n(r.odds_ratio, 3) }, { key: "p", label: "p", render: (r: any) => f.p(r.p) }, { key: "V", label: "V", render: (r: any) => f.n(r.V, 4) },
        ]} rows={c.strata.filter((s: any) => s.Z === z)} highlight={(r: any) => r.p < 0.05} />
      </div>
    </Card>
  );
}

function CovarianceCard() {
  const R = useR(); const c = R.u1_cov; const h = useChart<any>("u1_heatmaps"); const pal = usePalette(); const narrow = useNarrow();
  const [v, setV] = useState<"cov10" | "corr10" | "corr49">("corr10");
  const map = h && { cov10: { z: h.cov10, t: h.tickers10, title: "covariance ×10⁴" }, corr10: { z: h.corr10, t: h.tickers10, title: "correlation" }, corr49: { z: h.corr49, t: h.tickers49, title: "correlation" } }[v];
  return (
    <Card id="covariance" title="Covariance and correlation" q="How do stocks move together, and which measure should we trust?" m={<>Compare <T k="cov" /> and <T k="corr" /> on {c.n_days} common days for ten stocks, then all {c.n_pairs} pairs of stocks by industry.</>} tags={["Expectation & covariance"]}
      meaning={<>Switching returns to percent multiplies Cov(RELIANCE, TCS) by ten thousand ({f.sci(c.cov_rt, 3)} → {f.n(c.cov_rt_pct, 4)}), while correlation stays at {f.n(c.corr_rt, 4)}. Stocks in one industry correlate more ({f.n(c.mean_within, 3)}) than across industries ({f.n(c.mean_cross, 3)}). {c.n_negative === 0 ? `Every pair is positive (minimum ${f.n(c.corr_min, 3)}), ` : `${c.n_negative} pairs are negative, `}which points to a common market factor. Unit 3 tests that directly.</>}>
      <Controls><Segmented label="Show" value={v} onChange={setV} options={[{ value: "corr10", label: "Correlation (10)" }, { value: "cov10", label: "Covariance (10)" }, { value: "corr49", label: "Correlation (all, by industry)" }]} /></Controls>
      {map ? <Chart ariaLabel={`Heatmap of ${map.title}`} legend={false} height={v === "corr49" ? 620 : 460} heightNarrow={v === "corr49" ? 380 : 380} layout={{ xaxis: { tickangle: -60, showticklabels: !(narrow && v === "corr49"), tickfont: { size: v === "corr49" ? 8 : narrow ? 9 : 11 }, dtick: 1, showgrid: false }, yaxis: { autorange: "reversed", showticklabels: !(narrow && v === "corr49"), tickfont: { size: v === "corr49" ? 8 : narrow ? 9 : 11 }, dtick: 1, showgrid: false } }}
        data={[{ type: "heatmap", z: map.z, x: map.t, y: map.t, colorscale: pal.seq, ...(v === "cov10" ? {} : { zmin: 0, zmax: 1 }), colorbar: { title: { text: map.title, side: "right" }, thickness: 12, tickfont: { color: pal.muted } }, hovertemplate: "%{y} and %{x}<br>" + map.title + ": %{z}<extra></extra>", xgap: 1, ygap: 1 }]}
        note={narrow && v === "corr49" ? "Tick labels are hidden at this width. Tap a cell to read its pair and value." : undefined} /> : <Skeleton h={420} />}
      <Note>Highest-correlation pair: {c.max_pair.join(" and ")} ({f.n(c.corr_max, 3)}). Lowest: {c.min_pair.join(" and ")} ({f.n(c.corr_min, 3)}).</Note>
    </Card>
  );
}

function PolyCard() {
  const R = useR(); const c = useChart<any>("u1_poly"); const pal = usePalette();
  const [deg, setDeg] = useState<number | null>(null);
  const P = R.u1_poly;
  const M = deg ?? P.best_future;
  const row = P.table.find((r: any) => r.degree === M);
  const lastDegree = P.table[P.table.length - 1].degree;
  const curve = useMemo(() => (c ? c.curves[M] : []), [c, M]);
  return (
    <Card id="polynomial" title="Polynomial curve fitting" q="Can a polynomial describe the price trend, and can it forecast?" m="Fit degrees across the training period on scaled time, then compare training error, a future hold-out and a random hold-out." tags={["Polynomial curve fitting", "Overfitting"]}
      meaning={<>Training error always falls with degree, to {f.n(P.train_at_12, 1)} INR. The future hold-out is best at degree {P.best_future} ({f.n(P.future_at_best_future, 0)} INR) and explodes to {f.i(P.future_at_12)} INR at the top degree, since polynomials shoot away outside the range they were fitted on. A random hold-out picks degree {P.best_random}, the wrong answer, because it only tests interpolation. A polynomial can describe the past, but it is not a forecast.</>}>
      <label className="block mb-2 text-sm"><span className="text-ink2">Polynomial degree: </span><strong className="tnum">{M}</strong>
        <input type="range" min={P.table[0].degree} max={lastDegree} step={1} value={M} onChange={(e) => setDeg(Number(e.target.value))} aria-label="Polynomial degree" />
      </label>
      <KpiRow>
        <Kpi label="Train RMSE" value={f.n(row.train_rmse, 1)} note="INR" />
        <Kpi label="Future hold-out RMSE" value={f.i(row.future_rmse)} note="INR" />
        <Kpi label="Random hold-out RMSE" value={f.n(row.random_rmse, 1)} note="INR" />
        <Kpi label="Best degree" value={<span className="text-lg">future {P.best_future}, random {P.best_random}</span>} />
      </KpiRow>
      {c ? (
        <div className="grid gap-4 lg:grid-cols-2 min-w-0">
          <Chart ariaLabel="RELIANCE price with polynomial fit" height={360} x="date" y="INR (y-axis clipped)" hover="x" layout={{ yaxis: { range: [0, 3000] }, shapes: [{ type: "line", x0: c.split_date, x1: c.split_date, yref: "paper", y0: 0, y1: 1, line: { color: pal.muted, dash: "dash", width: 1 } }], annotations: [{ x: c.split_date, y: 1, yref: "paper", xanchor: "right", text: "future →", showarrow: false, font: { size: 11, color: pal.muted } }] }} data={[
            { x: c.date, y: c.actual, type: "scatter", mode: "lines", name: "Adjusted Close", line: { color: pal.s.grey, width: 2 }, hovertemplate: "%{y:.1f} INR<extra>Close</extra>" },
            { x: c.date, y: curve, type: "scatter", mode: "lines", name: `Degree ${M} fit`, line: { color: pal.s.blue, width: 2 }, hovertemplate: "%{y:.1f} INR<extra>Fit</extra>" },
          ]} />
          <Chart ariaLabel="RMSE against polynomial degree" height={360} x="polynomial degree" y="RMSE (INR, log scale)" hover="x" layout={{ yaxis: { type: "log" }, xaxis: { dtick: 1 }, shapes: [{ type: "line", x0: M, x1: M, yref: "paper", y0: 0, y1: 1, line: { color: pal.muted, dash: "dot", width: 1 } }] }} data={[
            { x: P.table.map((r: any) => r.degree), y: P.table.map((r: any) => r.train_rmse), type: "scatter", mode: "lines+markers", name: "Train", line: { color: pal.s.blue, width: 2 }, marker: { size: 8 }, hovertemplate: "%{y:.1f}<extra>Train</extra>" },
            { x: P.table.map((r: any) => r.degree), y: P.table.map((r: any) => r.future_rmse), type: "scatter", mode: "lines+markers", name: "Future hold-out", line: { color: pal.s.orange, width: 2 }, marker: { size: 8, symbol: "square" }, hovertemplate: "%{y:.1f}<extra>Future</extra>" },
            { x: P.table.map((r: any) => r.degree), y: P.table.map((r: any) => r.random_rmse), type: "scatter", mode: "lines+markers", name: "Random hold-out", line: { color: pal.s.aqua, width: 2 }, marker: { size: 8, symbol: "diamond" }, hovertemplate: "%{y:.1f}<extra>Random</extra>" },
          ]} />
        </div>
      ) : <Skeleton />}
    </Card>
  );
}

export default function Unit1() {
  const R = useR(); const D = R.u1_dist; const E = R.u1_expect; const Q = D.reliance; const b = R.u1_bayes; const P = R.u1_poly;
  const st = useChart<any>("u1_stock_table");
  return (
    <>
      <PageHead title="Unit 1: Distributions, risk & dependence" assigned="Distribution analysis of daily log-returns of NIFTY-50 stocks; quantiles (Value-at-Risk), volatility & covariance"
        mapping="ML: what & why; supervised vs unsupervised view of the data; polynomial curve fitting (price trend); discrete RVs, fundamental rules & Bayes rule (P(up-day | previous up-day)); independence & conditional independence; continuous RVs: quantiles, mean, variance; probability densities, expectation & covariance."
        practice="1: import, load & view dataset (see Data & pipeline). 2: summary & statistics." />
      <Card id="views" title="ML: what and why, and two views of this data" q="What is machine learning, and what kind of problem is this dataset?" m="Frame the same data as a prediction problem and as a structure-finding problem." tags={["ML: what & why", "Supervised vs unsupervised"]}
        meaning={<>Unit 1 first describes the data with probability, because those properties decide which models make sense. Units 2 to 5 then use both views.</>}>
        <p className="mt-0">Machine learning estimates a mapping or a structure from data, rather than coding rules by hand. A probabilistic treatment also says how uncertain that estimate is.</p>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="rounded border border-line p-3"><strong><T k="supervised" /></strong><br />Each day has inputs (past returns, volatility, volume) and a label known one day later. Next-day Close is regression, and next-day up/down is classification. See Units 2 and 5.</div>
          <div className="rounded border border-line p-3"><strong><T k="unsupervised" /></strong><br />There is no label, so we look for structure: stocks that behave alike, shared market factors and hidden regimes. See Units 3 and 4.</div>
        </div>
      </Card>
      <Card id="summary" title="Summary statistics per stock" q="How do the stocks compare on return, volatility and tail risk?" m={<>Compute the mean, standard deviation, <T k="annvol" />, skew, kurtosis and VaR for each stock. Sort any column.</>} tags={["Practice 2: summary & statistics", "Mean & variance"]}
        meaning={<>Volatility varies by a factor of about two across stocks, from {f.n(D.vol_min, 3)} ({D.vol_min_t}) to {f.n(D.vol_max, 3)} ({D.vol_max_t}). Excess kurtosis is above zero everywhere, from {f.n(D.kurt_min, 2)} to {f.n(D.kurt_max, 2)}, so fat tails are a shared property and not specific to a few stocks.</>}>
        {st ? <Table caption="Per-stock statistics" maxHeight={420} initialSort={{ key: "ann_vol", dir: -1 }} cols={[
          { key: "ticker", label: "Ticker", align: "left" }, { key: "industry", label: "Industry", align: "left" }, { key: "n", label: "Returns", render: (r: any) => f.i(r.n) }, { key: "mean", label: "Mean", render: (r: any) => f.n(r.mean, 5) }, { key: "std", label: "Std", render: (r: any) => f.n(r.std, 4) },
          { key: "ann_vol", label: "Ann. vol", render: (r: any) => f.n(r.ann_vol, 3) }, { key: "skew", label: "Skew", render: (r: any) => f.n(r.skew, 3) }, { key: "ex_kurt", label: "Ex. kurt", render: (r: any) => f.n(r.ex_kurt, 2) }, { key: "t_df", label: "t ν", render: (r: any) => f.n(r.t_df, 2) }, { key: "var99", label: "VaR 99%", render: (r: any) => f.n(r.var99, 4) }, { key: "up_rate", label: "P(up)", render: (r: any) => f.n(r.up_rate, 4) },
        ]} rows={st.rows} highlight={(r: any) => r.ticker === "RELIANCE"} /> : <Skeleton />}
        <Note>RELIANCE, the focus stock, is highlighted. Statistics use each stock's own available history.</Note>
      </Card>
      <Distributions />
      <VarCard />
      <BayesCard />
      <IndependenceCard />
      <ConditionalCard />
      <Card id="expectation" title="Expectation: the law of total expectation" q="Do conditional means add back up to the overall mean?" m={<>Verify the <T k="lte" /> numerically on {f.i(E.n)} pairs, conditioning on the previous direction, the weekday and the industry.</>} tags={["Expectation"]}
        meaning={<>The identity holds to rounding error ({f.sci(E.max_diff, 1)}). The mean after up days ({f.n(E.e_prev_up, 5)}) is above the mean after down days ({f.n(E.e_prev_dn, 5)}), but that is a pooled association and not a trading rule.</>}>
        <Formula>{f.n(E.e_prev_up, 7)} × {f.n(E.p_prev, 7)} + ({f.n(E.e_prev_dn, 7)}) × {f.n(1 - E.p_prev, 7)} = {f.n(E.E, 7)} = E[r]</Formula>
        <Table sortable={false} caption="Reconstruction of the mean" cols={[{ key: "z", label: "Conditioning on", align: "left" }, { key: "levels", label: "Levels", render: (r: any) => r.levels }, { key: "reconstructed", label: "Σ E[r|z]P(z)", render: (r: any) => f.n(r.reconstructed, 10) }, { key: "diff", label: "Gap from E[r]", render: (r: any) => f.sci(r.diff, 1) }]} rows={E.rows} />
      </Card>
      <CovarianceCard />
      <PolyCard />
      <Takeaways items={[
        <>Daily returns have a near-zero mean and heavy tails: pooled excess kurtosis {f.n(D.pooled.ex_kurt, 2)}, and 4σ days {f.n(Q.tail4.ratio, 0)} times more common than a Gaussian predicts on RELIANCE.</>,
        <>Gaussian VaR understates 99% risk ({f.n(Q.gauss_var99, 4)} against {f.n(Q.hist_var99, 4)}), while historical VaR held up in the back-test.</>,
        <>Yesterday's direction carries almost no information about today's (V = {f.n(b.V, 4)}), and testing against 0.5 would have misled us.</>,
        <>Conditioning can reverse an association, as the market-direction reversal shows, though every effect here is tiny.</>,
        <>A random hold-out picks the wrong polynomial degree ({P.best_random}, not {P.best_future}), which is why every later unit splits by time.</>,
      ]} />
    </>
  );
}
