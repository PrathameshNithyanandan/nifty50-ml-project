import { useMemo, useState } from "react";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { usePalette } from "../lib/colors";
import { useNarrow } from "../lib/useNarrow";
import { Card, Controls, Formula, Kpi, KpiRow, Note, PageHead, Segmented, T, Takeaways } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Dot, Table } from "../components/Table";

type S = "ew" | "reliance";
const SERIES = [{ value: "ew" as S, label: "Equal-weighted index" }, { value: "reliance" as S, label: "RELIANCE" }];
const rgba = (hex: string, a: number) => `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${a})`;

function Regimes() {
  const R = useR(); const c = useChart<any>("u4_regimes"); const pal = usePalette(); const narrow = useNarrow();
  const [s, setS] = useState<S>("ew");
  const [zoom, setZoom] = useState<"all" | "gfc" | "covid">("all");
  const d = c?.[s]; const M = R.u4[s];
  const ranges: Record<string, [string, string] | undefined> = { all: undefined, gfc: [M.checks.gfc.start, M.checks.gfc.end], covid: ["2020-01-15", "2020-07-15"] };
  const layout = useMemo(() => {
    if (!d) return {};
    const shapes = d.segments.map(([a, b, st]: [string, string, number]) => ({ type: "rect", xref: "x", yref: "paper", x0: a, x1: b, y0: 0, y1: 1, line: { width: 0 }, fillcolor: rgba(pal.regime(d.names[st]), pal.mode === "dark" ? 0.24 : 0.22), layer: "below" }));
    shapes.push({ type: "line", x0: d.split, x1: d.split, yref: "paper", y0: 0, y1: 1, line: { color: pal.ink2, dash: "dash", width: 1 } } as any);
    return {
      shapes, yaxis: { type: "log" }, xaxis: ranges[zoom] ? { range: ranges[zoom] } : { autorange: true },
      annotations: [
        { x: "2008-10-15", y: 1, yref: "paper", yanchor: "top", text: "2008", showarrow: false, font: { size: 12, color: pal.ink } },
        { x: "2020-03-23", y: 1, yref: "paper", yanchor: "top", xanchor: "right", text: "2020 ", showarrow: false, font: { size: 12, color: pal.ink } },
        { x: d.split, y: 0, yref: "paper", yanchor: "bottom", xanchor: "left", text: narrow ? " test →" : " test period →", showarrow: false, font: { size: 11, color: pal.ink2 } },
      ],
    };
  }, [d, zoom, pal, narrow]);
  return (
    <Card id="regimes" title="Hidden market regimes" q="Can a model find bull, bear and sideways markets with no labels?" m={<>A Gaussian <T k="hmm">HMM</T> on daily returns, fitted on the training period only. <T k="viterbi">Viterbi decoding</T> gives the most probable state path, shaded on the price.</>} tags={["Practice 1: HMM", "Markov models", "Viterbi decoding"]}
      meaning={<>The model finds both crises without being told about them. Bear covers {f.pct(M.checks.overall.bear_share)} of all days, but {f.pct(M.checks.oct2008.bear_share, 0)} of mid-September to November 2008 and {f.pct(M.checks.covid.bear_share, 0)} of March 2020, using parameters fitted on the training period only. Viterbi uses the whole series, including future days, so this chart describes the past and cannot be used for forecasting.</>}>
      <Controls>
        <Segmented label="Series" value={s} onChange={setS} options={SERIES} />
        <Segmented label="Zoom" value={zoom} onChange={setZoom} options={[{ value: "all", label: "All" }, { value: "gfc", label: "2008" }, { value: "covid", label: "2020" }]} />
      </Controls>
      {d ? <Chart ariaLabel="Price with Viterbi regimes shaded" height={420} legend={true} x="date" y={s === "ew" ? "index level (log scale)" : "INR (log scale)"} hover="x" layout={layout} data={[
        { x: d.date, y: d.price, type: "scatter", mode: "lines", name: d.label, line: { color: pal.ink, width: 1.5 }, hovertemplate: "%{x}<br>%{y:.2f}<extra></extra>" },
        ...d.names.map((n: string) => ({ x: [null], y: [null], type: "scatter", mode: "markers", name: n, marker: { size: 11, symbol: "square", color: rgba(pal.regime(n), pal.mode === "dark" ? 0.6 : 0.5) }, hoverinfo: "skip" })),
      ]} /> : <Skeleton h={420} />}
      <Note>The line is reduced to about {d ? f.i(d.date.length) : "—"} points that keep each bucket's lowest and highest values, so crash extremes remain. The shading uses every day. On a laptop you can also drag to zoom and double-click to reset.</Note>
    </Card>
  );
}

function States() {
  const R = useR(); const pal = usePalette();
  const [s, setS] = useState<S>("ew"); const M = R.u4[s];
  const rows = M.names.map((n: string, i: number) => ({ n, mean: M.mean[i], sd: M.sd[i], a: M.A[i][i], d: M.duration[i], st: M.stationary[i], occ: M.occupancy[i] }));
  const e = R.u4.ew, r = R.u4.reliance;
  return (
    <Card id="states" title="States, transitions and durations" q="What do the regimes look like, and how long do they last?" m={<>Name states from their parameters to handle <T k="labelswitch">label switching</T>: bear has the lowest mean, bull the highest. Expected duration is 1/(1−a_ii).</>} tags={["Markov models", "HMM"]}
      meaning={<>Regimes differ more in volatility than in direction. On the index, bull lasts about {f.n(e.duration[2], 0)} days with sd {f.n(e.sd[2], 2)}%, while bear lasts about {f.n(e.duration[0], 0)} days with sd {f.n(e.sd[0], 2)}%. On RELIANCE, the highest-mean state ({f.s(r.mean[2], 3)}%) has a higher sd ({f.n(r.sd[2], 2)}%) than the middle one ({f.n(r.sd[1], 2)}%), so "bull" there is not the calmest state. The name is a shorthand for the mean only.</>}>
      <Controls><Segmented label="Series" value={s} onChange={setS} options={SERIES} /></Controls>
      <Table sortable={false} caption="State parameters" cols={[
        { key: "n", label: "State", align: "left", render: (x: any) => <><Dot color={pal.regime(x.n)} />{x.n}</> }, { key: "m", label: "Mean (%/day)", render: (x: any) => f.s(x.mean, 3) }, { key: "s", label: "SD (%/day)", render: (x: any) => f.n(x.sd, 2) }, { key: "a", label: "Stay prob. a_ii", render: (x: any) => f.n(x.a, 4) },
        { key: "d", label: "Expected days", render: (x: any) => f.n(x.d, 1) }, { key: "st", label: "Stationary share", render: (x: any) => f.pct(x.st) }, { key: "o", label: "Viterbi share", render: (x: any) => f.pct(x.occ) },
      ]} rows={rows} />
      <h3 className="font-sans text-sm font-semibold mt-4 mb-1">Transition matrix (row: from, column: to)</h3>
      <div className="overflow-x-auto rounded border border-line" tabIndex={0} role="region" aria-label="Transition matrix">
        <table className="w-full border-collapse text-sm tnum">
          <thead className="bg-sunk"><tr><th className="px-2.5 py-1.5 text-left">From \ to</th>{M.names.map((n: string) => <th key={n} className="px-2.5 py-1.5 text-right">{n}</th>)}</tr></thead>
          <tbody>{M.A.map((row: number[], i: number) => (
            <tr key={i} className="border-t border-line"><th className="px-2.5 py-1.5 text-left font-medium">{M.names[i]}</th>
              {row.map((v: number, j: number) => <td key={j} className="px-2.5 py-1.5 text-right" style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(v * 40)}%, transparent)` }}>{f.n(v, 4)}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
      {M.lab_demo && <Note>Label switching seen directly: two restarts reach the same likelihood ({f.n(M.lab_demo.ll_a, 2)} and {f.n(M.lab_demo.ll_b, 2)}), with state means in different orders: ({M.lab_demo.raw_means_a.map((x: number) => f.n(x, 3)).join(", ")}) and ({M.lab_demo.raw_means_b.map((x: number) => f.n(x, 3)).join(", ")}).</Note>}
    </Card>
  );
}

function Lookahead() {
  const R = useR(); const c = useChart<any>("u4_lookahead"); const pal = usePalette();
  const [s, setS] = useState<S>("ew"); const [w, setW] = useState<"covid" | "gfc">("covid");
  const d = c?.[s]?.[w]; const L = R.u4.lookahead[s][w]; const M = R.u4[s]; const Fe = R.u4.forecast.ew;
  return (
    <Card id="lookahead" title="The look-ahead problem: filtered vs smoothed" q="Why can't we forecast with the regimes in the chart above?" m={<><T k="filtered">Filtered</T> P(bear) uses returns up to today. <T k="smoothed">Smoothed</T> P(bear) also uses later returns. Plot both through a crash.</>} tags={["Forward–backward", "HMM"]}
      meaning={<>The smoothed line knows what comes next: in this window it passes 0.5 on {L.first_smoothed}, against {L.first_filtered} for the filtered line, {L.lead_days} trading day(s) earlier. Used as a forecast, it would be cheating: on the index, smoothed probabilities appear to give accuracy {f.n(Fe.smoothed_wrong.accuracy, 4)} and AUC {f.n(Fe.smoothed_wrong.auc, 3)}, against an honest {f.n(Fe.filtered.accuracy, 4)} and {f.n(Fe.filtered.auc, 3)}. Only the filtered line is used for forecasts.</>}>
      <Controls>
        <Segmented label="Series" value={s} onChange={setS} options={SERIES} />
        <Segmented label="Window" value={w} onChange={setW} options={[{ value: "covid", label: "2020" }, { value: "gfc", label: "2008" }]} />
      </Controls>
      {d ? <Chart ariaLabel="Filtered and smoothed probability of the bear state" height={360} x="date" y="P(bear state)" hover="x" layout={{ yaxis: { range: [0, 1] } }} data={[
        { x: d.date, y: d.filtered, type: "scatter", mode: "lines", name: "Filtered (usable)", line: { color: pal.s.blue, width: 2 }, hovertemplate: "Filtered %{y:.3f}<extra></extra>" },
        { x: d.date, y: d.smoothed, type: "scatter", mode: "lines", name: "Smoothed (uses the future)", line: { color: pal.s.orange, width: 2, dash: "dash" }, hovertemplate: "Smoothed %{y:.3f}<extra></extra>" },
      ]} /> : <Skeleton />}
      <Note>Across all days, the most probable state under the filter differs from the smoothed one on {f.pct(M.f_vs_s)} of days, and from Viterbi on {f.pct(M.f_vs_v)}.</Note>
    </Card>
  );
}

function Forecast() {
  const R = useR(); const Fc = R.u4.forecast; const c = useChart<any>("u4_forecast"); const pal = usePalette();
  const [s, setS] = useState<S>("ew"); const F = Fc[s]; const d = c?.[s];
  return (
    <Card id="forecast" title="HMM forecast vs baseline" q="Does knowing the regime help predict tomorrow?" m={<>One-step predictive distribution from filtered probabilities only, for the same {F.n} target days as Units 2 and 5. Compare direction with the majority rate, and density with i.i.d. models using <T k="lpd">log predictive density</T>.</>} tags={["Practice 1: predict sequential data", "HMM"]}
      meaning={<><strong>For direction, no:</strong> accuracy is {f.n(F.filtered.accuracy, 4)} against a majority rate of {f.n(F.majority_rate, 4)} (p = {f.p(F.filtered.p_vs_majority)}). <strong>For risk, yes:</strong> the log predictive density beats an i.i.d. Normal and an i.i.d. Student-t, and the forecast sd moves from {f.n(F.sd_forecast_min, 2)}% to {f.n(F.sd_forecast_max, 2)}%, tracking volatility. Regimes tell us how much the market may move, not which way.</>}>
      <Controls><Segmented label="Series" value={s} onChange={setS} options={SERIES} /></Controls>
      <KpiRow>
        <Kpi label="HMM accuracy" value={f.n(F.filtered.accuracy, 4)} note={`${F.filtered.n_correct} of ${F.filtered.n}`} />
        <Kpi label="Majority rate" value={f.n(F.majority_rate, 4)} note={`p vs majority ${f.p(F.filtered.p_vs_majority)}`} />
        <Kpi label="Mean log density" value={f.n(F.lpd_hmm, 3)} note={`Normal ${f.n(F.lpd_normal, 3)}, Student-t ${f.n(F.lpd_t, 3)}`} />
        <Kpi label="95% interval coverage" value={f.pct(F.coverage95)} note={`i.i.d. Normal ${f.pct(F.coverage95_normal)}`} />
      </KpiRow>
      {d ? <Chart ariaLabel="Forecast standard deviation and absolute realised return" height={340} x="target date" y="% per day" hover="x" data={[
        { x: d.date, y: d.realised.map((v: number) => Math.abs(v)), type: "scatter", mode: "lines", name: "|Realised return|", line: { color: pal.s.grey2, width: 1 }, hovertemplate: "|return| %{y:.2f}%<extra></extra>" },
        { x: d.date, y: d.sd, type: "scatter", mode: "lines", name: "Forecast SD (filtered)", line: { color: pal.s.blue, width: 2 }, hovertemplate: "forecast SD %{y:.2f}%<extra></extra>" },
      ]} /> : <Skeleton />}
      <Note>Newey–West t-statistic for the density gain over the Normal: {f.n(F.lpd_t_stat, 2)} (p = {f.p(F.lpd_p)}). Mean |P(up) − 0.5| = {f.n(F.mean_abs_p, 3)}. The test period is one window with a large crash in it, so the density gain partly reflects that episode.</Note>
    </Card>
  );
}

function KalmanCard() {
  const R = useR(); const K = R.u4.kalman; const c = useChart<any>("u4_kalman"); const pal = usePalette();
  const [model, setModel] = useState<"mle" | "ill">("ill");
  const [show, setShow] = useState<"both" | "filtered" | "smoothed">("both");
  const [zoom, setZoom] = useState<"zoom" | "all">("zoom");
  const d = zoom === "zoom" ? c?.zoom : c;
  const traces = useMemo(() => {
    if (!d) return [];
    const t: any[] = [{ x: d.date, y: d.close, type: "scatter", mode: "lines", name: "Adjusted Close", line: { color: pal.s.grey, width: 1.5 }, hovertemplate: "Close %{y:.1f}<extra></extra>" }];
    if (show !== "smoothed") t.push({ x: d.date, y: d[`${model}_filtered`], type: "scatter", mode: "lines", name: "Filtered", line: { color: pal.s.blue, width: 2 }, hovertemplate: "Filtered %{y:.1f}<extra></extra>" });
    if (show !== "filtered") t.push({ x: d.date, y: d[`${model}_smoothed`], type: "scatter", mode: "lines", name: "Smoothed (uses the future)", line: { color: pal.s.orange, width: 2, dash: "dash" }, hovertemplate: "Smoothed %{y:.1f}<extra></extra>" });
    return t;
  }, [d, model, show, pal]);
  return (
    <Card id="kalman" title="Kalman filter for the price trend" q="What is the underlying trend of RELIANCE's price?" m={<>A <T k="lds">linear dynamical system</T> on 100·ln(Close): <T k="locallevel">local level</T> and <T k="llt">local linear trend</T>. Q and R come from MLE on training data. Filter and <T k="rts">RTS smoother</T> are from scratch, checked against statsmodels.</>} tags={["Linear dynamical system", "Kalman filter"]}
      meaning={<>MLE pushes R to its boundary ({f.sci(K.ll.R, 1)}, against Q = {f.n(K.ll.Q[0], 2)}), so the <T k="gain">gain</T> is {f.n(K.gain, 4)} and the filtered level is just today's price. There is a reason: this model implies lag-1 autocorrelation of −R/(Q+2R) ≤ 0 in daily changes, but training returns have {f.s(K.autocorr1, 4)}. To see a real trend line, switch to the hand-fixed q = {K.q_fixed} (gain {f.n(K.gain_fixed, 3)}, mean lag {f.n(K.mean_lag, 1)} days). That line lags the price, and the smoother avoids the lag only by using future data.</>}>
      <Controls>
        <Segmented label="Model" value={model} onChange={setModel} options={[{ value: "ill", label: `q = ${K.q_fixed} (by hand)` }, { value: "mle", label: "MLE" }]} />
        <Segmented label="Show" value={show} onChange={setShow} options={[{ value: "both", label: "Both" }, { value: "filtered", label: "Filtered" }, { value: "smoothed", label: "Smoothed" }]} />
        <Segmented label="Range" value={zoom} onChange={setZoom} options={[{ value: "zoom", label: "2020" }, { value: "all", label: "All" }]} />
      </Controls>
      {d ? <Chart ariaLabel="Close with filtered and smoothed trend" height={400} x="date" y="INR" hover="x unified" layout={zoom === "all" ? { yaxis: { type: "log", title: { text: "INR (log scale)" } } } : {}} data={traces} /> : <Skeleton />}
      <Formula>x(t+1) = T·x(t) + w, y(t) = Z·x(t) + v, with Var(w) = Q and Var(v) = R</Formula>
      <Table sortable={false} caption="From-scratch filter against statsmodels" cols={[
        { key: "n", label: "Model", align: "left" }, { key: "r", label: "R (MLE)", render: (x: any) => f.sci(x.v.R, 1) }, { key: "q", label: "Q (MLE)", render: (x: any) => x.v.Q.map((q: number) => (q < 1e-3 ? f.sci(q, 1) : f.n(q, 3))).join(", ") }, { key: "l", label: "ln L", render: (x: any) => f.n(x.v.ll, 3) }, { key: "s", label: "statsmodels ln L", render: (x: any) => f.n(x.v.sm_ll, 3) },
        { key: "a", label: "Filter gap", render: (x: any) => f.sci(x.v.filt_diff, 1) }, { key: "b", label: "Smoother gap", render: (x: any) => f.sci(x.v.smooth_diff, 1) },
      ]} rows={[{ n: "Local level", v: K.ll }, { n: "Local linear trend", v: K.llt }]} />
      <Note>Filter and smoother gaps compare both implementations at the same parameters. The line in the "All" view is reduced to about {c ? f.i(c.date.length) : "—"} points, keeping extremes.</Note>
    </Card>
  );
}

export default function Unit4() {
  const R = useR(); const U = R.u4; const K = U.kalman; const pal = usePalette();
  const [s, setS] = useState<S>("ew");
  const M = U[s]; const B = U.brute_force;
  const fp = K.forecast[0];
  return (
    <>
      <PageHead title="Unit 4: HMM & Kalman" assigned="HMM for detecting hidden market regimes (bull / bear / sideways) with Viterbi decoding; Kalman filter for price trend"
        mapping="Markov models, HMM, maximum likelihood for HMM (Baum–Welch), forward–backward & sum-product algorithms, scaling factors, Viterbi decoding → daily return regimes; linear dynamical system / Kalman filter → Kalman filtering of closing price."
        practice="1: HMM to predict sequential data." />
      <Regimes />
      <States />
      <Card id="selection" title="Choosing the number of states" q="How many regimes does the data support?" m={<>K = 2, 3 and 4 states, each with {M.table[0].n_restart} random restarts on the training period. Select by <T k="bic">BIC</T>.</>} tags={["Maximum likelihood for HMM"]}
        meaning={<>BIC selects K = {M.K}, which fits the title's three regimes, though that was not forced. AIC would allow more states. Restarts matter when K is larger: at the largest K, only {M.table[M.table.length - 1].n_at_best} of {M.table[0].n_restart} restarts reach the best likelihood.</>}>
        <Controls><Segmented label="Series" value={s} onChange={setS} options={SERIES} /></Controls>
        <Table sortable={false} caption="HMM selection" highlight={(r: any) => r.K === M.K} cols={[{ key: "K", label: "States K", render: (r: any) => r.K }, { key: "n_params", label: "Parameters", render: (r: any) => r.n_params }, { key: "ll", label: "Best ln L", render: (r: any) => f.n(r.ll, 2) }, { key: "w", label: "Worst restart", render: (r: any) => f.n(r.ll_worst, 2) }, { key: "bic", label: "BIC", render: (r: any) => f.n(r.bic, 1) }, { key: "aic", label: "AIC", render: (r: any) => f.n(r.aic, 1) }, { key: "n_at_best", label: "Restarts at best", render: (r: any) => `${r.n_at_best} of ${r.n_restart}` }]} rows={M.table} />
      </Card>
      <Card id="match" title="From-scratch algorithms and their checks" q="Are the hand-written algorithms correct?" m={<><T k="baumwelch">Baum–Welch</T>, <T k="fb">forward–backward</T> with <T k="scaling">scaling factors</T>, and Viterbi in log space. Compare with hmmlearn and with brute-force enumeration.</>} tags={["Baum–Welch", "Forward–backward", "Sum-product", "Scaling factors", "Viterbi"]}
        meaning={<>Without scaling, the forward variable underflows to exactly 0 by day {U.underflow_day} of {f.i(U.n_train_days)}. With scaling, the log-likelihood matches a log-space check to {f.sci(U.ll_scaling_diff, 1)}. Forward–backward is <T k="sumproduct" /> on a chain: α is the left-to-right message, β is the right-to-left message, and p(z_t | r) ∝ αβ. Swapping the sum for a max gives Viterbi. Brute force over all {f.i(B.paths)} paths of a short chain agrees.</>}>
        <KpiRow>
          <Kpi label="ln L gap vs hmmlearn" value={f.sci(M.cmp.ll_diff, 1)} note={`${M.cmp.iters} Baum–Welch iterations`} />
          <Kpi label="Max parameter gap" value={f.sci(Math.max(M.cmp.mu_diff, M.cmp.var_diff, M.cmp.A_diff), 1)} note="means, variances, transitions" />
          <Kpi label="Viterbi agreement" value={f.pct(M.vit_agree)} note={`posterior gap ${f.sci(M.gam_diff, 1)}`} />
          <Kpi label="Brute force" value={<span className="text-base">{B.viterbi_match ? "Viterbi path matches" : "mismatch"}</span>} note={`${B.T} days, ${B.K} states; marginal gap ${f.sci(B.marg_diff, 1)}`} />
        </KpiRow>
        <Formula>c(t) = p(r_t | r_1..t−1), α̂(t) = p(z_t | r_1..t), ln p(r) = Σ ln c(t) = {f.n(U.ll_scaled, 3)}</Formula>
        <Note>Baum–Welch's likelihood never fell ({M.cmp.monotone ? "checked on every iteration" : "not monotone"}). The figures above are for {s === "ew" ? "the index" : "RELIANCE"}; change the series under "Choosing the number of states".</Note>
      </Card>
      <Lookahead />
      <Forecast />
      <KalmanCard />
      <Card id="kalman-forecast" title="Kalman forecast vs persistence" q="Does filtering the price improve a one-day forecast?" m={<>One-step predictions from parameters fitted on the training period, for the same {R.split.n_test} target days as Unit 2. <T k="dm">Diebold–Mariano</T> against <T k="persistence" />.</>} tags={["Kalman filter", "Forecast evaluation"]}
        meaning={<><strong>No.</strong> {K.n_sig_better} of the models beat persistence significantly. The local level ties it because, with gain {f.n(K.gain, 3)}, it has become persistence. The local linear trend's slope variance also goes to zero, leaving a constant drift. Forcing heavier smoothing looks tidier but forecasts far worse. Likelihood estimation is correctly saying that there is nothing to filter out for one-day forecasting.</>}>
        <Table caption="Kalman forecast comparison" initialSort={{ key: "rmse", dir: 1 }} highlight={(r: any) => r.key === "persistence"} cols={[
          { key: "model", label: "Model", align: "left", render: (r: any) => <><Dot color={r.key === "persistence" ? pal.model("persistence") : r.key === "ll" ? pal.s.blue : r.key === "llt" ? pal.s.orange : pal.s.aqua} />{r.model}</> }, { key: "rmse", label: "RMSE", render: (r: any) => f.n(r.rmse, 3) }, { key: "mae", label: "MAE", render: (r: any) => f.n(r.mae, 3) }, { key: "r2_ret", label: "OOS R² (b=0)", render: (r: any) => f.n(r.r2_ret, 4) }, { key: "gain", label: "Gain", render: (r: any) => f.n(r.gain, 4) }, { key: "dm_p", label: "DM p", render: (r: any) => f.p(r.dm_p) },
        ]} rows={K.forecast} />
        <Note>RMSE and MAE are in INR. The persistence RMSE of {f.n(fp.rmse, 3)} is the same as in Unit 2.</Note>
      </Card>
      <Takeaways items={[
        <>BIC selects {U.ew.K} regimes on both series. They differ mainly in volatility, and bear covers {f.pct(U.ew.checks.covid.bear_share, 0)} of March 2020, on parameters fitted on the training period only.</>,
        <>Viterbi and smoothed posteriors look ahead, so only filtered probabilities can forecast. Using smoothed ones inflates index accuracy from {f.n(U.forecast.ew.filtered.accuracy, 4)} to {f.n(U.forecast.ew.smoothed_wrong.accuracy, 4)}.</>,
        <>The HMM does not beat the majority baseline on direction, but it beats i.i.d. models on density, so risk can be forecast to a degree.</>,
        <>Scaling is required: the unscaled forward variable underflows by day {U.underflow_day}.</>,
        <>The Kalman filter's MLE gain is {f.n(K.gain, 3)}, so it reduces to persistence. A near-random-walk price has little noise to filter.</>,
      ]} />
    </>
  );
}
