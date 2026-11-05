import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { usePalette } from "../lib/colors";
import { Card, Controls, Formula, Kpi, KpiRow, Note, PageHead, Segmented, T } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Table } from "../components/Table";

function Timeline() {
  const R = useR(); const s = R.split; const pal = usePalette();
  const total = s.n_train + s.n_test;
  return (
    <div>
      <div className="flex w-full rounded overflow-hidden text-xs sm:text-sm" role="img" aria-label="Training period followed by test period">
        <div className="px-2 py-3 text-white font-medium min-w-0" style={{ flexGrow: s.n_train, flexBasis: 0, background: pal.mode === "dark" ? "#1c5cab" : "#256abf" }}>Train: {f.i(s.n_train)} days</div>
        <div className="px-1 sm:px-2 py-3 font-medium min-w-0 text-black" style={{ flexGrow: s.n_test, flexBasis: 0, background: pal.s.yellow }}>Test: {f.i(s.n_test)}</div>
      </div>
      <div className="flex justify-between flex-wrap text-[11px] sm:text-xs text-ink2 mt-1 gap-x-2 tnum whitespace-nowrap"><span>{s.train_start}</span><span>{s.train_end} | {s.test_start}</span><span>{s.test_end}</span></div>
      <h3 className="font-sans text-sm font-semibold mt-4 mb-1">Tuning inside the training period: <T k="tscv" /> folds</h3>
      <div className="grid gap-1">
        {s.folds.map((x: any) => (
          <div key={x.fold} className="flex items-center gap-2 text-xs">
            <span className="w-11 shrink-0 text-ink2">Fold {x.fold}</span>
            <div className="flex-1 flex h-4 rounded overflow-hidden bg-sunk min-w-0" role="img" aria-label={`Fold ${x.fold}: train on ${x.n_train} days up to ${x.train_end}, validate on ${x.n_val} days up to ${x.val_end}`}>
              <div title={`train to ${x.train_end}`} style={{ width: `${(100 * x.n_train) / s.n_train}%`, background: pal.s.blue }} />
              <div title={`validate to ${x.val_end}`} style={{ width: `${(100 * x.n_val) / s.n_train}%`, background: pal.s.aqua }} />
            </div>
          </div>
        ))}
      </div>
      <Note>Blue: fold training data. Green: fold validation data. Validation always comes after training, and the test period is never touched. Fold bars are drawn to scale over the {f.i(s.n_train)} training days, which are {f.pct(s.n_train / total, 0)} of the {f.i(total)} modelling rows.</Note>
    </div>
  );
}

export default function Pipeline() {
  const R = useR(); const a = R.audit, s = R.split; const pal = usePalette();
  const c = useChart<any>("adjust_before_after"); const leak = useChart<any>("leakage");
  const [view, setView] = useState<"full" | "event">("event");
  const auditRows: [string, any][] = [
    ["Rows × columns in the combined file", `${f.i(a.rows_all)} × ${a.cols_all}`],
    ["Per-stock files, and empty ones", `${a.n_files}, of which empty: ${a.empty_files.join(", ")}`],
    ["Tickers with data", a.tickers],
    ["Raw symbols, and companies renamed", `${a.raw_symbols}, ${a.renamed} renamed`],
    [`Missing in Trades (none from ${a.first_trades})`, f.i(a.missing_trades)],
    ["Missing in the deliverable columns", f.i(a.missing_deliverable)],
    ["Duplicate (ticker, date) pairs", a.dup_pairs],
    ["OHLC inconsistencies", a.ohlc_bad],
    ["Consecutive-day return pairs", f.i(a.n_raw_returns)],
    ["Corporate-action days removed", a.n_ca],
    ["Returns across data holes removed", `${a.n_hole} (${a.n_overlap} also flagged as corporate actions)`],
    ["Daily log-returns kept", f.i(a.n_returns)],
  ];
  const arith = `${f.i(a.n_raw_returns)} − ${a.n_ca} − ${a.n_hole} + ${a.n_overlap} = ${f.i(a.n_returns)}`;
  const data = useMemo(() => {
    if (!c) return [];
    const d = view === "full" ? c.full : c.window;
    return [
      { x: d.date, y: d.raw, type: "scatter", mode: "lines", name: "Raw Close", line: { color: pal.s.grey, width: 2 }, hovertemplate: "%{x}<br>Raw: %{y:.2f} INR<extra></extra>" },
      { x: d.date, y: d.adj, type: "scatter", mode: "lines", name: "Back-adjusted Close", line: { color: pal.s.blue, width: 2 }, hovertemplate: "%{x}<br>Adjusted: %{y:.2f} INR<extra></extra>" },
    ];
  }, [c, view, pal]);
  return (
    <>
      <PageHead title="Data & pipeline"><p className="text-ink2 m-0">Everything else depends on these steps. Each fact below was checked by an assertion in the notebook.</p></PageHead>
      <Card id="audit" title="Data audit" q="Is the raw data clean enough to use, and what has to change?" m="Load every file, compare them with the combined file, and count missing values, duplicates and price errors." tags={["Unit 1, Practice 1: load & view", "Data quality"]}
        meaning={<>The data is clean apart from two issues. Trades and the deliverable columns have gaps, so they are dropped and not imputed. Prices are not adjusted for splits, so that has to be fixed first. The file name, not the Symbol column, is used as the ticker so that a renamed company stays one company.</>}>
        <Table sortable={false} caption="Audit facts" cols={[{ key: "0", label: "Check", align: "left", wrap: true, render: (r: any) => r[0] }, { key: "1", label: "Result", wrap: true, render: (r: any) => r[1] }]} rows={auditRows} />
        <Formula>Returns kept: {arith}</Formula>
      </Card>
      <Card id="adjust" title="Corporate-action adjustment" q="A split makes a price fall by half in one day. How do we stop that looking like a crash?" m={<>Flag a day when its whole range lies outside a band of {f.n(a.rule.high, 2)}–{f.n(a.rule.low, 2)} × the previous Close, or its open is outside {f.n(a.rule.open_lo, 2)}–{f.n(a.rule.open_hi, 3)} ×. Then rebuild the price backwards.</>} tags={["Preprocessing", "Log-returns"]}
        meaning={<>A <T k="corpaction" /> is not a loss for the holder, so its return is removed and the earlier prices are rescaled. Genuine crashes trade through the band during the day, so they are kept: the largest remaining move is {f.n(a.max_abs_r, 3)} ({a.max_abs_r_row.ticker}, {a.max_abs_r_row.date}). The cost is that one real daily move is lost on each flagged day.</>}>
        <Controls><Segmented label="View" value={view} onChange={setView} options={[{ value: "event", label: `Around ${c?.window_event ?? "the event"}` }, { value: "full", label: "Full history" }]} /></Controls>
        {c ? <Chart ariaLabel="RELIANCE raw and adjusted Close" data={data} x="date" y="price (INR)" hover="x" /> : <Skeleton />}
        <Note>RELIANCE has {a.rel_ca.length} flagged days: {a.rel_ca.join(", ")}. After the last one the two lines are identical, so the adjusted price is the quoted price throughout the test period.</Note>
      </Card>
      <Card id="split" title="Chronological train/test split" q="How do we test a forecast fairly?" m={<>Train on the past and test on the future. Tune with <T k="tscv" /> on training rows only, and fit scalers on training rows only.</>} tags={["Evaluation design"]}
        meaning={<>Test days all come after training days, as they would in real use. RELIANCE has {s.n_rel_price_days_test} price days in the test period but {s.n_test} target rows, because the last day has no next-day target. The test period starts after its last corporate action ({a.rel_last_ca}).</>}>
        <Timeline />
      </Card>
      <Card id="leakage" title="Leakage test" q="Could any feature be using tomorrow's data by accident?" m={<>For {s.leak_n} random dates, rebuild the whole pipeline from data cut off at that date, and compare with the full-data features.</>} tags={["Evaluation design", "Feature engineering"]}
        meaning={<>A feature that used future data would change when the future is removed. The largest difference across {s.n_features} features is {f.sci(s.leak_max, 1)}, which is floating-point round-off, so there is no <T k="leakage" />.</>}>
        <KpiRow>
          <Kpi label="Dates tested" value={s.leak_n} />
          <Kpi label="Features per date" value={s.n_features} />
          <Kpi label="Largest difference" value={f.sci(s.leak_max, 1)} />
          <Kpi label="Back-adjust check" value={f.sci(a.adj_check, 1)} note="adjusted vs clean returns" />
        </KpiRow>
        {leak && <Chart ariaLabel="Feature difference at each truncation date" height={260} legend={false} x="truncation date" y="largest |difference| (log scale)" layout={{ yaxis: { type: "log" } }}
          data={[{ x: leak.rows.map((r: any) => r.date), y: leak.rows.map((r: any) => Math.max(r.max_abs_diff, s.leak_max / 100)), type: "scatter", mode: "markers", marker: { size: 8, color: pal.s.blue }, hovertemplate: "%{x}<br>%{y:.2e}<extra></extra>" }]} note="A difference of exactly 0 is drawn at the bottom of the axis, since a log scale cannot show 0." />}
      </Card>
      <Card id="random" title="Why a random split is wrong" q="What would go wrong if we split days at random?" m={<>Compare the polynomial trend fit under a random hold-out and a future hold-out (<Link to="/unit-1/polynomial">Unit 1</Link>).</>} tags={["Evaluation design", "Overfitting"]}
        meaning={<>A random split trains on days from the future of the test days and surrounds each test day with its neighbours, so it tests interpolation, not forecasting. It would be <T k="lookahead" /> built into the evaluation.</>}>
        <KpiRow>
          <Kpi label="Degree chosen by random hold-out" value={R.u1_poly.best_random} note={`future RMSE ${f.i(R.u1_poly.future_at_best_random)} INR`} />
          <Kpi label="Best degree for the future" value={R.u1_poly.best_future} note={`future RMSE ${f.i(R.u1_poly.future_at_best_future)} INR`} />
        </KpiRow>
      </Card>
    </>
  );
}
