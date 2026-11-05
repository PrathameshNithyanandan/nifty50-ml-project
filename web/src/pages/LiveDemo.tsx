import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, nextWeekday, ServerDown, snapToTradingDay } from "../lib/api";
import { f } from "../lib/fmt";
import { usePalette } from "../lib/colors";
import { FEATURE_HELP } from "../lib/features";
import { Card, Controls, Formula, Kpi, KpiRow, Note, PageHead, Segmented, T } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Dot, Table } from "../components/Table";

type Tab = "replay" | "what-if" | "retrain" | "today";
const TABS: { value: Tab; label: string }[] = [{ value: "replay", label: "Replay a day" }, { value: "what-if", label: "What-if" }, { value: "retrain", label: "Retrain" }, { value: "today", label: "Today's market" }];
const DEBOUNCE_MS = 120;
const FRAME_MS = 70;
const FRAMES = 90;

const btn = "rounded-md border border-line bg-surface text-ink px-3 py-2 text-sm cursor-pointer hover:bg-sunk disabled:opacity-50 disabled:cursor-not-allowed";
const primary = "rounded-md border-0 bg-accent text-surface px-3 py-2 text-sm font-medium cursor-pointer hover:opacity-80 disabled:opacity-50 disabled:cursor-not-allowed";
const field = "rounded-md border border-line bg-surface text-ink px-2 py-1 text-sm tnum";

function Message({ kind = "info", children }: { kind?: "info" | "error"; children: ReactNode }) {
  return <div role={kind === "error" ? "alert" : "status"} className={`rounded-md px-3 py-2 text-sm my-3 border ${kind === "error" ? "border-2 border-ink bg-sunk text-ink" : "border-line bg-sunk text-ink2"}`}>{kind === "error" && <strong>Could not run this. </strong>}{children}</div>;
}

function useError() {
  const [down, setDown] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const report = (e: unknown) => { if (e instanceof ServerDown) setDown(true); else setErr(e instanceof Error ? e.message : "The request failed."); };
  return { down, err, report, clear: () => setErr(null) };
}

const ServerHelp = () => (
  <Message>
    <strong>The model server is not running, so nothing can be computed.</strong> The rest of this site still works without it. To start it, run this from the project folder, then open the address it prints:
    <Formula>.venv/bin/python -m demo.server</Formula>
  </Message>
);

/** Signed number that does not print a negative zero. */
const signed = (x: number, d: number) => (Math.abs(x) < 10 ** -d / 2 ? f.n(0, d) : f.s(x, d));

const Call = ({ up }: { up: boolean }) => <span className="font-medium">{up ? "▲ Up" : "▼ Down"}</span>;
const Verdict = ({ ok }: { ok: boolean }) => <span>{ok ? "✓ Right" : "✗ Wrong"}</span>;

function DirectionTable({ rows, caption }: { rows: any[]; caption: string }) {
  const pal = usePalette();
  const showResult = rows.some((r) => r.correct !== undefined);
  return (
    <Table caption={caption} sortable={false} highlight={(r: any) => r.group === "baseline"} rows={rows} cols={[
      { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}</> },
      { key: "call", label: "Next day", render: (r: any) => <Call up={r.call === "up"} /> },
      { key: "prob", label: "P(up)", render: (r: any) => (r.prob !== null ? f.n(r.prob, 3) : r.key === "same_as_today" ? "—" : <span className="text-muted" title="This model gives a margin, not a probability">margin</span>) },
      ...(showResult ? [{ key: "correct", label: "Result", render: (r: any) => <Verdict ok={r.correct} /> }] : []),
    ]} />
  );
}

function PriceTable({ rows, close, caption }: { rows: any[]; close: number; caption: string }) {
  const pal = usePalette();
  const showError = rows.some((r) => r.error !== undefined);
  return (
    <Table caption={caption} sortable={false} highlight={(r: any) => r.group === "baseline"} rows={rows} cols={[
      { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}</> },
      { key: "forecast", label: "Forecast (INR)", render: (r: any) => f.n(r.forecast, 2) },
      { key: "change", label: "Change", render: (r: any) => signed(100 * (r.forecast / close - 1), 2) + "%" },
      { key: "band", label: "95% band", render: (r: any) => (r.lo === null ? "—" : `${f.n(r.lo, 0)} to ${f.n(r.hi, 0)}`) },
      ...(showError ? [{ key: "error", label: "Error (INR)", render: (r: any) => f.n(r.error, 2) }] : []),
    ]} />
  );
}

function RegimeBar({ hmm }: { hmm: any }) {
  const pal = usePalette();
  return (
    <div>
      <div className="flex w-full h-3 rounded overflow-hidden gap-[2px] bg-surface" role="img" aria-label={`Filtered regime probabilities: ${hmm.names.map((n: string, i: number) => `${n} ${f.pct(hmm.state[i], 0)}`).join(", ")}`}>
        {hmm.names.map((n: string, i: number) => <div key={n} style={{ width: `${100 * hmm.state[i]}%`, background: pal.regime(n) }} />)}
      </div>
      <div className="flex flex-wrap gap-x-3 text-xs text-ink2 mt-1">
        {hmm.names.map((n: string, i: number) => <span key={n}><Dot color={pal.regime(n)} />{n} {f.pct(hmm.state[i], 0)}</span>)}
      </div>
    </div>
  );
}

/** Recent prices, the models' spread of forecasts for the next day, and the outcome once it is known. */
function ForecastChart({ d, ariaLabel }: { d: any; ariaLabel: string }) {
  const pal = usePalette();
  const data = useMemo(() => {
    const nextX = d.actual?.date ?? nextWeekday(d.date);
    const blr = d.price.find((r: any) => r.key === "blr_b_eng");
    const models = d.price.filter((r: any) => r.group !== "baseline");
    const traces: any[] = [
      { x: d.history.date, y: d.history.close, type: "scatter", mode: "lines", name: "Close", line: { color: pal.s.grey, width: 2 }, hovertemplate: "%{x}<br>%{y:.2f} INR<extra></extra>" },
    ];
    if (blr?.lo !== null && blr) traces.push({ x: [nextX, nextX], y: [blr.lo, blr.hi], type: "scatter", mode: "lines+markers", name: "Bayesian LR 95% band", line: { color: pal.model("blr_b_eng"), width: 3 }, marker: { size: 8, symbol: "line-ew", line: { width: 2, color: pal.model("blr_b_eng") } }, hovertemplate: "95% band<br>%{y:.2f} INR<extra></extra>" });
    traces.push({ x: models.map(() => nextX), y: models.map((r: any) => r.forecast), text: models.map((r: any) => r.label), type: "scatter", mode: "markers", name: "Model forecasts", marker: { size: 8, color: pal.s.blue, line: { width: 1, color: pal.surface } }, hovertemplate: "%{text}<br>%{y:.2f} INR<extra></extra>" });
    if (d.actual) traces.push({ x: [d.actual.date], y: [d.actual.close], type: "scatter", mode: "markers", name: "What happened", marker: { size: 13, symbol: "diamond", color: pal.ink, line: { width: 2, color: pal.surface } }, hovertemplate: "Actual<br>%{x}<br>%{y:.2f} INR<extra></extra>" });
    return traces;
  }, [d, pal]);
  return <Chart ariaLabel={ariaLabel} data={data} x="date" y="adjusted Close (INR)" height={340} />;
}

function DayResults({ d, label }: { d: any; label: string }) {
  return (
    <div className="grid gap-4 min-w-0">
      <ForecastChart d={d} ariaLabel={`${label}: recent prices and next-day forecasts`} />
      <div className="grid gap-4 min-w-0">
        <div className="min-w-0"><h3 className="font-sans text-sm font-semibold mt-0 mb-1">Up or down tomorrow</h3><DirectionTable rows={d.direction} caption={`${label}: direction forecasts`} /></div>
        <div className="min-w-0"><h3 className="font-sans text-sm font-semibold mt-0 mb-1">Next-day Close</h3><PriceTable rows={d.price} close={d.close} caption={`${label}: price forecasts`} /></div>
      </div>
    </div>
  );
}

// ---------- replay ----------
function PlayForward({ ticker, date, first, report, maxDays }: { ticker: string; date: string; first: string; report: (e: unknown) => void; maxDays: number }) {
  const pal = usePalette();
  const [run, setRun] = useState<any>(null);
  const [n, setN] = useState(0);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setRun(null); setPlaying(false); }, [ticker, date]);
  useEffect(() => {
    if (!playing || !run) return;
    const step = Math.max(1, Math.ceil(run.n / FRAMES));
    const t = window.setInterval(() => setN((v) => { const nx = Math.min(run.n, v + step); if (nx >= run.n) setPlaying(false); return nx; }), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, run]);
  const start = async (days: number, from = date) => {
    setBusy(true); setPlaying(false);
    try { const r = await api("replay/run", { ticker, date: from, days }); setRun(r); setN(1); setPlaying(true); }
    catch (e) { report(e); }
    finally { setBusy(false); }
  };
  const shown = ["majority", "logistic", "lda", "random_forest", "bma", "hmm"];
  const data = useMemo(() => {
    if (!run) return [];
    const x = run.dates.slice(0, n);
    return shown.map((k) => run.direction.find((r: any) => r.key === k)).filter(Boolean).map((r: any) => ({
      x, y: r.running.slice(0, n), type: "scatter", mode: "lines", name: r.label,
      line: { color: pal.model(r.key), width: r.key === "majority" ? 3 : 2, dash: r.key === "majority" ? "dash" : "solid" }, hovertemplate: "%{x}<br>" + r.label + ": %{y:.3f}<extra></extra>",
    }));
  }, [run, n, pal]);
  const done = run && n >= run.n;
  const models = run ? run.direction.filter((r: any) => r.group !== "baseline") : [];
  const nSig = models.filter((r: any) => r.p_vs_majority < .05).length;
  const best = models.length ? [...models].sort((a: any, b: any) => b.accuracy - a.accuracy)[0] : null;
  const rows = run ? run.direction.map((r: any) => ({ ...r, now: r.running[n - 1] })) : [];
  return (
    <div className="mt-5 border-t border-line pt-4 min-w-0">
      <h3 className="font-sans text-base font-semibold mt-0 mb-1">One day proves nothing: play many days</h3>
      <p className="text-sm text-ink2 mt-0 mb-2">Starting from {run ? run.dates[0] : date}, each model forecasts every following day. The dashed line is the <T k="majority" /> baseline, which any useful model has to beat.</p>
      <div className="flex flex-wrap gap-2 mb-3">
        <button type="button" className={btn} disabled={busy} onClick={() => start(50)}>Play 50 days</button>
        <button type="button" className={btn} disabled={busy} onClick={() => start(100)}>Play 100 days</button>
        <button type="button" className={btn} disabled={busy} onClick={() => start(maxDays)}>Play to the end of the data</button>
        <button type="button" className={btn} disabled={busy} onClick={() => start(maxDays, first)}>Play the whole test period</button>
        {run && !done && <button type="button" className={btn} onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Resume"}</button>}
        {run && !done && <button type="button" className={btn} onClick={() => { setPlaying(false); setN(run.n); }}>Skip to the end</button>}
      </div>
      {run && (
        <div className="grid gap-3 min-w-0">
          <KpiRow>
            <Kpi label="Days played" value={`${f.i(n)} of ${f.i(run.n)}`} />
            <Kpi label="Baseline so far" value={f.pct(rows.find((r: any) => r.key === "majority").now)} note="majority class" />
            {done && <Kpi label="Best model" value={f.pct(best.accuracy)} note={best.label} />}
            {done && <Kpi label="Beat the baseline significantly" value={`${nSig} of ${models.length}`} note="binomial test, 5% level" />}
          </KpiRow>
          <Chart ariaLabel="Running accuracy of selected models against the majority baseline" data={data} x="date" y="running accuracy (y-axis zoomed)" height={320} layout={{ yaxis: { tickformat: ".0%", range: [.3, .8] } }} note="Five models and the baseline are drawn, to keep the chart readable, and the table lists all of them. The first few days swing widely and can run off the chart." />
          {done && <Note>With this many models, one can pass the 5% level by luck, especially over a short run. Longer runs are the fairer test, and the notebook's full test period found none that beat the baseline.</Note>}
          <Table caption="Running accuracy of every direction model" rows={rows} highlight={(r: any) => r.group === "baseline"} initialSort={{ key: "now", dir: -1 }} maxHeight={320} cols={[
            { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}</> },
            { key: "now", label: "Accuracy so far", render: (r: any) => f.pct(r.now) },
            { key: "p_vs_majority", label: done ? "p vs baseline" : "p (at the end)", render: (r: any) => (done ? f.p(r.p_vs_majority) : "—") },
          ]} />
          {done && (
            <div className="min-w-0">
              <h3 className="font-sans text-sm font-semibold mt-1 mb-1">Price models over the same days</h3>
              <Table caption="Price model error over the days played" rows={run.price} highlight={(r: any) => r.group === "baseline"} initialSort={{ key: "rmse", dir: 1 }} maxHeight={320} cols={[
                { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}</> },
                { key: "rmse", label: "RMSE (INR)", render: (r: any) => f.n(r.rmse, 3) },
                { key: "dm_p", label: "DM p vs persistence", render: (r: any) => f.p(r.dm_p) },
              ]} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Replay({ ticker, stock, report, err, clear }: { ticker: string; stock: any; report: (e: unknown) => void; err: string | null; clear: () => void }) {
  const [date, setDate] = useState<string>(stock.highlights.biggest_fall);
  const [d, setD] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setDate(stock.highlights.biggest_fall); setD(null); }, [stock]);
  const pick = (v: string) => { if (!v) return; clear(); setDate(snapToTradingDay(stock.dates, v)); setD(null); };
  const move = (by: number) => { const i = stock.dates.indexOf(date) + by; if (i >= 0 && i < stock.dates.length) pick(stock.dates[i]); };
  const go = async (reveal: boolean) => {
    setBusy(true); clear();
    try { setD(await api("replay", { ticker, date, reveal })); }
    catch (e) { report(e); }
    finally { setBusy(false); }
  };
  const models = d ? d.direction.filter((r: any) => r.group !== "baseline") : [];
  const right = models.filter((r: any) => r.correct).length;
  const maj = d?.direction.find((r: any) => r.key === "majority");
  return (
    <Card id="replay" title="Replay a day" q={<>If you had been standing on any day in the test period, what would each model have said about the next day?</>}
      m={<>Pick a day. The models, fitted only on data before {stock.first}, run now on that day's features. The outcome stays hidden on the server until you press Reveal, so there is no <T k="lookahead" />.</>} tags={["Units 2, 4 and 5", "Runs live"]}
      meaning={d?.actual ? <>On this one day {right} of {models.length} models called the direction correctly, and the majority baseline was {maj.correct ? "right" : "wrong"}. A single day is an anecdote, not evidence, so play many days to see the longer record.</> : <>A confident probability is not a correct one, and the price models tend to stay close to today's price. Reveal the outcome, then play many days to see whether any of them is better than the baseline.</>}>
      <Controls>
        <label className="text-sm text-ink2 flex items-center gap-2">Day <input aria-label="Day to replay" type="date" className={field} value={date} min={stock.first} max={stock.last} onChange={(e) => pick(e.target.value)} /></label>
        <div className="flex gap-1">
          <button type="button" className={btn} aria-label="Previous trading day" onClick={() => move(-1)}>←</button>
          <button type="button" className={btn} aria-label="Next trading day" onClick={() => move(1)}>→</button>
        </div>
        <div className="flex flex-wrap gap-1">
          <button type="button" className={btn} onClick={() => pick(stock.highlights.biggest_fall)}>Day before the biggest fall</button>
          <button type="button" className={btn} onClick={() => pick(stock.highlights.biggest_rise)}>Day before the biggest rise</button>
          <button type="button" className={btn} onClick={() => pick(stock.highlights.latest)}>Latest day</button>
        </div>
      </Controls>
      <div className="flex flex-wrap gap-2 mb-3">
        <button type="button" className={primary} disabled={busy} onClick={() => go(false)}>Run the models</button>
        {d && !d.actual && <button type="button" className={primary} disabled={busy} onClick={() => go(true)}>Reveal what happened</button>}
      </div>
      {err && <Message kind="error">{err}</Message>}
      {busy && !d && <Skeleton />}
      {d && (
        <div className="grid gap-3 min-w-0">
          <KpiRow>
            <Kpi label={`Close on ${d.date}`} value={`${f.n(d.close, 2)} INR`} note="adjusted" />
            <Kpi label="Models saying up" value={`${d.n_up} of ${d.n_models}`} note={d.seconds !== undefined ? `all models ran in ${f.n(d.seconds, 2)} s` : undefined} />
            <Kpi label="HMM regime today" value={d.hmm.current} note={`P(up tomorrow) ${f.n(d.hmm.p_up, 3)}`} />
            {d.actual ? <Kpi label={`Actual on ${d.actual.date}`} value={`${f.n(d.actual.close, 2)} INR`} note={`${d.actual.direction === "up" ? "▲" : "▼"} ${f.s(100 * d.actual.r, 2)}% log return`} /> : <Kpi label="Outcome" value="Hidden" note="press Reveal" />}
          </KpiRow>
          <RegimeBar hmm={d.hmm} />
          <DayResults d={d} label={`Replay of ${d.date}`} />
          {d.failed?.length > 0 && <Note>Could not be fitted on this stock, so left out: {d.failed.join(", ")}.</Note>}
        </div>
      )}
      <PlayForward ticker={ticker} date={date} first={stock.first} report={report} maxDays={stock.dates.length} />
    </Card>
  );
}

// ---------- what-if ----------
const SCALE: Record<string, { factor: number; unit: string; digits: number }> = {
  r_lag0: { factor: 100, unit: "%", digits: 2 }, mkt_r: { factor: 100, unit: "%", digits: 2 }, oc_ret: { factor: 100, unit: "%", digits: 2 }, rstd_20: { factor: 100, unit: "%", digits: 2 },
  mom_60: { factor: 100, unit: "%", digits: 1 }, dist_sma50: { factor: 100, unit: "%", digits: 1 }, hl_range: { factor: 100, unit: "%", digits: 2 }, vol_ratio20: { factor: 1, unit: "×", digits: 2 },
};
const DAYS = [{ value: "mon", label: "Mon" }, { value: "tue", label: "Tue" }, { value: "wed", label: "Wed" }, { value: "thu", label: "Thu" }, { value: "fri", label: "Fri" }];

function WhatIf({ ticker, stock, sliders, report, err, clear }: { ticker: string; stock: any; sliders: string[]; report: (e: unknown) => void; err: string | null; clear: () => void }) {
  const pal = usePalette();
  const [date, setDate] = useState<string>(stock.highlights.latest);
  const [values, setValues] = useState<Record<string, number>>({});
  const [dow, setDow] = useState<string | null>(null);
  const [base, setBase] = useState<any>(null);
  const [d, setD] = useState<any>(null);
  const seq = useRef(0);
  useEffect(() => { setDate(stock.highlights.latest); setValues({}); setDow(null); setBase(null); setD(null); }, [stock]);
  useEffect(() => {
    const id = ++seq.current;
    const t = window.setTimeout(async () => {
      try {
        const r = await api("whatif", { ticker, date, values, ...(dow ? { dow } : {}) });
        if (id !== seq.current) return;   // a newer slider position has already been sent
        clear(); setD(r);
        if (!Object.keys(values).length && !dow) setBase(r);
      } catch (e) { if (id === seq.current) report(e); }
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [ticker, date, values, dow]);
  const bump = (k: string, v: number) => setValues((o) => ({ ...o, [k]: v }));
  const dowOf = (features: any) => (features.dow_tue ? "tue" : features.dow_wed ? "wed" : features.dow_thu ? "thu" : features.dow_fri ? "fri" : "mon");
  const probs = d ? d.direction.filter((r: any) => r.prob !== null && r.key !== "majority") : [];
  const ordered = [...probs].sort((a: any, b: any) => a.prob - b.prob);
  const lo = Math.min(.4, ...probs.map((r: any) => r.prob - .02)), hi = Math.max(.6, ...probs.map((r: any) => r.prob + .02));
  const chart = d && [
    { x: ordered.map((r: any) => r.prob), y: ordered.map((r: any) => r.label), type: "scatter", mode: "markers", marker: { size: 11, color: ordered.map((r: any) => pal.model(r.key)), line: { width: 2, color: pal.surface } }, hovertemplate: "%{y}<br>P(up) %{x:.3f}<extra></extra>" },
  ];
  const nChanged = Object.keys(values).length + (dow ? 1 : 0);
  return (
    <Card id="what-if" title="What-if" q="What does each model react to? If today had looked different, would its forecast change?" m={<>Start from a real day's features and drag them. Every model reruns on the changed inputs. The other {stock.ranges ? Object.keys(stock.ranges).length - sliders.length - 4 : ""} return and volume features stay at that day's values.</>} tags={["Units 2 and 5", "Runs live"]}
      meaning={<>Strongly regularised models barely move: logistic regression and Bayesian logistic stay near the training share of up days, because the data gave them little reason to do otherwise. Trees move in jumps, because they split on thresholds. A forecast that changes is not a forecast that is right, and the Replay tab is where accuracy gets tested.</>}>
      <Controls>
        <label className="text-sm text-ink2 flex items-center gap-2">Start from <input aria-label="Starting day" type="date" className={field} value={date} min={stock.first} max={stock.last} onChange={(e) => { if (e.target.value) { setDate(snapToTradingDay(stock.dates, e.target.value)); setValues({}); setDow(null); } }} /></label>
        <button type="button" className={btn} disabled={nChanged === 0} onClick={() => { setValues({}); setDow(null); }}>Reset to the real day</button>
        <Segmented label="Weekday" value={dow ?? (d ? dowOf(d.features) : "mon")} options={DAYS} onChange={(v) => setDow(v)} />
      </Controls>
      {err && <Message kind="error">{err}</Message>}
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 mb-4 min-w-0">
        {sliders.map((k) => {
          const r = stock.ranges[k], sc = SCALE[k], span = r.hi - r.lo;
          const real = base?.features?.[k];
          const v = values[k] ?? real ?? r.mean;
          const out = v < r.lo || v > r.hi;
          return (
            <div key={k} className="min-w-0">
              <label className="flex justify-between gap-2 text-sm"><span className="text-ink">{FEATURE_HELP[k]}</span><span className="tnum font-medium shrink-0">{f.n(v * sc.factor, sc.digits)}{sc.unit}</span>
              </label>
              <input aria-label={FEATURE_HELP[k]} type="range" className="w-full accent-[var(--accent)]" min={r.lo - span / 2} max={r.hi + span / 2} step={span / 200} value={v} onChange={(e) => bump(k, Number(e.target.value))} />
              <div className="text-xs text-muted flex justify-between gap-2 tnum">
                <span>{out ? "▲ outside what 98% of training days saw" : `training range ${f.n(r.lo * sc.factor, sc.digits)} to ${f.n(r.hi * sc.factor, sc.digits)}${sc.unit}`}</span>
                {real !== undefined && values[k] !== undefined && <span className="shrink-0">real {f.n(real * sc.factor, sc.digits)}{sc.unit}</span>}
              </div>
            </div>
          );
        })}
      </div>
      {!d ? <Skeleton /> : (
        <div className="grid gap-3 min-w-0">
          <KpiRow>
            <Kpi label="Models saying up" value={`${d.n_up} of ${d.n_models}`} note={base ? `real day: ${base.n_up} of ${base.n_models}` : undefined} />
            <Kpi label="Inputs changed" value={nChanged} />
            <Kpi label="Outside the training range" value={d.outside.length} note="features, so trust these forecasts less" />
            <Kpi label="Recomputed in" value={`${f.n(d.seconds, 2)} s`} />
          </KpiRow>
          <Chart ariaLabel="Probability of an up day from each model" data={chart} legend={false} height={Math.max(300, 30 * ordered.length)} x="P(up tomorrow): x-axis zoomed, does not run from 0 to 1" layout={{ xaxis: { range: [lo, hi] }, yaxis: { automargin: true, type: "category" }, shapes: [{ type: "line", x0: .5, x1: .5, yref: "paper", y0: 0, y1: 1, line: { color: pal.muted, dash: "dash", width: 1 } }], margin: { l: 10 } }} note="The dashed line is a coin flip. The two SVMs give a margin, not a probability, so they appear only in the table." />
          <div className="grid gap-4 min-w-0">
            <div className="min-w-0"><h3 className="font-sans text-sm font-semibold mt-0 mb-1">Up or down tomorrow</h3><DirectionTable rows={d.direction} caption="What-if direction forecasts" /></div>
            <div className="min-w-0"><h3 className="font-sans text-sm font-semibold mt-0 mb-1">Next-day Close</h3><PriceTable rows={d.price} close={d.close} caption="What-if price forecasts" /><Note>Models on the price level are left out here, because these sliders do not change the price they read.</Note></div>
          </div>
        </div>
      )}
    </Card>
  );
}

// ---------- retrain ----------
const log10 = (x: number) => Math.log(x) / Math.LN10;

function Retrain({ ticker, meta, report, err, clear }: { ticker: string; meta: any; report: (e: unknown) => void; err: string | null; clear: () => void }) {
  const pal = usePalette();
  const [cutoff, setCutoff] = useState<string>(meta.test_start);
  const [useReg, setUseReg] = useState(false);
  const [logReg, setLogReg] = useState(0);
  const [useDepth, setUseDepth] = useState(false);
  const [depth, setDepth] = useState(3);
  const [rows, setRows] = useState<any[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const stop = useRef(false);
  const all = useMemo(() => [...meta.direction, { key: "hmm", label: "HMM (filtered)" }, ...meta.price], [meta]);
  useEffect(() => () => { stop.current = true; }, []);
  useEffect(() => { stop.current = true; setRows([]); }, [ticker]);
  const train = async () => {
    stop.current = false; setRows([]); clear();
    for (const m of all) {
      if (stop.current) break;
      setRunning(m.label);
      try {
        const r = await api("retrain", { ticker, cutoff, model: m.key, reg: useReg ? 10 ** logReg : null, depth: useDepth ? depth : null });
        if (!stop.current) setRows((o) => [...o, r]);
      } catch (e) {
        report(e);
        if (e instanceof ServerDown) break;
        if (!(e instanceof Error && /could not be fitted/.test(e.message))) break;
      }
    }
    setRunning(null);
  };
  const dirRows = rows.filter((r) => r.kind === "direction"), priceRows = rows.filter((r) => r.kind === "price");
  const first = rows[0];
  const models = dirRows.filter((r) => r.group !== "baseline"), prices = priceRows.filter((r) => r.group !== "baseline");
  const nSig = models.filter((r) => r.p_vs_majority !== null && r.p_vs_majority < .05).length;
  const nBetter = prices.filter((r) => r.dm_p !== null && r.dm_p < .05 && r.rmse < r.persistence_rmse).length;
  const cart = rows.find((r) => r.key === "cart");
  const finished = !running && rows.length > 0;
  const changed = (r: any) => (r.changed ? <span className="ml-1 rounded bg-sunk px-1.5 py-0.5 text-xs text-ink2">{r.changed} changed</span> : null);
  return (
    <Card id="retrain" title="Retrain on the spot" q="Would different training data or settings have found something the notebook missed?" m={<>Move the training cutoff or change a setting, then press Train. Each model refits on the days before the cutoff and is scored on the days after it, with scalers fitted on the training rows only.</>} tags={["Units 2, 4 and 5", "Fits live"]}
      meaning={finished ? <>{nSig} of {models.length} direction models beat the baseline significantly and {nBetter} of {prices.length} price models beat <T k="persistence" /> significantly with these settings. Try a deep tree: training accuracy climbs towards 100% while test accuracy stays near a coin flip, which is overfitting you can watch. With this many models and tries, the odd significant result is expected by chance.</> : <>Press Train to fit every model. With the default cutoff and tuned settings on RELIANCE, each row should match the notebook, which is a live check that this demo runs the same models.</>}>
      <div className="grid gap-4 sm:grid-cols-3 mb-3 min-w-0">
        <label className="text-sm text-ink2 grid gap-1 content-start">Training cutoff
          <input aria-label="Training cutoff" type="date" className={field} value={cutoff} min={meta.cutoff_range[0]} max={meta.cutoff_range[1]} onChange={(e) => e.target.value && setCutoff(e.target.value)} />
          <span className="text-xs text-muted">Train before it, test after it.</span>
        </label>
        <div className="text-sm min-w-0">
          <label className="flex items-center gap-2 text-ink2"><input type="checkbox" checked={useReg} onChange={(e) => setUseReg(e.target.checked)} /> Set regularisation λ</label>
          <input aria-label="Regularisation strength, log scale" type="range" className="w-full accent-[var(--accent)] disabled:opacity-40" disabled={!useReg} min={log10(meta.reg_range[0])} max={log10(meta.reg_range[1])} step={.5} value={logReg} onChange={(e) => setLogReg(Number(e.target.value))} />
          <span className="text-xs text-muted tnum">{useReg ? `λ = ${f.g(10 ** logReg)}` : "Tuned values from the notebook."} Applies to logistic, Bayesian logistic and ridge.</span>
        </div>
        <div className="text-sm min-w-0">
          <label className="flex items-center gap-2 text-ink2"><input type="checkbox" checked={useDepth} onChange={(e) => setUseDepth(e.target.checked)} /> Set tree depth</label>
          <input aria-label="Maximum tree depth" type="range" className="w-full accent-[var(--accent)] disabled:opacity-40" disabled={!useDepth} min={meta.depth_range[0]} max={meta.depth_range[1]} step={1} value={depth} onChange={(e) => setDepth(Number(e.target.value))} />
          <span className="text-xs text-muted tnum">{useDepth ? `Depth ${depth}, no pruning.` : "Tuned values from the notebook."} Applies to CART, Random Forest and Gradient Boosting.</span>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 items-center mb-3">
        <button type="button" className={primary} disabled={!!running} onClick={train}>Train all models</button>
        {running && <button type="button" className={btn} onClick={() => { stop.current = true; }}>Stop</button>}
        <button type="button" className={btn} disabled={!!running} onClick={() => { setCutoff(meta.test_start); setUseReg(false); setUseDepth(false); }}>Back to the notebook's settings</button>
        {running && <span className="text-sm text-ink2" role="status">Fitting {running}: {rows.length} of {all.length} done</span>}
      </div>
      {err && <Message kind="error">{err}</Message>}
      {first && (
        <div className="grid gap-3 min-w-0">
          <KpiRow>
            <Kpi label="Training days" value={f.i(first.n_train)} note={`before ${first.cutoff}`} />
            <Kpi label="Test days" value={f.i(first.n_test)} />
            <Kpi label="Majority baseline" value={f.pct(first.majority_rate)} note="test accuracy" />
            <Kpi label="Persistence baseline" value={`${f.n(first.persistence_rmse, 2)} INR`} note="test RMSE" />
          </KpiRow>
          {cart?.changed && <Note>CART at this depth has {f.i(cart.leaves)} leaves: training accuracy {f.pct(cart.train_accuracy)}, test accuracy {f.pct(cart.accuracy)}.</Note>}
          <h3 className="font-sans text-sm font-semibold mt-1 mb-0">Up or down tomorrow</h3>
          <Table caption="Direction models after retraining" rows={dirRows} sortable={finished} highlight={(r: any) => r.group === "baseline"} cols={[
            { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}{changed(r)}</> },
            { key: "train_accuracy", label: "Train accuracy", render: (r: any) => f.pct(r.train_accuracy) },
            { key: "accuracy", label: "Test accuracy", render: (r: any) => f.pct(r.accuracy) },
            { key: "auc", label: <T k="auc">AUC</T>, render: (r: any) => f.n(r.auc, 3) },
            { key: "p_vs_majority", label: "p vs baseline", render: (r: any) => f.p(r.p_vs_majority) },
            { key: "seconds", label: "Fit (s)", render: (r: any) => f.n(r.seconds, 2) },
            { key: "nb", label: "Notebook", value: (r: any) => r.notebook?.accuracy, render: (r: any) => (r.notebook ? f.pct(r.notebook.accuracy) : "—") },
          ]} />
          {priceRows.length > 0 && <h3 className="font-sans text-sm font-semibold mt-1 mb-0">Next-day Close</h3>}
          {priceRows.length > 0 && <Table caption="Price models after retraining" rows={priceRows} sortable={finished} highlight={(r: any) => r.group === "baseline"} cols={[
            { key: "label", label: "Model", align: "left", render: (r: any) => <><Dot color={pal.model(r.key)} />{r.label}{changed(r)}</> },
            { key: "rmse", label: <T k="rmse">RMSE</T>, render: (r: any) => f.n(r.rmse, 3) },
            { key: "vs", label: "vs persistence", value: (r: any) => r.rmse - r.persistence_rmse, render: (r: any) => signed(r.rmse - r.persistence_rmse, 3) },
            { key: "dm_p", label: <T k="dm">DM p</T>, render: (r: any) => f.p(r.dm_p) },
            { key: "seconds", label: "Fit (s)", render: (r: any) => f.n(r.seconds, 2) },
            { key: "nb", label: "Notebook", value: (r: any) => r.notebook?.rmse, render: (r: any) => (r.notebook ? f.n(r.notebook.rmse, 3) : "—") },
          ]} />}
          <Note>Prices are in adjusted INR. The other settings stay at values tuned on RELIANCE with the notebook's cutoff, so an earlier cutoff is a fair look at the models but not a full re-tune. The Notebook column appears only for RELIANCE at the notebook's cutoff and settings.</Note>
        </div>
      )}
    </Card>
  );
}

// ---------- today ----------
function Today({ ticker, report, err, clear }: { ticker: string; report: (e: unknown) => void; err: string | null; clear: () => void }) {
  const [d, setD] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setD(null); }, [ticker]);
  const go = async (refresh: boolean) => {
    setBusy(true); clear();
    try { setD(await api(`today?ticker=${encodeURIComponent(ticker)}${refresh ? "&refresh=1" : ""}`)); }
    catch (e) { report(e); }
    finally { setBusy(false); }
  };
  return (
    <Card id="today" title="Today's market" q="What do the models say about the next trading session, using the most recent prices?" m={<>Download recent daily prices for the NIFTY-50 stocks from Yahoo Finance, build the same features, and run models fitted on the whole dataset. The dataset ends in {d?.trained_through?.slice(0, 4) ?? "2021"}, so these models have seen nothing since.</>} tags={["New data", "Needs internet"]}
      meaning={<>This shows the pipeline working on data it has never seen, and nothing more. This forecast cannot be checked until the next session closes, and the notebook found no model that beats the baseline. It is a classroom exercise, not investment advice.</>}>
      <div className="flex flex-wrap gap-2 mb-3">
        <button type="button" className={primary} disabled={busy} onClick={() => go(false)}>{busy ? "Fetching prices…" : d ? "Run again" : "Fetch prices and run the models"}</button>
        {d && <button type="button" className={btn} disabled={busy} onClick={() => go(true)}>Download fresh prices</button>}
      </div>
      {err && <Message kind="error">{err}</Message>}
      {busy && !d && <Skeleton />}
      {d && !d.available && <Message><strong>{d.ticker} has no recent prices in the feed,</strong> most likely because it has been merged, split or renamed since the dataset ended. Pick another stock. Also unavailable: {d.missing.filter((m: string) => m !== d.ticker).join(", ") || "none"}.</Message>}
      {d?.available && (
        <div className="grid gap-3 min-w-0">
          {d.source === "saved" ? <Message><strong>The price feed could not be reached, so this uses the last saved download,</strong> from {d.fetched_at}. Its latest price is from {d.last_date}.</Message> : <Note>Downloaded from Yahoo Finance at {d.fetched_at}. {d.n_market} stocks available{d.missing.length ? `; unavailable: ${d.missing.join(", ")}` : ""}. An unfinished trading day is left out.</Note>}
          <KpiRow>
            <Kpi label={`Last close, ${d.date}`} value={`${f.n(d.close, 2)} INR`} />
            <Kpi label="Models saying up" value={`${d.n_up} of ${d.n_models}`} note="for the next session" />
            <Kpi label="HMM regime now" value={d.hmm.current} note={`P(up) ${f.n(d.hmm.p_up, 3)}`} />
            <Kpi label="Models trained up to" value={d.trained_through} note={`all ran in ${f.n(d.seconds, 2)} s`} />
          </KpiRow>
          <RegimeBar hmm={d.hmm} />
          <DayResults d={d} label={`Forecast after ${d.date}`} />
          {d.outside.length > 0 && <Note>Outside the range of 98% of training days today, so these forecasts deserve less trust: {d.outside.map((k: string) => FEATURE_HELP[k]).join(", ")}.</Note>}
          <Note>Models on the price level are left out here, because prices have moved far from the range they were trained on. Yahoo's feed is unofficial, so a figure can differ slightly from the exchange's.</Note>
        </div>
      )}
    </Card>
  );
}

// ---------- page ----------
export default function LiveDemo() {
  const { section } = useParams();
  const nav = useNavigate();
  const tab: Tab = TABS.some((t) => t.value === section) ? (section as Tab) : "replay";
  const [meta, setMeta] = useState<any>(null);
  const [ticker, setTicker] = useState<string | null>(null);
  const [stock, setStock] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const { down, err, report, clear } = useError();
  useEffect(() => { api("meta").then((m) => { setMeta(m); setTicker(m.default_stock); }).catch(report); }, []);
  useEffect(() => {
    if (!ticker) return;
    let on = true; setLoading(true); setStock(null); clear();
    api(`stock?ticker=${encodeURIComponent(ticker)}`).then((s) => { if (on) setStock(s); }).catch((e) => { if (on) report(e); }).finally(() => { if (on) setLoading(false); });
    return () => { on = false; };
  }, [ticker]);
  useEffect(() => { clear(); }, [tab]);
  const shared = { report, err, clear };
  return (
    <>
      <PageHead title="Live demo"><p className="text-ink2 m-0">The other pages show results the notebook saved. Here the same models run on request on this computer, so every number below is computed as you ask for it. Every forecast appears next to its baseline, because the notebook found no model that beats one.</p></PageHead>
      {down && <ServerHelp />}
      {!down && !meta && <Skeleton h={160} />}
      {meta && !down && (
        <>
          <div className="lg:sticky lg:top-0 z-30 bg-page py-2 mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line">
            <label className="text-sm text-ink2 flex items-center gap-2">Stock
              <select aria-label="Stock" className={field} value={ticker ?? ""} onChange={(e) => setTicker(e.target.value)}>
                {meta.stocks.map((s: string) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            <Segmented label="Mode" value={tab} options={TABS} onChange={(v) => nav(`/live-demo/${v}`)} />
          </div>
          {loading && <Message>Loading the fitted models for {ticker}. If they have not been fitted yet, this takes under a minute.</Message>}
          {ticker !== "RELIANCE" && stock && <Note>{ticker} uses settings tuned on RELIANCE, as the notebook's robustness check did, and its HMM has three states. Prices are in adjusted INR, so they can differ from the prices quoted at the time if the company later split its shares.</Note>}
          <div className="mt-3">
            {stock && tab === "replay" && <Replay ticker={ticker!} stock={stock} {...shared} />}
            {stock && tab === "what-if" && <WhatIf ticker={ticker!} stock={stock} sliders={meta.sliders} {...shared} />}
            {stock && tab === "retrain" && <Retrain ticker={ticker!} meta={meta} {...shared} />}
            {stock && tab === "today" && <Today ticker={ticker!} {...shared} />}
            {!stock && !loading && err && <Message kind="error">{err}</Message>}
          </div>
        </>
      )}
    </>
  );
}
