import { Link } from "react-router-dom";
import { useR } from "../lib/data";
import { f } from "../lib/fmt";
import { Kpi, KpiRow, PageHead, T } from "../components/ui";

export default function Overview() {
  const R = useR();
  const a = R.audit, h = R.headlines, s = R.split;
  const units = [
    { to: "/unit-1", name: "Unit 1", title: "Distributions, risk & dependence", text: <>Returns are fat-tailed: excess kurtosis is {f.n(h.u1.kurt, 2)}, and 4σ days occur {f.n(h.u1.tail, 0)} times more often than a Gaussian predicts on RELIANCE. Yesterday's direction says almost nothing about today's (Cramér's V = {f.n(h.u1.V, 4)}).</> },
    { to: "/unit-2", name: "Unit 2", title: "Regression & classification", text: <>{h.u2.n_sig_reg} of {R.u2_reg.n_models} regression models beat persistence significantly, and {h.u2.n_sig_clf} of {R.u2_clf.n_models} classifiers beat the majority rate of {f.n(h.u2.maj, 4)}.</> },
    { to: "/unit-3", name: "Unit 3", title: "Clustering & PCA", text: <>Stocks form a continuum (silhouette {f.n(h.u3.sil, 3)} at k = {h.u3.k}), and BIC finds no separate clusters. PC1 alone explains {f.pct(h.u3.pc1)} of variance and is the market factor (correlation {f.n(h.u3.pc1_corr, 3)} with the index).</> },
    { to: "/unit-4", name: "Unit 4", title: "HMM & Kalman", text: <>BIC picks {h.u4.K} regimes, which differ mostly in volatility and flag {f.pct(h.u4.covid, 0)} of March 2020 as bear. Direction forecasts stay at {f.n(h.u4.acc, 3)} against a majority rate of {f.n(h.u4.maj, 3)}. The Kalman filter's gain is {f.n(h.u4.gain, 3)}, so it becomes persistence.</> },
    { to: "/unit-5", name: "Unit 5", title: "Ensembles", text: <>{h.u5.n_sig} of {h.u5.n_models} tree, ensemble, GAM and BMA models beat the majority rate of {f.n(h.u5.maj, 4)}. The most accurate only matches it ({h.u5.best_label}: {f.n(h.u5.best_acc, 4)}).</> },
  ];
  return (
    <>
      <PageHead title="Can past prices tell us what NIFTY-50 stocks do tomorrow?">
        <p className="text-lg text-ink2 m-0">This course project studies daily data for NIFTY-50 stocks across all five Machine Learning units. It measures how returns behave, how stocks group together, what hidden regimes exist, and whether models can predict the next day. In short, <strong className="text-ink">risk can be partly predicted, but direction cannot</strong>, at least with these features and models.</p>
      </PageHead>
      <h2 className="text-2xl font-semibold mt-0 mb-3">The dataset at a glance</h2>
      <KpiRow>
        <Kpi label="Rows" value={f.i(a.rows_all)} note={`${a.cols_all} columns`} />
        <Kpi label="Tickers" value={a.tickers} note={`${a.raw_symbols} raw symbols, ${a.renamed} renames`} />
        <Kpi label="Date range" value={<span className="text-base sm:text-lg block leading-tight"><span className="whitespace-nowrap">{a.date_start} to</span> <span className="whitespace-nowrap">{a.date_end}</span></span>} />
        <Kpi label="Returns kept" value={f.i(a.n_returns)} note={`${a.n_ca} corporate-action days removed`} />
      </KpiRow>
      <p className="text-sm text-ink2 mb-8">Source: Kaggle "NIFTY-50 Stock Market Data (2000–2021)". Prediction uses RELIANCE, the most liquid stock with a full history: {f.i(s.n_train)} training days and {f.i(s.n_test)} test days, split in time order.</p>
      <h2 className="text-2xl font-semibold mb-3">The five units</h2>
      <div className="grid gap-3 md:grid-cols-2 mb-8">
        {units.map((u) => (
          <Link key={u.to} to={u.to} className="block no-underline rounded-lg border border-line bg-surface px-4 py-3 text-ink hover:border-accent">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted">{u.name}</div>
            <div className="font-serif text-lg font-semibold mb-1">{u.title}</div>
            <div className="text-sm text-ink2">{u.text}</div>
          </Link>
        ))}
        <Link to="/pipeline" className="block no-underline rounded-lg border border-line bg-surface px-4 py-3 text-ink hover:border-accent">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted">Start here</div>
          <div className="font-serif text-lg font-semibold mb-1">Data & pipeline</div>
          <div className="text-sm text-ink2">How the raw files were audited and adjusted, and why the split follows time.</div>
        </Link>
      </div>
      <h2 className="text-2xl font-semibold mb-3">How to read this site</h2>
      <ul className="pl-5 list-disc grid gap-1.5 mb-4">
        <li>Each result has four parts: <strong>Question</strong>, <strong>Method</strong> (with its syllabus topic), <strong>Result</strong> and <strong>What it means</strong>.</li>
        <li>Terms with a dotted underline, such as <T k="var" />, show a definition on hover, focus or tap. The full list is in the <Link to="/conclusions/glossary">glossary</Link>.</li>
        <li>Every number comes from the executed notebook through data files, and none is typed into the page.</li>
        <li>Charts have hover values. On a laptop you can drag to zoom and double-click to reset. A model keeps its colour across charts, and baselines are always grey.</li>
        <li>A model must beat a simple baseline to count. Where one does not, the text says so.</li>
      </ul>
    </>
  );
}
