import { useMemo, useState } from "react";
import { useR } from "../lib/data";
import { f } from "../lib/fmt";
import { GLOSSARY } from "../lib/glossary";
import { PageHead, T } from "../components/ui";

export default function Conclusions() {
  const R = useR(); const h = R.headlines; const [q, setQ] = useState("");
  const terms = useMemo(() => Object.values(GLOSSARY).sort((a, b) => a.term.localeCompare(b.term)).filter((g) => (g.term + " " + g.def).toLowerCase().includes(q.toLowerCase())), [q]);
  return (
    <>
      <PageHead title="Conclusions & glossary" />
      <section id="findings" className="scroll-mt-20 rounded-lg border border-line bg-surface px-4 sm:px-6 py-4 mb-8">
        <h2 className="text-2xl font-semibold mt-0 mb-2">Findings</h2>
        <p className="mt-0">Across all five units, risk can be described and partly forecast, but direction cannot. This is consistent with <T k="wfe" /> at the daily horizon, at least for these features and models.</p>
        <ul className="pl-5 list-disc grid gap-2 m-0">
          <li><strong>Unit 1.</strong> Returns have a near-zero mean and fat tails (pooled excess kurtosis {f.n(h.u1.kurt, 2)}), so Gaussian VaR understates 99% risk ({f.n(h.u1.gauss_var99, 4)} against {f.n(h.u1.hist_var99, 4)} on RELIANCE). Direction is close to independent of yesterday's (V = {f.n(h.u1.V, 4)}).</li>
          <li><strong>Unit 2.</strong> {h.u2.n_sig_reg} of {R.u2_reg.n_models} regression models beat persistence significantly (RMSE {f.n(h.u2.pers_rmse, 3)} INR), and {h.u2.n_sig_clf} of {R.u2_clf.n_models} classifiers beat the majority rate of {f.n(h.u2.maj, 4)}.</li>
          <li><strong>Unit 3.</strong> Stocks form a continuum (silhouette {f.n(h.u3.sil, 3)}), with one strong market factor (PC1 {f.pct(h.u3.pc1)}). Industries show up in correlations (ARI {f.n(h.u3.ari_corr, 3)}) more than in risk summaries (ARI {f.n(h.u3.ari, 3)}).</li>
          <li><strong>Unit 4.</strong> {h.u4.K} regimes differ mainly in volatility and catch the crises, but honest direction forecasts stay at {f.n(h.u4.acc, 3)} against {f.n(h.u4.maj, 3)}. The Kalman gain of {f.n(h.u4.gain, 3)} means that filtering reduces to persistence.</li>
          <li><strong>Unit 5.</strong> {h.u5.n_sig} of {h.u5.n_models} tree, ensemble, GAM and BMA models beat the majority rate of {f.n(h.u5.maj, 4)}.</li>
        </ul>
      </section>
      <section id="limitations" className="scroll-mt-20 rounded-lg border border-line bg-surface px-4 sm:px-6 py-4 mb-8">
        <h2 className="text-2xl font-semibold mt-0 mb-2">Limitations</h2>
        <ul className="pl-5 list-disc grid gap-2 m-0">
          <li><T k="survivorship">Survivorship bias</T>: the universe is the list of index members at the end of the sample, so companies that dropped out are missing.</li>
          <li>Corporate actions are found by a rule, not an official list, so a small action could be missed. One real daily move is lost on each flagged day.</li>
          <li>Dividends are not included, so returns are price returns only.</li>
          <li>Prediction uses one focus stock with a three-stock check, and one chronological split whose test window includes the COVID crash.</li>
          <li>Same-day observations across stocks are correlated, so pooled p-values are optimistic.</li>
          <li>Many hypothesis tests are run. Only some are corrected for multiplicity, as noted beside them.</li>
          <li>No transaction costs are modelled. They would only make any trading rule look worse.</li>
          <li>With {R.u3.n} stocks, clustering results are descriptive.</li>
          <li>These findings cover these features and daily horizons. They do not rule out signals in other data or at other horizons.</li>
        </ul>
      </section>
      <section id="glossary" className="scroll-mt-20 rounded-lg border border-line bg-surface px-4 sm:px-6 py-4 mb-8">
        <h2 className="text-2xl font-semibold mt-0 mb-2">Glossary</h2>
        <label className="block mb-3 text-sm"><span className="text-ink2">Search {terms.length} terms </span>
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} className="w-full mt-1 rounded border border-line bg-surface text-ink px-2 py-1" />
        </label>
        <dl className="m-0 grid gap-x-6 gap-y-3 md:grid-cols-2">
          {terms.map((g) => (
            <div key={g.term} className="min-w-0"><dt className="font-semibold">{g.term}</dt><dd className="m-0 text-ink2 text-sm">{g.def}</dd></div>
          ))}
        </dl>
      </section>
    </>
  );
}
