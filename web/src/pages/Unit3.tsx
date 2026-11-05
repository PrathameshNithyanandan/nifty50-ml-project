import { useMemo, useState } from "react";
import { useChart, useR } from "../lib/data";
import { f } from "../lib/fmt";
import { SYMBOLS, usePalette } from "../lib/colors";
import { useNarrow } from "../lib/useNarrow";
import { Card, Controls, Kpi, KpiRow, Note, PageHead, Segmented, T, Takeaways } from "../components/ui";
import { Chart, Skeleton } from "../components/Chart";
import { Table } from "../components/Table";

type Method = "kmeans" | "gmm" | "hier";
const METHOD_LABEL: Record<Method, string> = { kmeans: "K-means", gmm: "GMM", hier: "Hierarchical" };

function Scatter() {
  const R = useR(); const c = useChart<any>("u3_clusters"); const pal = usePalette(); const U = R.u3;
  const [method, setMethod] = useState<Method>("kmeans");
  const [space, setSpace] = useState<"6d" | "2d">("6d");
  const [k, setK] = useState<number | null>(null);
  const kVal = k ?? (space === "6d" ? U.k_chosen : U.k_chosen_2d);
  const data = useMemo(() => {
    if (!c) return [];
    const lab: number[] = c.labels[method][space][String(kVal)];
    const groups = Array.from(new Set(lab)).sort((a, b) => a - b);
    return groups.map((g) => {
      const idx = lab.map((v, i) => (v === g ? i : -1)).filter((i) => i >= 0);
      return {
        x: idx.map((i) => c.features.ann_vol[i]), y: idx.map((i) => c.features.ann_return[i]), type: "scatter", mode: "markers", name: `Cluster ${g + 1} (${idx.length})`,
        marker: { color: pal.cluster(g), symbol: SYMBOLS[g % SYMBOLS.length], size: 11, line: { width: 1, color: pal.surface } },
        text: idx.map((i) => c.tickers[i]), customdata: idx.map((i) => [c.industry[i], c.features.beta[i], c.features.max_dd[i]]),
        hovertemplate: "<b>%{text}</b><br>%{customdata[0]}<br>volatility %{x:.3f}, return %{y:.3f}<br>beta %{customdata[1]:.2f}, max drawdown %{customdata[2]:.2f}<extra></extra>",
      };
    });
  }, [c, method, space, kVal, pal]);
  const metric = c?.metrics.find((m: any) => m.method === method && m.space === space && m.k === kVal);
  const ks: number[] = c ? Array.from(new Set<number>(c.metrics.map((m: any) => m.k))).sort((a, b) => a - b) : [];
  const S = U.cluster_summary;
  return (
    <Card id="scatter" title="Return–volatility map" q="Do stocks fall into natural groups by return and volatility?" m={<>Cluster standardised per-stock features: annualised return and volatility, <T k="beta">beta</T>, skew, kurtosis and <T k="drawdown">max drawdown</T>. The 2-D version uses only return and volatility, as the assigned title names.</>} tags={["Practice 1: K-means, GMM & hierarchical", "K-means", "Mixtures of Gaussians", "Hierarchical clustering"]}
      meaning={<>The points form a cloud, not separate islands, and moving k just reshuffles its boundaries. At k = {U.k_chosen} on six features, K-means gives a lower-beta group (β = {f.n(S[0].beta, 2)}, n = {S[0].n}), a fat-tailed group (excess kurtosis {f.n(S[1].ex_kurt, 1)}, n = {S[1].n}) and a high-volatility, high-beta group (volatility {f.n(S[2].ann_vol, 2)}, β = {f.n(S[2].beta, 2)}, n = {S[2].n}). With only {U.n} stocks, this is description and not proof of real groups.</>}>
      <Controls>
        <Segmented label="Method" value={method} onChange={setMethod} options={(["kmeans", "gmm", "hier"] as Method[]).map((m) => ({ value: m, label: METHOD_LABEL[m] }))} />
        <Segmented label="Features" value={space} onChange={(v) => { setSpace(v); setK(null); }} options={[{ value: "6d", label: "Six features" }, { value: "2d", label: "Return & volatility" }]} />
      </Controls>
      <label className="block mb-3 text-sm"><span className="text-ink2">Clusters k: </span><strong className="tnum">{kVal}</strong>
        {ks.length > 0 && <input type="range" min={ks[0]} max={ks[ks.length - 1]} step={1} value={kVal} onChange={(e) => setK(Number(e.target.value))} aria-label="Number of clusters" />}
      </label>
      {metric && <KpiRow>
        <Kpi label="Silhouette" value={f.n(metric.silhouette, 3)} note="higher is better" />
        <Kpi label="Davies–Bouldin" value={f.n(metric.db, 3)} note="lower is better" />
        <Kpi label="ARI with industry" value={f.n(metric.ari_industry, 3)} note="0 = chance" />
        <Kpi label="Smallest cluster" value={metric.min_size} note="stocks" />
      </KpiRow>}
      {c ? <Chart ariaLabel="Stocks by annualised volatility and return, coloured by cluster" height={460} heightNarrow={420} x="annualised volatility" y="annualised mean log-return" data={data} /> : <Skeleton h={460} />}
      <Note>Cluster numbers are sorted by mean volatility, so a cluster keeps its colour as far as possible when settings change. Each stock uses its own available history, so newer listings cover a different period.</Note>
    </Card>
  );
}

function KmeansCard() {
  const R = useR(); const U = R.u3; const c = useChart<any>("u3_clusters"); const pal = usePalette();
  const [sp, setSp] = useState<"6d" | "2d">("6d");
  const rows = c ? c.kmeans.filter((r: any) => r.space === sp) : [];
  const line = (y: string, name: string, color: string) => ({ x: rows.map((r: any) => r.k), y: rows.map((r: any) => r[y]), type: "scatter", mode: "lines+markers", name, line: { color, width: 2 }, marker: { size: 8 }, hovertemplate: `k = %{x}<br>${name}: %{y:.3f}<extra></extra>` });
  const drop = U.inertia_drop;
  return (
    <Card id="kmeans" title="Choosing k: elbow, silhouette, Davies–Bouldin" q="How many clusters does the data support?" m={<>Fit <T k="kmeans" /> for each k with many restarts. Read the elbow, <T k="silhouette" /> and <T k="db">DB index</T>, and require at least three stocks in a cluster.</>} tags={["K-means", "Cluster evaluation"]}
      meaning={<>The silhouette peaks at k = {U.k_chosen} on six features ({f.n(U.sil_chosen, 3)}) and k = {U.k_chosen_2d} on two ({f.n(U.sil_chosen_2d, 3)}). There is no sharp elbow: each extra cluster removes {f.pct(drop["3"], 0)}, {f.pct(drop["4"], 0)}, {f.pct(drop["5"], 0)}, then {f.pct(drop["6"], 0)} of the inertia at the smallest k. DB would rather have more clusters. We keep k = {U.k_chosen} because the silhouette agrees and the groups can be read. Silhouettes this low mean the clusters overlap.</>}>
      <Controls><Segmented label="Features" value={sp} onChange={setSp} options={[{ value: "6d", label: "Six features" }, { value: "2d", label: "Return & volatility" }]} /></Controls>
      {c ? <div className="grid gap-4 md:grid-cols-3 min-w-0">
        <Chart ariaLabel="Inertia against k" legend={false} height={280} x="k" y="inertia" hover="x" layout={{ xaxis: { dtick: 1 } }} data={[line("inertia", "Inertia", pal.s.blue)]} />
        <Chart ariaLabel="Silhouette against k" legend={false} height={280} x="k" y="silhouette" hover="x" layout={{ xaxis: { dtick: 1 } }} data={[line("silhouette", "Silhouette", pal.s.blue)]} />
        <Chart ariaLabel="Davies-Bouldin index against k" legend={false} height={280} x="k" y="Davies–Bouldin" hover="x" layout={{ xaxis: { dtick: 1 } }} data={[line("db", "Davies–Bouldin", pal.s.blue)]} />
      </div> : <Skeleton h={280} />}
    </Card>
  );
}

function GmmCard() {
  const R = useR(); const G = R.u3.gmm; const H = R.u3.hard_limit; const c = useChart<any>("u3_gmm"); const pal = usePalette();
  const types = ["spherical", "diag", "tied", "full"]; const hue: Record<string, string> = { spherical: pal.s.blue, diag: pal.s.orange, tied: pal.s.aqua, full: pal.s.violet };
  const last = H[H.length - 1], first = H[0];
  return (
    <Card id="gmm" title="Gaussian mixtures and the EM algorithm" q="Is there evidence for separate clusters, and what does EM actually do?" m={<>Choose components and covariance type by <T k="bic">BIC</T> and <T k="aic">AIC</T>. Write <T k="em">EM</T> from scratch, check it against scikit-learn, and follow the log-likelihood.</>} tags={["Mixtures of Gaussians", "EM (alternative view)"]}
      meaning={<>BIC picks K = {G.bic_k}: a single Gaussian is enough, so <strong>BIC finds no evidence of separate clusters</strong> among {R.u3.n} points. AIC picks K = {G.aic_k} ({G.aic_cov}), with {G.aic_params} parameters, which is overfitting its weaker penalty does not stop. From-scratch EM matches scikit-learn to {f.sci(G.sk_diff, 1)}, and its log-likelihood never falls.</>}>
      <div className="grid gap-3 md:grid-cols-2 mb-3">
        <div className="rounded border border-line p-3 text-sm"><strong>The alternative view of EM.</strong> With latent assignments Z and any q(Z): ln p(X|θ) = L(q, θ) + KL(q ‖ p(Z|X, θ)). KL ≥ 0, so L is a lower bound. The E-step sets q to the <T k="resp">responsibilities</T>, which makes KL = 0. The M-step raises L over θ. So ln p(X|θ) can never fall. Here the gap after each E-step is at most {f.sci(G.em_gap, 1)}.</div>
        <div className="rounded border border-line p-3 text-sm"><strong>K-means as the hard limit.</strong> Fix Σ = εI and equal weights. As ε → 0, each responsibility goes to 0 or 1, and the M-step mean is the K-means centroid. Here the mean largest responsibility rises from {f.n(first.mean_max_resp, 3)} (ε = {first.eps}) to {f.n(last.mean_max_resp, 3)} (ε = {f.g(last.eps)}), and agreement with K-means reaches ARI = {f.n(last.ari_vs_kmeans, 3)}.</div>
      </div>
      {c ? <div className="grid gap-4 lg:grid-cols-2 min-w-0">
        <Chart ariaLabel="BIC by number of components and covariance type" height={320} x="components" y="BIC (lower is better)" hover="x" layout={{ xaxis: { dtick: 1 } }} data={types.map((t) => {
          const r = c.rows.filter((x: any) => x.space === "6d" && x.cov === t);
          return { x: r.map((x: any) => x.k), y: r.map((x: any) => x.bic), type: "scatter", mode: "lines+markers", name: t, line: { color: hue[t], width: 2 }, marker: { size: 7 }, hovertemplate: `${t}: %{y:.1f}<extra></extra>` };
        })} />
        <Chart ariaLabel="Log-likelihood by EM iteration" height={320} legend={false} x="EM iteration" y="log-likelihood" hover="x" data={[{ x: c.em_ll.map((_: number, i: number) => i), y: c.em_ll, type: "scatter", mode: "lines", line: { color: pal.s.blue, width: 2 }, hovertemplate: "iteration %{x}<br>log-likelihood %{y:.4f}<extra></extra>" }]} note={`K = ${G.em_k}, ${G.em_cov} covariance, ${G.em_iters} iterations: K ≥ 2 is used so that EM has something to demonstrate.`} />
      </div> : <Skeleton />}
    </Card>
  );
}

function HierCard() {
  const R = useR(); const c = useChart<any>("u3_dendrogram"); const pal = usePalette(); const narrow = useNarrow();
  const [dist, setDist] = useState<"euclid" | "corr">("euclid");
  const [link, setLink] = useState<"ward" | "complete" | "average">("ward");
  const d = c?.[`${dist}_${link}`];
  const rows = R.u3.hier; const best = [...rows].sort((a: any, b: any) => b.cophenetic - a.cophenetic)[0];
  const data = useMemo(() => {
    if (!d) return [];
    const xs: (number | null)[] = [], ys: (number | null)[] = [];
    d.icoord.forEach((ic: number[], i: number) => { xs.push(...d.dcoord[i], null); ys.push(...ic, null); });
    return [{ x: xs, y: ys, type: "scatter", mode: "lines", line: { color: pal.s.blue, width: 1.5 }, hoverinfo: "skip" }];
  }, [d, pal]);
  return (
    <Card id="hierarchical" title="Hierarchical clustering and dissimilarity" q="What does the merge tree look like, and does the choice of distance matter?" m={<>Agglomerative clustering with ward, complete and average <T k="linkage" />, using Euclidean distance on features or correlation distance √(2(1−ρ)). The <T k="cophenetic">cophenetic correlation</T> scores each tree.</>} tags={["Hierarchical clustering", "Dissimilarity measures"]}
      meaning={<>The two distances answer different questions. Features group stocks with similar risk summaries, while correlation distance groups stocks that move together on the same days. They disagree a lot (ARI = {f.n(R.u3.ari_eu_vs_corr, 3)}). The tree that best keeps its original distances is {best.linkage} linkage on {best.distance} distance (cophenetic {f.n(best.cophenetic, 3)}).</>}>
      <Controls>
        <Segmented label="Distance" value={dist} onChange={setDist} options={[{ value: "euclid", label: "Euclidean" }, { value: "corr", label: "Correlation" }]} />
        <Segmented label="Linkage" value={link} onChange={setLink} options={[{ value: "ward", label: "Ward" }, { value: "complete", label: "Complete" }, { value: "average", label: "Average" }]} />
      </Controls>
      <Note>Cophenetic correlation for this tree: <strong className="tnum">{f.n(rows.find((r: any) => r.distance === (dist === "euclid" ? "euclidean" : "correlation") && r.linkage === link)?.cophenetic, 3)}</strong></Note>
      {d ? <Chart ariaLabel="Dendrogram" height={840} heightNarrow={820} legend={false} x="merge distance" layout={{ hovermode: false, dragmode: false, margin: { l: narrow ? 84 : 100, t: 8 }, yaxis: { tickvals: d.ivl.map((_: string, i: number) => 5 + 10 * i), ticktext: d.ivl, tickfont: { size: 11 }, showgrid: false, range: [0, 10 * d.ivl.length], title: undefined } }} data={data} /> : <Skeleton h={500} />}
    </Card>
  );
}

function IndustryCard() {
  const R = useR(); const U = R.u3; const c = useChart<any>("u3_contingency");
  const [v, setV] = useState<"features" | "correlation">("correlation");
  const t = c?.[v];
  const max = t ? Math.max(...t.counts.flat()) : 1;
  const A = U.ari;
  return (
    <Card id="industry" title="Clusters against industry" q="Do data-driven clusters match the official industry labels?" m={<>Cross-tabulate clusters with Industry and compute the <T k="ari">adjusted Rand index</T>, with a permutation test for its null range. Industry is not used in the fitting.</>} tags={["Cluster evaluation"]}
      meaning={<>On risk–return features, agreement is weak (ARI = {f.n(A.km_6d, 3)}, permutation p = {f.p(U.ari_p)}). On correlation distance it is clearer (ARI = {f.n(A.corr_ward, 3)}, p = {f.p(U.ari_corr_p)}): IT ({U.corr_members["1"].length} stocks) and consumer and pharma defensives ({U.corr_members["2"].length}) split off, leaving {U.corr_members["3"].length} in one large group. Sectors do move together, though silhouettes near zero ({f.n(U.sil_corr_ward, 3)}) show the structure is weak. With n = {U.n}, this is descriptive.</>}>
      <Controls><Segmented label="Clustering" value={v} onChange={setV} options={[{ value: "correlation", label: "Correlation distance (ward)" }, { value: "features", label: "Six features (K-means)" }]} /></Controls>
      {t ? <div className="overflow-x-auto rounded border border-line" tabIndex={0} role="region" aria-label="Cluster by industry counts">
        <table className="w-full border-collapse text-sm tnum">
          <thead className="bg-sunk"><tr><th className="px-2.5 py-1.5 text-left">Industry</th>{t.clusters.map((k: number) => <th key={k} className="px-2.5 py-1.5 text-right whitespace-nowrap">Cluster {v === "features" ? k + 1 : k}</th>)}</tr></thead>
          <tbody>{t.industries.map((ind: string, i: number) => (
            <tr key={ind} className="border-t border-line"><td className="px-2.5 py-1 whitespace-nowrap">{ind}</td>
              {t.counts[i].map((n: number, j: number) => <td key={j} className="px-2.5 py-1 text-right" style={{ background: n ? `color-mix(in srgb, var(--accent) ${Math.round((n / max) * 45)}%, transparent)` : undefined }}>{n || "·"}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div> : <Skeleton h={300} />}
      <div className="mt-3"><Table sortable={false} caption="ARI with industry" cols={[{ key: "n", label: "Clustering", align: "left" }, { key: "v", label: "ARI with industry", render: (r: any) => f.n(r.v, 3) }]} rows={[
        { n: "K-means, six features", v: A.km_6d }, { n: "GMM, six features", v: A.gmm_6d }, { n: "Hierarchical, six features", v: A.hier_6d }, { n: "K-means, return & volatility", v: A.km_2d }, { n: "Correlation distance, ward", v: A.corr_ward }, { n: "Correlation distance, average", v: A.corr_average },
      ]} /></div>
      <Note>Under random labels, ARI averages {f.n(U.ari_null_mean, 4)} with a 95th percentile of {f.n(U.ari_null_95, 3)}.</Note>
    </Card>
  );
}

function Scree() {
  const R = useR(); const P = R.u3_pca; const c = useChart<any>("u3_pca"); const pal = usePalette();
  const xs = c?.eigen.map((_: number, i: number) => i + 1);
  return (
    <Card id="scree" title="PCA: how many latent dimensions?" q="How many hidden factors are needed to describe returns on all stocks?" m={<>Run <T k="pca" /> on the standardised stocks × days matrix from {P.start} to {P.end}. Compare ≥ 80% variance, <T k="kaiser">Kaiser</T>, <T k="parallel">parallel analysis</T> and <T k="brokenstick">broken stick</T>.</>} tags={["Practice 2: PCA", "Choosing latent dimensions"]}
      meaning={<>The rules disagree because they ask different things. {P.n80} components are needed for ≥ 80% of variance, since each stock has much noise of its own. Parallel analysis finds {P.n_parallel} components above chance, and broken stick keeps {P.n_broken}. We use {P.n_parallel} as the number of real shared factors. The single big step after PC1 is the main result.</>}>
      <KpiRow>
        <Kpi label="≥ 80% variance" value={P.n80} note={`${f.pct(P.cum_at_80)} explained`} />
        <Kpi label="Kaiser" value={P.n_kaiser} note={`${f.pct(P.cum_at_kaiser)} explained`} />
        <Kpi label="Parallel analysis" value={P.n_parallel} note={`${f.pct(P.cum_at_parallel)} explained`} />
        <Kpi label="Broken stick" value={P.n_broken} note={`PC1 alone: ${f.pct(P.pc1)}`} />
      </KpiRow>
      {c ? <Chart ariaLabel="Scree plot with component selection rules" height={380} x={`component (first ${c.n_show} of ${P.stocks})`} y="eigenvalue (log scale)" hover="x" layout={{ yaxis: { type: "log" }, xaxis: { dtick: 1 } }} data={[
        { x: xs, y: c.eigen, type: "scatter", mode: "lines+markers", name: "Eigenvalue", line: { color: pal.s.blue, width: 2 }, marker: { size: 8 }, customdata: c.cum, hovertemplate: "PC%{x}: eigenvalue %{y:.3f}<br>cumulative %{customdata:.3f}<extra></extra>" },
        { x: xs, y: c.parallel, type: "scatter", mode: "lines", name: "Parallel analysis (95%)", line: { color: pal.s.orange, width: 2, dash: "dash" }, hovertemplate: "parallel %{y:.3f}<extra></extra>" },
        { x: xs, y: c.broken, type: "scatter", mode: "lines", name: "Broken stick", line: { color: pal.s.aqua, width: 2, dash: "dot" }, hovertemplate: "broken stick %{y:.3f}<extra></extra>" },
        { x: [xs[0], xs[xs.length - 1]], y: [1, 1], type: "scatter", mode: "lines", name: "Kaiser (eigenvalue 1)", line: { color: pal.muted, width: 1 }, hoverinfo: "skip" },
      ]} /> : <Skeleton />}
      <Note>Window: {P.days_in_range} days in range, {P.days} after dropping days with any missing return. {P.stocks} of {R.u3.n} stocks kept. It starts when the latest listing ({P.latest_listing}) begins. Parallel analysis uses {P.n_shuffles} shuffles.</Note>
    </Card>
  );
}

function Pc1() {
  const R = useR(); const P = R.u3_pca; const c = useChart<any>("u3_pca"); const s = useChart<any>("u3_pc1_series"); const pal = usePalette();
  const order = c ? c.tickers.map((_: string, i: number) => i).sort((a: number, b: number) => c.pc1[a] - c.pc1[b]) : [];
  return (
    <Card id="pc1" title="PC1 is the market factor" q="What is the largest shared source of movement?" m={<>Check the signs of PC1's <T k="loading">loadings</T>, and correlate its score with an equal-weighted index.</>} tags={["PCA"]}
      meaning={<>All {P.stocks} loadings have the same sign, from {f.n(P.pc1_min, 3)} ({P.pc1_min_t}) to {f.n(P.pc1_max, 3)} ({P.pc1_max_t}), and the score correlates {f.n(P.pc1_corr, 4)} with the index. So PC1 is the market as a whole. It also explains why every pair of stocks in Unit 1 has a positive correlation.</>}>
      {c && s ? <div className="grid gap-4 lg:grid-cols-[3fr_2fr] min-w-0 items-start">
        <Chart ariaLabel="PC1 loadings by stock" height={780} heightNarrow={780} legend={false} x="PC1 loading" layout={{ yaxis: { dtick: 1, tickfont: { size: 10 } }, margin: { l: 90 } }} data={[{ type: "bar", orientation: "h", y: order.map((i: number) => c.tickers[i]), x: order.map((i: number) => c.pc1[i]), marker: { color: pal.s.blue }, customdata: order.map((i: number) => c.industry[i]), hovertemplate: "%{y}<br>%{customdata}<br>loading %{x:.3f}<extra></extra>" }]} />
        <Chart ariaLabel="PC1 score against equal-weighted index return" height={360} heightNarrow={320} legend={false} x="equal-weighted return" y="PC1 score" data={[{ x: s.ew, y: s.pc1, type: "scatter", mode: "markers", marker: { color: pal.s.blue, size: 4, opacity: 0.5 }, hovertemplate: "%{text}<br>index %{x:.4f}<br>PC1 %{y:.2f}<extra></extra>", text: s.date }]} note="A subset of days that keeps the extremes." />
      </div> : <Skeleton h={400} />}
    </Card>
  );
}

function Pc23() {
  const R = useR(); const P = R.u3_pca; const pal = usePalette();
  const [pc, setPc] = useState<"PC2" | "PC3">("PC2");
  const rows = [...P.pc_industry].sort((a: any, b: any) => a[pc] - b[pc]);
  const it = P.pc_industry.find((r: any) => r.industry === "IT");
  const en = P.pc_industry.find((r: any) => r.industry === "ENERGY"), me = P.pc_industry.find((r: any) => r.industry === "METALS"), fi = P.pc_industry.find((r: any) => r.industry === "FINANCIAL SERVICES"), co = P.pc_industry.find((r: any) => r.industry === "CONSUMER GOODS");
  const top = pc === "PC2" ? P.pc2_top : P.pc3_top, bottom = pc === "PC2" ? P.pc2_bottom : P.pc3_bottom;
  return (
    <Card id="pc23" title="PC2 and PC3 by industry" q="After removing the market, what structure is left?" m="Average the loadings within each industry, and list the stocks at each end." tags={["PCA"]}
      meaning={<>PC2 ({f.pct(P.pc2)} of variance) separates IT (mean loading {f.n(it.PC2, 2)}) from the rest. PC3 ({f.pct(P.pc3)}) separates energy and metals ({f.n(en.PC3, 2)}, {f.n(me.PC3, 2)}) from financials and consumer goods ({f.n(fi.PC3, 2)}, {f.n(co.PC3, 2)}). Among industries with several stocks, industry explains {f.pct(P.eta2.PC2, 0)} and {f.pct(P.eta2.PC3, 0)} of the loading variation.</>}>
      <Controls><Segmented label="Component" value={pc} onChange={setPc} options={[{ value: "PC2", label: "PC2" }, { value: "PC3", label: "PC3" }]} /></Controls>
      <Chart ariaLabel={`Mean ${pc} loading by industry`} height={420} heightNarrow={420} legend={false} x={`mean ${pc} loading`} layout={{ yaxis: { dtick: 1, tickfont: { size: 10 } }, margin: { l: 150 } }} data={[{ type: "bar", orientation: "h", y: rows.map((r: any) => r.industry), x: rows.map((r: any) => r[pc]), marker: { color: pal.s.blue }, customdata: rows.map((r: any) => r.n), hovertemplate: "%{y}<br>mean loading %{x:.3f}<br>%{customdata} stock(s)<extra></extra>" }]} />
      <Note>Highest: {Object.keys(top).join(", ")}. Lowest: {Object.keys(bottom).join(", ")}.</Note>
    </Card>
  );
}

function FaCard() {
  const R = useR(); const F = R.u3_pca.fa; const c = useChart<any>("u3_pca"); const pal = usePalette();
  const order = c ? c.tickers.map((_: string, i: number) => i).sort((a: number, b: number) => c.fa_psi[a] - c.fa_psi[b]) : [];
  return (
    <Card id="fa" title="PCA vs factor analysis" q="Does giving each stock its own noise level change the picture?" m={<><T k="fa">Factor analysis</T> with {F.k} factors allows a separate noise variance ψ for each stock, while <T k="ppca">probabilistic PCA</T> forces one σ². Compare loadings and noise.</>} tags={["Factor analysis", "PCA"]}
      meaning={<>The leading factors agree (cosines {f.n(F.cos[0], 3)} and {f.n(F.cos[1], 3)}), while later columns differ because factor loadings are only defined up to a rotation. The main difference is noise: ψ runs from {f.n(F.psi_min, 2)} ({F.psi_min_t}) to {f.n(F.psi_max, 2)} ({F.psi_max_t}), where PPCA forces {f.n(F.sigma2_ppca, 2)} on every stock. BIC prefers factor analysis ({f.i(F.bic_fa)} vs {f.i(F.bic_ppca)}), because stocks clearly differ in how much of their movement is their own.</>}>
      <KpiRow>
        <Kpi label="Loading cosines" value={<span className="text-base">{F.cos.map((x: number) => f.n(x, 2)).join(", ")}</span>} note="by column, sign-free" />
        <Kpi label="Subspace angles" value={<span className="text-base">{f.n(Math.min(...F.angles), 1)}° to {f.n(Math.max(...F.angles), 1)}°</span>} />
        <Kpi label="Log-likelihood FA / PPCA" value={<span className="text-base">{f.i(F.ll_fa)} / {f.i(F.ll_ppca)}</span>} />
        <Kpi label="Corr(ψ, PCA residual)" value={f.n(F.psi_corr, 3)} />
      </KpiRow>
      {c ? <Chart ariaLabel="Factor analysis noise variance by stock with the single PPCA value" height={780} heightNarrow={780} x="noise variance" layout={{ yaxis: { dtick: 1, tickfont: { size: 10 } }, margin: { l: 90 } }} data={[
        { type: "bar", orientation: "h", y: order.map((i: number) => c.tickers[i]), x: order.map((i: number) => c.fa_psi[i]), name: "Factor analysis ψ", marker: { color: pal.s.blue }, hovertemplate: "%{y}<br>ψ = %{x:.3f}<extra></extra>" },
        { type: "scatter", mode: "lines", y: [c.tickers[order[0]], c.tickers[order[order.length - 1]]], x: [c.sigma2, c.sigma2], name: "PPCA σ²", line: { color: pal.s.orange, width: 2, dash: "dash" }, hovertemplate: "PPCA σ² = %{x:.3f}<extra></extra>" },
      ]} /> : <Skeleton />}
    </Card>
  );
}

export default function Unit3() {
  const R = useR(); const U = R.u3, P = R.u3_pca;
  return (
    <>
      <PageHead title="Unit 3: Clustering & PCA" assigned="Clustering NIFTY-50 stocks by return–volatility behaviour using K-means, GMM & hierarchical clustering; PCA market factors"
        mapping="K-means, mixtures of Gaussians, EM (alternative view), factor analysis, PCA & choosing latent dimensions, dissimilarity measures & cluster evaluation (silhouette, DB index), hierarchical clustering → stocks by return & volatility."
        practice="1: K-means, GMM & hierarchical clustering. 2: PCA." />
      <Scatter />
      <KmeansCard />
      <GmmCard />
      <HierCard />
      <IndustryCard />
      <Scree />
      <Pc1 />
      <Pc23 />
      <FaCard />
      <Takeaways items={[
        <>Stocks form a continuum, not separate groups: silhouette {f.n(U.sil_chosen, 3)} at k = {U.k_chosen}, and BIC prefers one Gaussian.</>,
        <>Risk–return clusters barely match industries (ARI {f.n(U.ari.km_6d, 3)}), but correlation-based clusters do better (ARI {f.n(U.ari.corr_ward, 3)}), with IT and defensives separating.</>,
        <>One factor dominates: PC1 explains {f.pct(P.pc1)} of variance and correlates {f.n(P.pc1_corr, 3)} with the index.</>,
        <>From-scratch EM matches scikit-learn, and K-means appears as its hard-assignment limit.</>,
        <>With {U.n} stocks, and only the current index members, all clustering results are descriptive.</>,
      ]} />
    </>
  );
}
