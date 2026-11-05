export type Section = { id: string; title: string };
export type Page = { path: string; title: string; short: string; sections: Section[] };

export const PAGES: Page[] = [
  { path: "/", title: "Overview", short: "Overview", sections: [] },
  { path: "/pipeline", title: "Data & pipeline", short: "Data & pipeline", sections: [
    { id: "audit", title: "Data audit" }, { id: "adjust", title: "Corporate actions" }, { id: "split", title: "Chronological split" }, { id: "leakage", title: "Leakage test" }, { id: "random", title: "Why not a random split" },
  ] },
  { path: "/unit-1", title: "Unit 1: Distributions, risk & dependence", short: "Unit 1: Distributions & risk", sections: [
    { id: "views", title: "Two views of the data" }, { id: "summary", title: "Summary statistics" }, { id: "distributions", title: "Return distributions" }, { id: "var", title: "VaR and Expected Shortfall" }, { id: "bayes", title: "Bayes rule" }, { id: "independence", title: "Independence" }, { id: "conditional", title: "Conditional independence" }, { id: "expectation", title: "Expectation" }, { id: "covariance", title: "Covariance & correlation" }, { id: "polynomial", title: "Polynomial curve fit" }, { id: "takeaways", title: "Key takeaways" },
  ] },
  { path: "/unit-2", title: "Unit 2: Regression & classification", short: "Unit 2: Regression & classes", sections: [
    { id: "reg-table", title: "Next-day Close models" }, { id: "mle", title: "OLS equals MLE" }, { id: "ridge", title: "Huber and ridge" }, { id: "blr", title: "Bayesian linear regression" }, { id: "band", title: "Predictive band" }, { id: "clf-table", title: "Up/down models" }, { id: "roc", title: "ROC curves" }, { id: "laplace", title: "Laplace approximation" }, { id: "kernel", title: "Kernel trick" }, { id: "robust", title: "Robustness" }, { id: "takeaways", title: "Key takeaways" },
  ] },
  { path: "/unit-3", title: "Unit 3: Clustering & PCA", short: "Unit 3: Clustering & PCA", sections: [
    { id: "scatter", title: "Return–volatility map" }, { id: "kmeans", title: "Choosing k" }, { id: "gmm", title: "GMM and EM" }, { id: "hierarchical", title: "Hierarchical clustering" }, { id: "industry", title: "Clusters vs industry" }, { id: "scree", title: "PCA: how many dimensions" }, { id: "pc1", title: "PC1: the market factor" }, { id: "pc23", title: "PC2 and PC3" }, { id: "fa", title: "PCA vs factor analysis" }, { id: "takeaways", title: "Key takeaways" },
  ] },
  { path: "/unit-4", title: "Unit 4: HMM & Kalman", short: "Unit 4: HMM & Kalman", sections: [
    { id: "regimes", title: "Hidden regimes" }, { id: "states", title: "States and transitions" }, { id: "selection", title: "Choosing K" }, { id: "match", title: "From-scratch checks" }, { id: "lookahead", title: "Filtered vs smoothed" }, { id: "forecast", title: "HMM forecast" }, { id: "kalman", title: "Kalman trend" }, { id: "kalman-forecast", title: "Kalman forecast" }, { id: "takeaways", title: "Key takeaways" },
  ] },
  { path: "/unit-5", title: "Unit 5: Ensembles", short: "Unit 5: Ensembles", sections: [
    { id: "leaderboard", title: "Leaderboard" }, { id: "cart", title: "Pruned CART" }, { id: "ensembles", title: "Ensembles and tuning" }, { id: "basis", title: "Adaptive basis functions" }, { id: "importance", title: "Feature importance" }, { id: "gam", title: "GAM" }, { id: "bma", title: "Bayesian model averaging" }, { id: "robust", title: "Robustness" }, { id: "takeaways", title: "Key takeaways" },
  ] },
  { path: "/live-demo", title: "Live demo", short: "Live demo", sections: [
    { id: "replay", title: "Replay a day" }, { id: "what-if", title: "What-if" }, { id: "retrain", title: "Retrain" }, { id: "today", title: "Today's market" },
  ] },
  { path: "/conclusions", title: "Conclusions & glossary", short: "Conclusions & glossary", sections: [
    { id: "findings", title: "Findings" }, { id: "limitations", title: "Limitations" }, { id: "glossary", title: "Glossary" },
  ] },
];
