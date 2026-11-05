// Plain-language meaning of each engineered feature. Column names come from the data files.
export const FEATURE_HELP: Record<string, string> = {
  r_lag0: "today's return", r_lag1: "return one day ago", r_lag2: "return two days ago", r_lag3: "return three days ago", r_lag4: "return four days ago",
  rmean_10: "mean return, last 10 days", rmean_20: "mean return, last 20 days",
  rstd_5: "return volatility, last 5 days", rstd_10: "return volatility, last 10 days", rstd_20: "return volatility, last 20 days",
  mom_60: "momentum over 60 days", dist_sma50: "distance from the 50-day average", hl_range: "today's high–low range", oc_ret: "today's open-to-close return",
  vol_ratio20: "volume against its 20-day average", dlog_volume: "daily change in log volume", mkt_r: "equal-weighted market return",
  dow_tue: "Tuesday flag", dow_wed: "Wednesday flag", dow_thu: "Thursday flag", dow_fri: "Friday flag",
};
