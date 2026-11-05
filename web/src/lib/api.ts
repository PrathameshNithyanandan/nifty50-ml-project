// Calls to the local model server. Unlike the rest of the site, these numbers are computed on request.
export class ServerDown extends Error {}

export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`./api/${path}`, body === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new ServerDown("The model server is not running.");
  }
  // a static host answers with its own page instead of JSON, which also means there is no model server
  if (!(res.headers.get("content-type") ?? "").includes("json")) throw new ServerDown("The model server is not running.");
  const out = await res.json();
  if (!res.ok) throw new Error(out.error ?? "The request failed.");
  return out;
}

/** The first trading day on or after a date, from a sorted list of days. */
export function snapToTradingDay(dates: string[], d: string) {
  return dates.find((x) => x >= d) ?? dates[dates.length - 1];
}

/** The next weekday. Exchange holidays are not known here, so it is only a place on the chart's axis. */
export function nextWeekday(d: string) {
  const t = new Date(d);   // a bare date is read as UTC
  do t.setUTCDate(t.getUTCDate() + 1); while (t.getUTCDay() === 0 || t.getUTCDay() === 6);
  return t.toISOString().slice(0, 10);
}
