import { createContext, ReactNode, useContext, useEffect, useState } from "react";

// Every number on the site is read from these files, which the notebook writes.
const Ctx = createContext<any>(null);
const cache = new Map<string, Promise<any>>();
function load(name: string) {
  if (!cache.has(name)) cache.set(name, fetch(`./data/${name}.json`).then((r) => { if (!r.ok) throw new Error(name); return r.json(); }));
  return cache.get(name)!;
}

export function DataProvider({ children }: { children: ReactNode }) {
  const [R, setR] = useState<any>(null);
  const [err, setErr] = useState(false);
  useEffect(() => { load("results").then(setR).catch(() => setErr(true)); }, []);
  if (err) return <p style={{ padding: 24 }}>Could not load the results data. Serve this folder over HTTP, for example with npm run preview.</p>;
  if (!R) return <div style={{ padding: 24 }} aria-busy="true">Loading results…</div>;
  return <Ctx.Provider value={R}>{children}</Ctx.Provider>;
}
export const useR = () => useContext(Ctx);

export function useChart<T = any>(name: string): T | null {
  const [d, setD] = useState<T | null>(null);
  useEffect(() => { let on = true; load(name).then((x) => { if (on) setD(x); }); return () => { on = false; }; }, [name]);
  return d;
}
