import { ReactNode, useMemo, useState } from "react";

export type Col<R> = { key: string; label: ReactNode; value?: (r: R) => number | string | null | undefined; render?: (r: R) => ReactNode; align?: "left" | "right"; title?: string; wrap?: boolean };

/** Sortable table. It scrolls inside its own box on narrow screens, so the page never scrolls sideways. */
export function Table<R>({ cols, rows, highlight, caption, sortable = true, initialSort, maxHeight }: { cols: Col<R>[]; rows: R[]; highlight?: (r: R) => boolean; caption?: string; sortable?: boolean; initialSort?: { key: string; dir: 1 | -1 }; maxHeight?: number }) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(initialSort ?? null);
  const sorted = useMemo(() => {
    if (!sort) return rows;
    const c = cols.find((x) => x.key === sort.key);
    if (!c) return rows;
    const val = c.value ?? ((r: any) => r[c.key]);
    return [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
    });
  }, [rows, sort, cols]);
  return (
    <div className="overflow-x-auto rounded border border-line" style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined} tabIndex={0} role="region" aria-label={caption ?? "Table"}>
      <table className="w-full border-collapse text-sm tnum">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead className="sticky top-0 bg-sunk z-10">
          <tr>
            {cols.map((c) => (
              <th key={c.key} scope="col" title={c.title} aria-sort={sort?.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                className={`px-2.5 py-1.5 font-semibold whitespace-nowrap border-b border-line ${c.align === "left" ? "text-left" : "text-right"}`}>
                {sortable ? (
                  <button type="button" className="bg-transparent border-0 p-0 [font:inherit] text-inherit cursor-pointer" onClick={() => setSort((s) => (s?.key === c.key ? { key: c.key, dir: s.dir === 1 ? -1 : 1 } : { key: c.key, dir: 1 }))}>
                    {c.label}<span className="text-muted ml-1" aria-hidden>{sort?.key === c.key ? (sort.dir === 1 ? "▲" : "▼") : "↕"}</span>
                  </button>
                ) : c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={i} className={`border-b border-line last:border-0 ${highlight?.(r) ? "bg-accent-soft font-medium" : ""}`}>
              {cols.map((c) => (
                <td key={c.key} className={`px-2.5 py-1.5 ${c.wrap ? "" : "whitespace-nowrap"} ${c.align === "left" ? "text-left" : "text-right"}`}>{c.render ? c.render(r) : String((r as any)[c.key] ?? "—")}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Colour swatch that sits beside a model name, so identity never rests on the text colour. */
export function Dot({ color }: { color: string }) {
  return <span aria-hidden className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5" style={{ background: color }} />;
}
