import { ReactNode, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { GLOSSARY } from "../lib/glossary";
import { useTheme } from "../lib/theme";

/** Jargon with a glossary tooltip: hover, keyboard focus or tap. */
export function T({ k, children }: { k: string; children?: ReactNode }) {
  const g = GLOSSARY[k];
  const ref = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; up: boolean } | null>(null);
  if (!g) throw new Error("glossary key " + k);
  const show = () => {
    const r = ref.current!.getBoundingClientRect();
    const w = Math.min(300, window.innerWidth - 24);
    const left = Math.max(12, Math.min(r.left + r.width / 2 - w / 2, window.innerWidth - w - 12));
    const up = r.bottom + 130 > window.innerHeight;
    setPos({ left, top: up ? r.top - 8 : r.bottom + 8, up });
  };
  useEffect(() => {
    if (!pos) return;
    const hide = () => setPos(null);
    window.addEventListener("scroll", hide, true);
    return () => window.removeEventListener("scroll", hide, true);
  }, [pos]);
  return (
    <>
      <button ref={ref} type="button" className="cursor-help underline decoration-dotted decoration-1 underline-offset-[3px] [font:inherit] text-inherit p-0 m-0 bg-transparent border-0 inline"
        aria-label={`${g.term}: ${g.def}`} onMouseEnter={show} onMouseLeave={() => setPos(null)} onFocus={show} onBlur={() => setPos(null)} onClick={(e) => { e.preventDefault(); show(); }}>
        {children ?? g.term}
      </button>
      {pos && (
        <span role="tooltip" className="fixed z-50 block rounded-md border border-line bg-surface p-3 text-sm leading-snug text-ink shadow-lg text-left font-sans" style={{ left: pos.left, top: pos.top, width: Math.min(300, window.innerWidth - 24), transform: pos.up ? "translateY(-100%)" : undefined }}>
          <strong className="block mb-0.5">{g.term}</strong>{g.def}
        </span>
      )}
    </>
  );
}

function Label({ children }: { children: ReactNode }) {
  return <div className="text-xs font-semibold uppercase tracking-wider text-muted mb-1">{children}</div>;
}

function useBasePath() {
  const params = useParams();
  const base = window.location.hash.replace(/^#/, "").split("?")[0];
  return params.section ? base.slice(0, base.lastIndexOf("/")) : base.replace(/\/$/, "");
}

/** The four-part block used by every result. */
export function Card({ id, title, q, m, tags, meaning, children }: { id: string; title: string; q: ReactNode; m: ReactNode; tags: string[]; meaning: ReactNode; children: ReactNode }) {
  const base = useBasePath();
  return (
    <section id={id} className="scroll-mt-20 rounded-lg border border-line bg-surface mb-8">
      <header className="px-4 sm:px-6 pt-4 sm:pt-5 pb-3 border-b border-line">
        <h2 className="text-xl sm:text-2xl font-semibold m-0 leading-tight">
          {title} <Link to={`${base}/${id}`} aria-label={`Link to ${title}`} className="text-muted text-base no-underline hover:text-accent align-middle">#</Link>
        </h2>
      </header>
      <div className="px-4 sm:px-6 py-4 grid gap-4 min-w-0">
        <div className="grid gap-4 md:grid-cols-2 min-w-0">
          <div className="min-w-0"><Label>Question</Label><div className="text-ink">{q}</div></div>
          <div className="min-w-0">
            <Label>Method</Label>
            <div className="text-ink">{m}</div>
            <div className="mt-2 flex flex-wrap gap-1.5">{tags.map((t) => <span key={t} className="rounded bg-sunk px-2 py-0.5 text-xs text-ink2">{t}</span>)}</div>
          </div>
        </div>
        <div className="min-w-0"><Label>Result</Label><div className="min-w-0">{children}</div></div>
        <div className="rounded-md bg-accent-soft px-4 py-3 min-w-0"><Label>What it means</Label><div className="text-ink">{meaning}</div></div>
      </div>
    </section>
  );
}

export function PageHead({ title, assigned, mapping, practice, children }: { title: string; assigned?: string; mapping?: string; practice?: string; children?: ReactNode }) {
  return (
    <header className="mb-8">
      <h1 className="text-3xl sm:text-4xl font-semibold leading-tight mt-0 mb-3">{title}</h1>
      {assigned && <p className="text-ink2 m-0 mb-2"><span className="font-semibold">Assigned title:</span> {assigned}</p>}
      {mapping && <p className="text-ink2 text-sm m-0 mb-2"><span className="font-semibold">Syllabus mapping:</span> {mapping}</p>}
      {practice && <p className="text-ink2 text-sm m-0 mb-2"><span className="font-semibold">Practice:</span> {practice}</p>}
      {children}
    </header>
  );
}

export function Takeaways({ items }: { items: ReactNode[] }) {
  return (
    <section id="takeaways" className="scroll-mt-20 rounded-lg border-2 border-accent bg-surface px-4 sm:px-6 py-4 mb-8">
      <h2 className="text-xl sm:text-2xl font-semibold mt-0 mb-2">Key takeaways</h2>
      <ul className="m-0 pl-5 grid gap-2 list-disc">{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
    </section>
  );
}

export function Kpi({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-3 min-w-0">
      <div className="text-xs text-muted">{label}</div>
      <div className="text-xl sm:text-2xl font-semibold tnum leading-snug break-words">{value}</div>
      {note && <div className="text-xs text-ink2 mt-0.5">{note}</div>}
    </div>
  );
}
export const KpiRow = ({ children }: { children: ReactNode }) => <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">{children}</div>;

/** One row of mutually exclusive buttons. */
export function Segmented<V extends string | number>({ label, value, options, onChange }: { label: string; value: V; options: { value: V; label: string }[]; onChange: (v: V) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1" role="group" aria-label={label}>
      <span className="text-sm text-ink2">{label}</span>
      <div className="inline-flex flex-wrap rounded-md border border-line overflow-hidden">
        {options.map((o) => (
          <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}
            className={`px-2.5 py-1 text-sm border-0 cursor-pointer ${o.value === value ? "bg-accent text-surface font-medium" : "bg-surface text-ink hover:bg-sunk"}`}>{o.label}</button>
        ))}
      </div>
    </div>
  );
}
export const Controls = ({ children }: { children: ReactNode }) => <div className="flex flex-wrap gap-x-5 gap-y-2 mb-3 items-center">{children}</div>;

export function Note({ children }: { children: ReactNode }) {
  return <p className="text-sm text-ink2 mt-2 mb-0">{children}</p>;
}
export function Formula({ children }: { children: ReactNode }) {
  return <div className="formula rounded bg-sunk px-3 py-2 my-2 text-ink">{children}</div>;
}
export function ThemeToggle() {
  const { mode, toggle } = useTheme();
  return <button type="button" onClick={toggle} className="rounded-md border border-line bg-surface text-ink px-2.5 py-1 text-sm cursor-pointer hover:bg-sunk whitespace-nowrap">{mode === "dark" ? "Light mode" : "Dark mode"}</button>;
}
