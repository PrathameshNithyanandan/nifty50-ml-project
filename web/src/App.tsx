import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation, useParams } from "react-router-dom";
import { PAGES } from "./lib/nav";
import { ThemeToggle } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const Pipeline = lazy(() => import("./pages/Pipeline"));
const Unit1 = lazy(() => import("./pages/Unit1"));
const Unit2 = lazy(() => import("./pages/Unit2"));
const Unit3 = lazy(() => import("./pages/Unit3"));
const Unit4 = lazy(() => import("./pages/Unit4"));
const Unit5 = lazy(() => import("./pages/Unit5"));
const Conclusions = lazy(() => import("./pages/Conclusions"));
const LiveDemo = lazy(() => import("./pages/LiveDemo"));

function SectionScroll() {
  const { section } = useParams();
  const { pathname } = useLocation();
  useEffect(() => {
    if (pathname.startsWith("/live-demo") && section) return;   // here a section is a tab, and its controls should stay in view
    if (!section) { window.scrollTo({ top: 0, behavior: "auto" }); return; }
    // Charts above the section load late and push it down, so keep aligning until the layout settles or the reader scrolls.
    let tries = 0, timer = 0, stopped = false, lastTop = NaN, still = 0;
    const stop = () => { stopped = true; window.clearTimeout(timer); };
    const go = () => {
      if (stopped) return;
      const el = document.getElementById(section);
      if (el) {
        el.scrollIntoView({ behavior: "auto", block: "start" });
        const top = el.getBoundingClientRect().top + window.scrollY;
        still = top === lastTop ? still + 1 : 0; lastTop = top;
      }
      if (still < 6 && tries++ < 60) timer = window.setTimeout(go, 150);
    };
    go();
    window.addEventListener("wheel", stop, { passive: true });
    window.addEventListener("touchmove", stop, { passive: true });
    window.addEventListener("keydown", stop);
    return () => { stop(); window.removeEventListener("wheel", stop); window.removeEventListener("touchmove", stop); window.removeEventListener("keydown", stop); };
  }, [section, pathname]);
  return null;
}
const withScroll = (el: JSX.Element) => <><SectionScroll />{el}</>;

function Nav({ onPick }: { onPick?: () => void }) {
  const { pathname } = useLocation();
  return (
    <nav aria-label="Pages" className="text-sm">
      <ul className="list-none m-0 p-0 grid gap-0.5">
        {PAGES.map((p) => {
          const active = p.path === "/" ? pathname === "/" : pathname.startsWith(p.path);
          return (
            <li key={p.path}>
              <NavLink to={p.path} onClick={onPick} end={p.path === "/"} className={`block rounded px-2.5 py-1.5 no-underline ${active ? "bg-accent-soft text-ink font-semibold" : "text-ink hover:bg-sunk"}`}>{p.short}</NavLink>
              {active && p.sections.length > 0 && (
                <ul className="list-none m-0 mt-0.5 mb-1 pl-3 ml-2.5 border-l border-line grid gap-0.5">
                  {p.sections.map((s) => (
                    <li key={s.id}><NavLink to={`${p.path}/${s.id}`} onClick={onPick} className="block rounded px-2 py-0.5 text-ink2 no-underline hover:text-accent hover:bg-sunk">{s.title}</NavLink></li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const page = PAGES.find((p) => (p.path === "/" ? pathname === "/" : pathname.startsWith(p.path)));
  const routes: [string, JSX.Element][] = [["pipeline", <Pipeline />], ["unit-1", <Unit1 />], ["unit-2", <Unit2 />], ["unit-3", <Unit3 />], ["unit-4", <Unit4 />], ["unit-5", <Unit5 />], ["live-demo", <LiveDemo />], ["conclusions", <Conclusions />]];
  return (
    <div className="lg:flex min-h-screen">
      <aside className="hidden lg:block w-72 shrink-0 sticky top-0 h-screen overflow-y-auto border-r border-line bg-surface px-3 py-4">
        <div className="px-2.5 mb-3"><div className="font-serif text-lg font-semibold leading-tight">NIFTY-50 Results Explorer</div></div>
        <Nav />
        <div className="px-2.5 mt-4"><ThemeToggle /></div>
      </aside>
      <header className="lg:hidden sticky top-0 z-40 bg-surface border-b border-line">
        <div className="flex items-center justify-between gap-2 px-4 h-14">
          <button type="button" aria-expanded={open} aria-controls="mobile-nav" onClick={() => setOpen((o) => !o)} className="rounded-md border border-line bg-surface text-ink px-2.5 py-1 text-sm cursor-pointer shrink-0">{open ? "Close" : "Menu"}</button>
          <span className="font-serif font-semibold truncate text-sm">{page?.short ?? "Results"}</span>
          <ThemeToggle />
        </div>
        {open && <div id="mobile-nav" className="px-3 pb-3 max-h-[75vh] overflow-y-auto border-t border-line pt-2"><Nav onPick={() => setOpen(false)} /></div>}
      </header>
      <main className="flex-1 min-w-0 px-4 sm:px-8 py-6 sm:py-8">
        <div className="max-w-5xl mx-auto min-w-0">
          <Suspense fallback={<p aria-busy="true">Loading page…</p>}>
            <Routes>
              <Route path="/" element={withScroll(<Overview />)} />
              {routes.map(([p, el]) => (
                <Route key={p} path={`/${p}`}>
                  <Route index element={withScroll(el)} />
                  <Route path=":section" element={withScroll(el)} />
                </Route>
              ))}
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  );
}
