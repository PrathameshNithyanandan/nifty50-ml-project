"""Screenshots every page at two widths and in both themes, and reports layout and console problems.
Run from web/ with the site served by `npm run preview`:  ../.venv/bin/python scripts/screenshots.py"""
import json, sys, time, pathlib
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4173/"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "screenshots"); OUT.mkdir(parents=True, exist_ok=True)
PAGES = {"overview": "/", "pipeline": "/pipeline", "unit-1": "/unit-1", "unit-2": "/unit-2", "unit-3": "/unit-3", "unit-4": "/unit-4", "unit-5": "/unit-5", "live-demo": "/live-demo", "conclusions": "/conclusions"}
report = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for width, height, tag in [(1440, 900, "1440"), (390, 844, "390")]:
        for theme in ("light", "dark"):
            ctx = b.new_context(viewport=dict(width=width, height=height), device_scale_factor=1, color_scheme=theme)
            pg = ctx.new_page(); errs = []
            pg.on("console", lambda m, e=errs: e.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda x, e=errs: e.append(str(x)))
            pg.on("requestfailed", lambda r, e=errs: e.append("request failed " + r.url))
            external = []
            pg.on("request", lambda r, e=external: e.append(r.url) if not r.url.startswith((BASE, "data:", "blob:")) else None)
            for name, route in PAGES.items():
                errs.clear()
                pg.goto(f"{BASE}?theme={theme}#{route}"); pg.wait_for_load_state("networkidle")
                pg.wait_for_function("document.querySelector('main h1') !== null", timeout=30000)
                last, stable = -1, 0
                for _ in range(40):
                    n = pg.evaluate("document.querySelectorAll('.js-plotly-plot .main-svg').length")
                    stable = stable + 1 if n == last else 0; last = n
                    if stable >= 4: break
                    time.sleep(.25)
                time.sleep(.6)
                info = pg.evaluate("""() => {
                  const de = document.documentElement;
                  const figs = [...document.querySelectorAll('figure')];
                  const emptyCharts = figs.filter(f => !f.querySelector('.js-plotly-plot .main-svg')).length;
                  const skeletons = document.querySelectorAll('.skeleton').length;
                  const text = document.body.innerText;
                  const bad = (text.match(/undefined|NaN|\\[object Object\\]/g) || []);
                  const wide = [...document.querySelectorAll('main *')].filter(e => { const r = e.getBoundingClientRect(); const box = e.closest('[role=region],svg,.js-plotly-plot'); return !box && r.right > de.clientWidth + 1 && r.width > 0; }).slice(0,5).map(e => e.tagName + '.' + String(e.className).slice(0,40));
                  return { pageOverflow: de.scrollWidth - de.clientWidth, charts: figs.length, emptyCharts, skeletons, bad, wide, h: de.scrollHeight };
                }""")
                f = OUT / f"{name}_{tag}_{theme}.png"
                pg.screenshot(path=str(f), full_page=True)
                report.append(dict(page=name, width=tag, theme=theme, errors=list(errs), **info))
                print(name, tag, theme, "overflow", info["pageOverflow"], "charts", info["charts"], "empty", info["emptyCharts"], "skeleton", info["skeletons"], "bad", info["bad"], "wide", info["wide"], "errs", errs[:2], "height", info["h"])
            if external: print("EXTERNAL REQUESTS", set(external))
            ctx.close()
    b.close()
fails = [r for r in report if r["pageOverflow"] > 0 or r["emptyCharts"] or r["skeletons"] or r["bad"] or r["errors"]]
print("pages with problems:", len(fails), "of", len(report))
sys.exit(1 if fails else 0)
