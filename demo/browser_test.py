"""Drives every control on the Live demo page in a real browser and takes screenshots. The demo server must be running."""
import pathlib, sys, time
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "web/screenshots"); OUT.mkdir(parents=True, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name, detail)
    if not ok: fails.append(name)


def audit(pg, name):
    info = pg.evaluate("""() => { const de = document.documentElement; const t = document.body.innerText;
      return { overflow: de.scrollWidth - de.clientWidth, bad: t.match(/undefined|NaN|\\[object Object\\]/g) || [], empty: [...document.querySelectorAll('figure')].filter(f => !f.querySelector('.js-plotly-plot .main-svg')).length }; }""")
    check(f"{name}: no overflow, bad text or empty charts", info["overflow"] <= 0 and not info["bad"] and not info["empty"], str(info))


def click(pg, text, exact=False): pg.get_by_role("button", name=text, exact=exact).first.click()


def settle(pg, ms=500): pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(ms)


with sync_playwright() as p:
    b = p.chromium.launch()
    for width, height, tag in [(1440, 900, "1440"), (390, 844, "390")]:
        for theme in ("light", "dark"):
            ctx = b.new_context(viewport=dict(width=width, height=height), color_scheme=theme)
            pg = ctx.new_page(); errs = []
            pg.on("console", lambda m, e=errs: e.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda x, e=errs: e.append(str(x)))
            L = f"{tag} {theme}"
            pg.goto(f"{BASE}?theme={theme}#/live-demo/replay"); pg.wait_for_selector("text=Run the models", timeout=30000)
            settle(pg)
            # replay
            click(pg, "Run the models"); pg.wait_for_selector("text=Reveal what happened", timeout=30000); settle(pg, 1200)
            check(f"{L} outcome hidden", pg.get_by_text("Hidden", exact=True).count() == 1 and pg.get_by_text("✓ Right").count() == 0)
            click(pg, "Reveal what happened"); pg.wait_for_selector("text=✓ Right, text=✗ Wrong".split(",")[0], timeout=30000) if False else pg.wait_for_function("document.body.innerText.includes('Wrong') || document.body.innerText.includes('Right')", timeout=30000); settle(pg, 900)
            check(f"{L} reveal shows verdicts", pg.locator("td", has_text="Wrong").count() + pg.locator("td", has_text="Right").count() >= 16)
            pg.screenshot(path=str(OUT / f"live-replay_{tag}_{theme}.png"), full_page=True); audit(pg, f"{L} replay")
            click(pg, "Next trading day"); settle(pg, 200); check(f"{L} changing day hides the outcome", pg.get_by_text("Reveal what happened").count() == 0)
            click(pg, "Day before the biggest rise"); click(pg, "Latest day"); click(pg, "Previous trading day")
            click(pg, "Play 50 days"); pg.wait_for_selector("text=Days played", timeout=30000)
            click(pg, "Skip to the end"); pg.wait_for_selector("text=Price models over the same days", timeout=30000); settle(pg, 900)
            click(pg, "Day before the biggest fall"); click(pg, "Play to the end of the data"); pg.wait_for_selector("text=Days played", timeout=30000)
            pg.wait_for_selector("text=Price models over the same days", timeout=40000); settle(pg, 900)
            pg.screenshot(path=str(OUT / f"live-replay-played_{tag}_{theme}.png"), full_page=True); audit(pg, f"{L} played")
            # what-if
            click(pg, "What-if"); pg.wait_for_selector("text=Reset to the real day", timeout=15000); settle(pg, 1200)
            first = pg.locator("td").filter(has_text="0.").first.inner_text()
            s = pg.get_by_label("today's return", exact=True); s.focus(); pg.keyboard.press("End"); settle(pg, 800)
            check(f"{L} slider flags out-of-range", pg.get_by_text("outside what 98%").count() >= 1)
            click(pg, "Fri", exact=True); settle(pg, 500)
            pg.screenshot(path=str(OUT / f"live-whatif_{tag}_{theme}.png"), full_page=True); audit(pg, f"{L} what-if")
            click(pg, "Reset to the real day"); settle(pg, 600)
            # retrain
            click(pg, "Retrain", exact=True); pg.wait_for_selector("text=Train all models", timeout=15000)
            click(pg, "Train all models"); pg.wait_for_function("document.body.innerText.includes('Kalman, local linear trend') && !document.body.innerText.includes('Fitting ')", timeout=90000); settle(pg, 500)
            nb = pg.locator("table").first.inner_text()
            pg.screenshot(path=str(OUT / f"live-retrain_{tag}_{theme}.png"), full_page=True); audit(pg, f"{L} retrain")
            pg.get_by_label("Set tree depth").check(); pg.get_by_label("Maximum tree depth").focus(); pg.keyboard.press("End")
            pg.get_by_label("Set regularisation λ").check()
            click(pg, "Train all models"); pg.wait_for_function("document.body.innerText.includes('leaves')", timeout=90000); pg.wait_for_function("!document.body.innerText.includes('Fitting ')", timeout=90000); settle(pg, 300)
            check(f"{L} deep tree reports overfit", "training accuracy 100.0%" in pg.inner_text("main"))
            click(pg, "Back to the notebook's settings")
            # today
            click(pg, "Today's market"); click(pg, "Fetch prices and run the models"); pg.wait_for_selector("text=Models trained up to", timeout=60000); settle(pg, 1200)
            pg.screenshot(path=str(OUT / f"live-today_{tag}_{theme}.png"), full_page=True); audit(pg, f"{L} today")
            # another stock, and a delisted one
            pg.get_by_label("Stock", exact=True).select_option("HDFC"); pg.wait_for_selector("text=Fetch prices and run the models", timeout=60000); click(pg, "Fetch prices and run the models"); pg.wait_for_selector("text=no recent prices", timeout=30000)
            pg.get_by_label("Stock", exact=True).select_option("TCS"); pg.wait_for_selector("text=Fetch prices and run the models", timeout=60000)
            click(pg, "Replay a day"); pg.wait_for_selector("text=Run the models", timeout=15000); click(pg, "Run the models"); pg.wait_for_selector("text=Reveal what happened", timeout=30000)
            check(f"{L} no console errors", not errs, str(errs[:3]))
            ctx.close()
    # with the API unreachable, the page explains how to start it
    ctx = b.new_context(viewport=dict(width=1440, height=900)); pg = ctx.new_page()
    pg.route("**/api/**", lambda r: r.abort()); pg.goto(f"{BASE}#/live-demo"); pg.wait_for_selector("text=The model server is not running", timeout=15000)
    settle(pg); pg.screenshot(path=str(OUT / "live-server-off_1440_light.png"), full_page=True)
    check("server-off message shows the command", pg.get_by_text("demo.server").count() == 1)
    pg.goto(f"{BASE}#/unit-2"); pg.wait_for_selector("main h1", timeout=15000); check("other pages work without the API", pg.locator("main h1").inner_text().startswith("Unit 2"))
    ctx.close(); b.close()
print("FAILED:" if fails else "All browser checks passed.", fails if fails else "")
sys.exit(1 if fails else 0)
