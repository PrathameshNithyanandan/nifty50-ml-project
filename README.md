# Stock Market Analytics & Trend Prediction on NIFTY-50 Stocks

A Machine Learning course project covering Units 1–5, with a web frontend that explains the results.

## Contents

| Path | What it is |
|---|---|
| `analysis.ipynb` | All analysis, executed top to bottom with outputs saved |
| `results/results.json` | Every reported number |
| `figures/` | Every notebook figure |
| `data/` | The Kaggle dataset, unzipped (`archive.zip`) |
| `web/` | The results explorer (Vite, React, TypeScript, Tailwind CSS, Plotly) |
| `web/public/data/` | Data the site reads, written by the notebook |
| `demo/` | The live demo: a local server that runs the models on request |
| `from_scratch/` | Units 2 and 3 with no libraries, and their mathematical model |
| `compare_units_4_5.py` | Compares the Unit 4 and Unit 5 algorithms; writes `results/units_4_5_comparison.md` |
| `web/screenshots/` | Final screenshots at 1440 px and 390 px, in light and dark |

The dataset, fitted-model cache, screenshots and installed packages are not stored in the repository. Download the [NIFTY-50 dataset](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) and unzip it into `data/` before running anything.

## Units 2 and 3 with no libraries

[`from_scratch/MATHEMATICAL_MODEL.md`](from_scratch/MATHEMATICAL_MODEL.md) drafts the mathematical model, and the scripts implement it using only what comes with Python, including the matrix routines. Nothing needs installing:

```bash
python3 from_scratch/unit2.py
python3 from_scratch/unit3.py
python3 from_scratch/check.py
```

`unit2.py` takes under a minute and `unit3.py` about two minutes. `unit2.py --tune-rbf` reruns all the tuning on the training days, in about five minutes on a 10-core laptop. `check.py` compares every result with the notebook, confirms that nothing outside Python's standard library is imported, and writes [`from_scratch/COMPARISON.md`](from_scratch/COMPARISON.md).

## Compare Units 4 and 5

```bash
python3 compare_units_4_5.py
```

This ranks the HMM and the Unit 5 algorithms on the same test days, and the Kalman filter against persistence. The result is in [`results/units_4_5_comparison.md`](results/units_4_5_comparison.md).

## Run the notebook

Requires Python 3.14. A full run takes about four minutes on a 10-core laptop.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 analysis.ipynb
```

## Run the site

```bash
cd web
npm install
npm run dev
```

For the production build, which works offline once installed:

```bash
npm run build
npm run preview
```

The build in `web/dist/` has relative paths and no external requests, but it must be served over HTTP (browsers block these files on `file://`). Any static server works, for example `python3 -m http.server --directory dist`.

## Run the live demo

The site's other pages show saved results. Its Live demo page runs the real models on this computer, using a small local server. Run this from the project folder:

```bash
./start-demo.sh
```

Then open <http://localhost:8000/#/live-demo>. It has four modes:

- **Replay a day:** every model forecasts the day after any test day, and then the outcome is revealed. A player scores them over many days against the baseline.
- **What-if:** drag a day's features and watch every model respond.
- **Retrain:** change the training cutoff, regularisation or tree depth, and refit every model.
- **Today's market:** download recent prices and forecast the next session. This needs internet. If the feed fails, it uses the last saved download and says so.

The fitted models are saved in `demo/cache/`. To rebuild them, which takes about 25 minutes for all stocks:

```bash
.venv/bin/python -m demo.train --all --refit
```

To confirm the demo matches the notebook, with no server running:

```bash
.venv/bin/python -m demo.check
```

With the server running, `demo/test_api.py` and `demo/browser_test.py` test its API and every control on the page.

## Checks

```bash
npm run check:numbers
```

This fails if a number in the site source looks like a result instead of coming from the data files. `web/scripts/screenshots.py` retakes the screenshots and reports page overflow, empty charts and console errors.
