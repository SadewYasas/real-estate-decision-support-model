# Results summary – Intelligent Real Estate Investment Decision Support Using Market Forecasting

Reference for writing Chapter 5. Every number below is read from a file in `artefacts/`
(named in each table), produced by the code in `src/`. Nothing is estimated by hand.

> **Objectives O1–O6.** The objective wording is not stored in the repository, so this
> summary uses the mapping below, taken from the system description in `CLAUDE.md`. If
> your Chapter 1 objectives are worded or numbered differently, keep the evidence and
> change the labels.
>
> | ID | Objective (as assumed here) |
> |----|-----------------------------|
> | O1 | Predict the current sale price of a US home from location and size (ML model 1) |
> | O2 | Predict the monthly rent of an equivalent home (ML model 2) |
> | O3 | Forecast next-year house price appreciation (state) and rent growth from macroeconomic indicators, and test whether the indicators help (RQ4) |
> | O4 | Build a transparent rent-versus-buy NPV engine with break-even, sensitivity and macro scenarios |
> | O5 | Explain the price and rent predictions (SHAP) |
> | O6 | Integrate everything in a usable, fast decision-support system and show that forecast-based growth changes decisions (the research gap) |

---

## 1. Reproducibility

| Item | Result | Source |
|------|--------|--------|
| Full pipeline, raw data → all outputs (`python -m src.run_pipeline`) | 15 steps, all OK, 4.0 min | `metrics/pipeline_run.json` |
| Re-run compared with the previous run (timings / timestamps ignored) | **13 of 13 metrics files identical** | `metrics/pipeline_run.json` |
| Saved sale and rent models re-scored on the test split rebuilt from freshly cleaned data | **identical metrics** (max difference 0.0) for both | `metrics/price_models_reproducibility.json` |
| Automated tests (`pytest tests`) | **85 passed**, 0 failed | `logs/pipeline/tests.log` |
| Front end (`eslint`, `vite build`) | clean, builds | – |
| Random seed / library versions | `random_state = 42` everywhere; versions pinned in `requirements.txt`, full list in `requirements-lock.txt`, recorded in `models/*_meta.json` | – |

Retraining the sale and rent models from scratch takes about 3 h (CatBoost search) and is
run separately: `python -m src.run_pipeline --train`. The 4-minute run re-scores the saved
models instead and shows they reproduce the reported metrics exactly.

---

## 2. Data and data-quality findings

### 2.1 Data used

| Data | Rows after cleaning | Notes | Source |
|------|--------------------|-------|--------|
| D1 sale listings (`Final Dataset.csv`) | 39,981 → **38,058** (29 states) | 1,447 duplicates, 474 out-of-range rows and 2 rows with missing income removed; six macro columns dropped | `metrics/cleaning_sale.json` |
| D2 rent listings (UCI, Dec 2018 – Dec 2019) | 99,492 → **98,827** (51 states) | 84 repeated ids and 471 rows missing key fields removed; studios kept | `metrics/cleaning_rent.json` |
| D3 FRED state panel | 51 states, Jan 2012 – Sep 2025 | Sep 2025 partly placeholder (2.2.4) | `data/processed/panel_data_quality.json` |
| D4 FHFA HPI, all-transactions, state, quarterly | 51 states, 1975 Q1 – 2026 Q2 | forecast target | – |
| D5 BLS CPI rent, US city average (CUUR0000SEHA) | monthly to Aug 2026 | forecast target and rent inflation factor; Oct 2025 missing (2.2.5) | – |
| Census 2020 ZCTA Gazetteer + ACS 2020-24 5-year | 32,793 ZIPs, 51 states | ZIP lookup for the API | `metrics/zip_lookup_comparison.json` |

### 2.2 Data-quality findings (state these in Chapter 3/5)

**2.2.1 PPSq target leakage (old training file `house_train_proc.csv`).** PPSq × living area
reproduces the price to within 4.8 × 10⁻¹⁰, so PPSq is the target in disguise. Test R² on
log price is 0.608 (RF) / 0.648 (XGBoost) without PPSq and **0.996 / 0.993 with it**.
`Zipcode_AvgPrice`, a ZIP mean of the target over the whole file, also inflates R² to
0.749 / 0.753. The original prototype's model was trained on this file. It is not used, and
it has been moved to `legacy/`. Source: `metrics/leakage_audit.json`.

**2.2.2 Fabricated macro columns in D1.**
- **They vary where real data can't.** Real state macro values are identical for every house
  in a state at a given date, and national values (CPI, mortgage rate) are identical in every
  state. In D1 all six vary from house to house within a state.
- **Two of them track price within a state:** mortgage rate r = −0.648 and interest rate
  r = −0.533 (pooled within-state correlation with log price).
- **The shuffle test exposes the leak.** Shuffling the columns within each state keeps any
  genuine state-level signal. Test R² falls from **0.983 to 0.849** (XGBoost; RF 0.934 →
  0.832), almost exactly the R² with the columns removed (0.856 / 0.842). So they carry
  leaked, not real, information. All six were dropped.

Source: `metrics/leakage_audit.json`.

**2.2.3 D1 "median household income" is actually mean household income.** Comparing D1 with
ACS 2020-24 ZCTAs in the same 29 states, weighted by population (quantiles 5/25/50/75/95%):

| | 5% | 25% | 50% | 75% | 95% |
|---|---|---|---|---|---|
| D1 `median_household_income` | 53.6k | 76.7k | 100.6k | 134.9k | 201.2k |
| ACS median household income (B19013) | 45.9k | 63.3k | 80.1k | 104.4k | 151.5k |
| ACS **mean** household income (B19025 ÷ B11001) | 62.3k | 82.5k | **102.6k** | 132.7k | 197.9k |

33 D1 ZIP profiles exceed the ACS median's top-code of $250,001, which no ACS median can do.
The sale model is therefore fed ACS **mean** income in the live system.

D1 density is people per **square mile**: its median is 1,589, against 1,678 per sq mi and
648 per sq km for ACS. Source: `metrics/zip_lookup_comparison.json`.

**2.2.4 Placeholder values in the last month of `fred_final.csv` (Sep 2025).**
- **State series filled with single values.** State unemployment is 4.4% in all 51 states
  (the national rate; e.g. CA really 5.5%), and state permits are 1,258.4 in every state.
- **National series carried forward.** National CPI and national permits repeat August
  exactly.

They were detected automatically, set to missing, and the latest genuine month used instead
(Aug 2025 for these four series; Sep 2025 for the mortgage and fed funds rates, which are
genuine). The rule for national permits is restricted to proven placeholder months: a
genuine repeat in Jan 2023 was correctly kept. Source: `data/processed/panel_data_quality.json`.

**2.2.5 Missing CPI rent value, Oct 2025** (not published because of the US government
shutdown). All series are put on a full monthly calendar before lags are taken, so shifts are
by date, never by row. Without this, every rent target after the gap would have been
misaligned. Source: `data/processed/panel_data_quality.json`.

**2.2.6 Other findings.**
- **D1 repeats houses.** It contains the same house up to 5 times with different macro values
  (deduplicated on the kept columns).
- **D2 relists flats.** 14,530 D2 rows re-list the same flat under a new id.
- **D2 coordinates are city-level.** 98,827 listings sit on only 7,361 distinct coordinate
  points (city/ZIP centroids).
- **ACS income gaps.** 2,600 ZIPs have no ACS median income (handled by state-median
  fallback, reported in the API response).
- **Annual GDP is stamped too early.** Annual state GDP is stamped in January of its year in
  FRED, so it is lagged to when it is actually published (2.3).

### 2.3 Methodological decisions

| Decision | Why | Evidence |
|----------|-----|----------|
| **Grouped train/test split and GroupKFold**. Sale groups: same ZIP + size + beds + baths (2,920 rows in repeated groups). Rent groups: same lat + lon + floor area (55,909 rows) | The same home, or an identical floor plan, must never be in both train and test | Rent: an ungrouped split would report R² 0.843 / MAPE 12.6% against the honest **0.824 / 13.6%**. Sale: 0.862 vs **0.857** |
| **Monotonic constraints** on the HPI model: mortgage rate, its 12-month change, unemployment and its 12-month change can only lower predicted growth | Unconstrained, a +1 pp rate shock *raised* forecast growth (learned from 2020–22 coincidences) | Constrained RMSE 5.38 vs 5.41 (ratio 0.995, within the 5% rule); DM p = 0.715 (no accuracy cost); rate shock now lowers g by 1.0 pp |
| **ARIMA for rent growth**, with macro shocks applied as the change predicted by the rent gradient-boosting model | ARIMA had the lowest rent backtest RMSE (1.62 vs 2.02) | `metrics/forecast.json` |
| **CPI rent factor to Sep 2025** (the rent forecast origin), listing-weighted 1.327 (1.301–1.349 by month) | The rent level and the growth forecast start on the same date, so growth isn't counted twice | `models/cpi_rent_factors.json` |
| **ACS mean income** for the sale model | Matches the D1 training scale (2.2.3) | `metrics/zip_lookup_comparison.json` |
| **Strictly as-of forecasting**: rows train a model only if their target was known at the origin; state GDP lagged to its publication date; settings fixed in advance, not tuned on the backtest | No look-ahead | `src/models/train_forecast.py` |
| National rent growth only (regional CPI rent not used) | The student's decision; the same q applies in every state | – |
| RandomizedSearchCV n_iter = 20; one level of parallelism (CatBoost single search worker, all threads per fit) | The first run crashed (worker memory) | `logs/train_*.log` |
| Prices back-transformed with exp(log prediction) | Estimates the median; explains R² in $ < R² in log | – |

---

## 3. Chapter 5 – Evaluation Criteria

| What is evaluated | Criteria | Protocol |
|-------------------|----------|----------|
| Price and rent models (O1, O2) | MAE, RMSE, MAPE and R² in dollars; R² and RMSE on log scale; CV mean ± std | Grouped 80/20 split; 5-fold GroupKFold on train; RandomizedSearchCV (20 settings); selection by lowest mean CV RMSE (log). The linear regression baseline is always reported |
| Forecasts (O3) | MAE and RMSE (percentage points of growth); directional accuracy; acceleration accuracy (did growth speed up or slow down vs last year); 10th/90th percentile error band and its coverage; Diebold–Mariano test (Newey–West, lag = horizon − 1) | Expanding-window rolling origin from 2018: HPI 30 quarterly origins × 51 states, rent 91 monthly origins. Baselines: persistence, ARIMA, the same model without macro features |
| Engine (O4) | Pass/fail per scenario, to the cent | 11 hand-calculated scenarios checked three ways (hand, independent month-by-month code, Excel formulas) plus property tests |
| Explanations (O5) | SHAP contributions must reconstruct each prediction exactly | Tested in `tests/test_api.py` |
| System (O6) | All API endpoints and validation tested; full analysis under 3 s (mean and p95 over 100 requests); decision-flip rate with a 95% CI | Flask test client and HTTP; state-cluster bootstrap (2,000 resamples) |

---

## 4. Chapter 5 – Test Results

| Test group | Result | Source |
|------------|--------|--------|
| Engine manual scenarios (zero growth, 100% down payment, H > T, interest-free loan, discounting, rent compounding, published $1,438.92 payment, never breaks even, user cost, falling prices, typical case) | **11 / 11 pass**: hand values to the cent, independent monthly simulation to 1e-9, break-even year | `metrics/engine_tests.json` |
| Engine property and validation tests (loan repaid exactly at T for 12 rate × term combinations, Δ monotone in g and r, 10 invalid inputs rejected, disclaimer present) | **28 / 28 pass** | `metrics/engine_tests.json` |
| Excel check workbook, recalculated by Microsoft Excel | **99 / 99 values match the engine, 0 formula errors, "ALL PASS"** | `engine_test_scenarios.xlsx`, `metrics/engine_tests.json` |
| Scenario and sensitivity tests (documented method, zero shock changes nothing, constrained model monotone in all 51 states for each of the 4 features) | 9 / 9 pass | `tests/test_scenario.py` |
| API tests (every endpoint, 14 invalid inputs → 422, unknown ZIP with / without state, ZIP–state mismatch, out-of-training-state and rent-extrapolation warnings, SHAP reconstructs predictions, disclaimer on every response including errors, removed legacy route → 404) | 37 / 37 pass | `tests/test_api.py` |
| **Total** | **85 passed, 0 failed** | `logs/pipeline/tests.log` |
| Front end (manual browser test, laptop 1,300–1,400 px and phone 375 px emulation) | No horizontal overflow on any page; sliders update the result (Austin flips rent → buy at 25 years / 7% growth); Reset restores it; error and loading states shown | Step 9 notes |

---

## 5. Chapter 5 – System Performance

| Measure | Result | Requirement | Source |
|---------|--------|-------------|--------|
| Full analysis (`POST /api/analyse`: price, rent, SHAP, forecasts, 30-year engine table), 100 varied requests, in-process | mean **136 ms**, median 134 ms, p95 **147 ms**, max 165 ms | < 3 s ✔ | `metrics/performance.json` |
| Same over HTTP (localhost, Flask development server) | mean **131 ms**, p95 **144 ms**, max 160 ms | < 3 s ✔ | `metrics/performance.json` |
| Requests succeeding | 105 / 105 (HTTP 200) | – | `metrics/performance.json` |
| Model loading at start-up (once) | 1.5 s | load once ✔ | `metrics/performance.json` |
| Decision-flip experiment (7,622 properties × 3 settings × 2 price sources) | 11 s | – | `logs/pipeline/decision_flip.log` |

Measured on the student's Windows 11 laptop (Python 3.14.7). Timings vary slightly between
runs. 100% of requests finished under 3 s.

---

## 6. Chapter 5 – Technical Measures

### 6.1 Sale price model (O1) – `metrics/sale_models.json`
Train 30,436 / test 7,622 homes, 29 states. Features: state, beds, baths, living area, ZIP
density, ZIP income.

| Model | CV RMSE (log) | CV R² (log) | Test R² (log) | Test R² ($) | MAPE | MAE | RMSE |
|-------|---------------|-------------|---------------|-------------|------|-----|------|
| Linear regression (baseline) | 0.374 ± 0.005 | 0.764 ± 0.003 | 0.762 | 0.715 | 30.2% | $161,742 | $310,314 |
| Random forest | 0.314 ± 0.007 | 0.833 ± 0.004 | 0.836 | 0.786 | 23.4% | $130,491 | $269,361 |
| XGBoost | 0.298 ± 0.007 | 0.850 ± 0.004 | 0.856 | 0.820 | 22.0% | $121,891 | $246,876 |
| **CatBoost (selected)** | **0.297 ± 0.006** | **0.851 ± 0.003** | **0.857** | **0.829** | **22.0%** | **$120,036** | **$240,674** |

- **Beats the baseline:** R² (log) +0.095 and MAPE −8.2 pp. Meets the plan's target of
  R² ≥ 0.80.
- **CatBoost and XGBoost are statistically tied.** The CV RMSE gap (0.0008) is much smaller
  than one standard deviation (≈ 0.006).
- **Error varies by state.** Test MAPE is lowest in NC (16.5%), WA (17.0%) and NE (17.2%), and
  highest in WI (30.3%), LA (33.9%) and MI (41.3%). Cheaper homes have larger percentage
  errors.

### 6.2 Rent model (O2) – `metrics/rent_models.json`
Train 78,935 / test 19,892 listings, 51 states. Features: state, beds, baths, floor area,
latitude, longitude.

| Model | CV RMSE (log) | CV R² (log) | Test R² (log) | Test R² ($) | MAPE | MAE | RMSE |
|-------|---------------|-------------|---------------|-------------|------|-----|------|
| Linear regression (baseline) | 0.287 ± 0.003 | 0.566 ± 0.006 | 0.569 | 0.491 | 22.2% | $341 | $570 |
| Random forest | 0.192 ± 0.002 | 0.805 ± 0.003 | 0.819 | 0.792 | 13.8% | $209 | $364 |
| XGBoost | 0.189 ± 0.003 | 0.813 ± 0.005 | 0.824 | 0.791 | 13.6% | $209 | $365 |
| **CatBoost (selected)** | **0.187 ± 0.002** | **0.816 ± 0.004** | **0.824** | **0.792** | **13.6%** | **$209** | **$364** |
| CatBoost without coordinates (ablation; fallback model) | – | – | 0.611 | 0.538 | 20.9% | $324 | $542 |

- **Beats the baseline:** R² (log) +0.255 and MAPE −8.6 pp.
- **Location is essential.** Removing coordinates costs 0.21 of R². This is why the API
  includes a ZIP → coordinates lookup.
- **Model predicts 2019 rents.** Its output is multiplied by the CPI factor 1.327 to reach
  Sep 2025.
- **Small-state errors are unreliable.** The worst states have 2–25 test listings each
  (WV 38.5%, RI 28.6%, WY 27.9%).

### 6.3 Forecasts (O3, RQ4) – `metrics/forecast.json`, `models/forecasts.json`

**House price growth**: 4-quarter-ahead growth (percentage points), 51 states × 30 origins
(2018 Q1 – 2025 Q2), 1,530 forecasts.

| Method | MAE | RMSE | Direction acc. | Acceleration acc. |
|--------|-----|------|----------------|-------------------|
| **Gradient boosting + macro, monotonic (selected)** | **3.58** | **5.38** | 98.4% | 59.0% |
| Gradient boosting + macro, unconstrained | 3.59 | 5.41 | 98.4% | 58.4% |
| Gradient boosting, no macro | 4.90 | 7.09 | 98.4% | 46.4% |
| Persistence | 4.35 | 6.59 | 97.1% | – |
| ARIMA | 4.15 | 6.03 | 97.9% | 61.1% |

Diebold–Mariano tests (constrained model vs each method):

| Comparison | p | Significant? |
|------------|---|--------------|
| vs no-macro | **0.043** | Yes |
| vs unconstrained | 0.715 | No |
| vs persistence | 0.281 | No |
| vs ARIMA | 0.204 | No |

**Rent growth**: 12-month-ahead, US, 91 monthly origins (Jan 2018 – Aug 2025).

| Method | MAE | RMSE | Direction acc. | Acceleration acc. |
|--------|-----|------|----------------|-------------------|
| Gradient boosting + macro | 1.35 | 2.02 | 100% | 75.8% |
| Gradient boosting, no macro | 1.65 | 2.29 | 100% | 62.6% |
| Persistence | 1.74 | 2.24 | 100% | – |
| **ARIMA (selected)** | **1.12** | **1.62** | 100% | **82.4%** |

Diebold–Mariano tests (gradient boosting + macro vs each method):

| Comparison | p | Significant? |
|------------|---|--------------|
| vs no-macro | 0.267 | No |
| vs persistence | 0.442 | No |
| vs ARIMA | 0.107 | No (ARIMA better) |

**Answer to RQ4 (do macro indicators improve the forecast?)**
- **House prices: yes, partly.** Macro features significantly improve the same model
  (p = 0.043), and the macro model has the lowest error of all methods. Its advantage over
  persistence and ARIMA is not statistically significant: 30 overlapping forecasts give the
  tests little power.
- **Most of the gain comes from 2022.** That year the model's MAE was 3.06 against 10.46 for
  persistence, because rising mortgage rates signalled the slowdown. In 2018–19 and 2023–25
  the methods are close, and all of them missed the 2020–21 boom (MAE about 8–9).
- **Rent: macro features help the gradient-boosting model, but ARIMA is best.** With one
  national series there is too little data for the macro model.
- **Directional accuracy isn't informative.** Growth was positive in 98.4% (HPI) and 100%
  (rent) of cases, so "always up" scores almost perfectly. Use acceleration accuracy instead.

**Latest forecasts** (origin 2025 Q3 / Sep 2025):
- **House prices:** mean across states **+4.42%** (range +1.35% to +5.99%), with an average
  80% band of [+0.73%, +13.24%]. The band is lopsided because of the 2020–21 boom errors.
  Band coverage on the backtest is 80% by construction.
- **Rent:** **+3.57%** with band [+2.58%, +6.94%].
- **Mortgage rate used:** 6.35%.
- **Reality check** (already observed, not a metric): house prices rose 2.63% on average over
  the first 3 quarters; rents rose 2.46% over the first 11 months. Both forecasts look
  somewhat high so far.

### 6.4 Explanations (O5) – `metrics/*_models.json`, `figures/*_shap_summary.png`
Mean |SHAP| on log scale, from 2,000 test homes:

| Model | 1st | 2nd | 3rd | 4th | 5th | 6th |
|-------|-----|-----|-----|-----|-----|-----|
| Price | living area 0.260 | state 0.257 | ZIP income 0.223 | ZIP density 0.155 | baths 0.075 | beds 0.024 |
| Rent | longitude 0.126 | state 0.117 | floor area 0.112 | latitude 0.111 | baths 0.055 | beds 0.018 |

- **Location matters as much as size for price, and more than size for rent.**
- **Per-prediction explanations are exact.** Each request returns its top-5 contributions,
  which reconstruct the prediction exactly (tested).

### 6.5 Rent-vs-buy engine, sensitivity and scenarios (O4)
Worked example (illustrative, not a valuation):
- **Property:** 3-bed, 2-bath, 1,800 sq ft house in Austin, TX.
- **Inputs:** predicted price **$374,575** and rent **$2,600**/month; g 3.09%, q 3.57%,
  r 6.35%, H = 7.

Engine results (`metrics/engine_example.json`):
- monthly payment $1,865;
- PV cost of buying **$153,662** against renting **$199,725**, so **buy, saving $46,063**;
- break-even in **year 3**;
- user cost $15,956 a year against $31,201 rent.

Scenarios (`metrics/scenario_example.json`):

| Scenario | g | q | r | Δ(7) | Decision | Break-even |
|----------|---|---|---|------|----------|------------|
| Base | 3.09% | 3.57% | 6.35% | +$46,063 | buy | 3 |
| Pessimistic (p10) | −0.60% | 2.58% | 6.35% | −$23,788 | **rent (flips)** | 12 |
| Optimistic (p90) | 11.91% | 6.94% | 6.35% | +$290,615 | buy | 1 |
| Rates +1 pp | 2.09% | 3.56% | 7.35% | +$10,102 | buy | 6 |
| Rates −1 pp | 3.05% | 2.39% | 5.35% | +$55,387 | buy | 3 |
| Recession | 3.08% | 3.55% | 5.85% | +$54,073 | buy | 3 |
| Inflation | 3.29% | 5.01% | 7.10% | +$45,516 | buy | 4 |

**Sensitivity** (`metrics/sensitivity_example.json`), swing in Δ:
1. house price growth g: $288,659, the **only assumption that flips the decision** within
   its forecast band;
2. holding period: $69,596;
3. price (± the model's 22% error): $67,591;
4. rent (± 14%): $54,437;
5. mortgage rate: $33,711;
6. rent growth: $25,744.

The decision depends on the growth forecast more than on any cost assumption.

### 6.6 Decision-flip experiment (O6, the research gap) – `metrics/decision_flip.json`
**Setup:**
- **Properties:** all 7,622 sale test homes (29 states).
- **Inputs:** predicted price; rent from the no-coordinates rent model × 1.327; CLAUDE.md
  defaults; r = 6.35%.
- **Growth settings:**
  - (a) fixed g = q = 3%, as typical calculators use;
  - (b) each state's 10-year history (mean g 6.87%; q 4.28%);
  - (c) the forecast (mean g 4.03%; q 3.57%).

| H | Buy (a) | Buy (b) | Buy (c) | **Flip (a)↔(c)** [95% CI] | Direction (a→c) | **Flip (b)↔(c)** [95% CI] | Direction (b→c) |
|---|---|---|---|---|---|---|---|
| 5 | 55.1% | 93.8% | 66.1% | **11.5%** [8.4, 15.7] | 858 rent→buy, 21 buy→rent | **27.7%** [19.5, 34.9] | 2,109 buy→rent, 0 rent→buy |
| 7 | 62.9% | 96.5% | 72.7% | **10.3%** [7.3, 13.8] | 763 rent→buy, 21 buy→rent | **23.8%** [16.2, 29.9] | 1,816 buy→rent, 0 rent→buy |
| 10 | 69.0% | 97.9% | 78.3% | **9.8%** [6.8, 13.2] | 730 rent→buy, 20 buy→rent | **19.6%** [12.8, 24.9] | 1,495 buy→rent, 0 rent→buy |

**Break-even shift:**
- **(a) → (c):** mean **−2.01 years** (median −1) across 6,121 homes that break even in both;
  560 more homes never break even under (a) but do under (c).
- **(b) → (c):** mean **+2.75 years** (median +1) across 6,681 homes; 908 homes break even
  under (b) but never under (c).

**By state (H = 7):** the (a)↔(c) flip rate ranges from 0% (LA, where the forecast of 2.9% is
close to 3%) to 30% (NM). The largest (b)↔(c) flip rate is WA at 88% (history 8.1% against a
4.0% forecast).

**Robustness:** with actual listing prices instead of predicted ones, the H = 7 flip rates are
10.2% and 24.2% (main run: 10.3% and 23.8%).

---

## 7. Chapter 5 – Comparison with Existing Solutions

| Compared with | Result | Evidence |
|---------------|--------|----------|
| Typical online rent-vs-buy calculators (fixed 3% growth) | Forecast-based growth changes **≈10% of decisions** (95% CI 7–14% at H = 7) and moves break-even by about 2 years | §6.6 |
| "Extrapolate the last 10 years" | Recommends buying for 94–98% of homes because of the pandemic boom; the forecast reverses about 1 in 4 of those (all buy → rent) | §6.6 |
| Simple forecasting baselines (persistence, ARIMA, model without macro) | The macro model has the lowest HPI error (RMSE 5.38 vs 6.03–7.09); significant only against no-macro | §6.3 |
| Linear regression price/rent models | Gradient boosting reduces MAPE by 8.2 pp (sale) and 8.6 pp (rent) | §6.1–6.2 |
| The original prototype (PPSq model) | Its apparent R² ≈ 0.99 came from target leakage; the honest, leakage-free model reaches 0.857 and adds rent, forecasts, the NPV engine, explanations, uncertainty ranges and a disclaimer | §2.2.1 |

Comparisons with commercial products (for example automated valuation models) need
published figures from the literature, with citations. None are quoted here.

---

## 8. Chapter 5 – Results Related to Objectives

| Objective | Status | Key evidence |
|-----------|--------|--------------|
| **O1** sale price | **Met** | CatBoost test R² (log) 0.857, MAPE 22.0%, beats the linear baseline by 0.095 R²; leakage removed; reproduced exactly (§6.1) |
| **O2** rent | **Met** | CatBoost test R² (log) 0.824, MAPE 13.6%, beats the baseline by 0.255 R²; grouped split prevents relisting leakage (§6.2) |
| **O3** forecasts + RQ4 | **Met, with a nuanced answer** | Macro features significantly improve the model (p = 0.043) and give the lowest HPI error, but not significantly better than persistence / ARIMA; ARIMA best for rent; 80% bands provided (§6.3) |
| **O4** NPV engine, sensitivity, scenarios | **Met** | 11/11 hand scenarios, 28/28 property tests, Excel 99/99; tornado and 7 scenarios; monotonic constraints make shocks economically coherent (§4, §6.5) |
| **O5** explanations | **Met** | TreeSHAP global summaries and per-prediction top-5 that reconstruct each prediction exactly (§6.4) |
| **O6** integrated system + research gap | **Met (system); user evaluation pending** | 5 API endpoints, 37 API tests, 136 ms mean / 147 ms p95 (< 3 s), responsive front end with the disclaimer on every result; forecast growth flips ≈10% of decisions vs a fixed 3% (§5, §6.6). SUS + TAM evaluation with 20–30 users still to do |

---

## 9. Limitations (for the Discussion)

1. **Data.**
   - D1 is undated, covers 29 states, and its income column is mislabelled.
   - D2 is 2019 apartments, so rents for houses are extrapolations, partly corrected by CPI.
   - ACS ZCTA values are 2020–24 averages.
2. **Rent in the decision-flip experiment** comes from the no-coordinates model (D1 has no
   coordinates; R² 0.61). Absolute buy shares depend on this rent level (median
   price-to-annual-rent ratio 13.2). Flip rates mainly reflect the growth difference.
3. **Forecasts.**
   - One-year forecasts are held constant over 5–10 years (equally for all three settings).
   - Only one live origin (2025 Q3).
   - The backtest period includes the pandemic boom, which no method predicted.
4. **Scenario shocks** reflect associations the models learned from 2014–2025, applied one at
   a time. Only four HPI features are constrained.
5. **Decision-flip shows sensitivity, not correctness.** It shows the decision depends on the
   growth input; it doesn't prove the forecast-based decisions are better. The forecast's
   accuracy evidence is the backtest (§6.3).
6. **No user evaluation yet** (SUS/TAM pending).

---

## 10. Figures (`artefacts/figures/`)

| Figure | Shows | Use in |
|--------|-------|--------|
| `sale_distributions.png`, `rent_distributions.png` | Distributions of every model variable | Ch. 4 data |
| `sale_correlation.png`, `rent_correlation.png` | Spearman correlations | Ch. 4 data |
| `sale_price_by_state.png`, `rent_price_by_state.png` | Price / rent by state | Ch. 4 data |
| `sale_pred_vs_actual.png`, `rent_pred_vs_actual.png` | Predicted vs actual, test set | 6.1, 6.2 |
| `sale_shap_summary.png`, `rent_shap_summary.png` | SHAP beeswarm | 6.4 |
| `forecast_backtest_hpi.png`, `forecast_backtest_rent.png` | Rolling-origin backtest, every method | 6.3 |
| `forecast_error_by_state.png` | HPI forecast MAE by state, model vs persistence | 6.3 |
| `cost_over_time_example.png` | Buy vs rent PV cost over 30 years, break-even | 6.5 |
| `tornado_sensitivity.png` | One-at-a-time sensitivity | 6.5 |
| `decision_flip.png` | Flip rate by H (with CIs) and by state | 6.6 |

## 11. Metrics and model files

`artefacts/metrics/`:
- **Data:** `cleaning_sale.json`, `cleaning_rent.json`, `leakage_audit.json`, `eda_summary.json`
- **Models:** `sale_models.json`, `rent_models.json`, `price_models_reproducibility.json`
- **Forecasts:** `forecast.json`, `forecast_backtest_hpi.csv`, `forecast_backtest_rent.csv`
- **Engine:** `engine_tests.json`, `engine_example.json`, `scenario_example.json`,
  `sensitivity_example.json`
- **Experiment:** `decision_flip.json`, `decision_flip_properties.csv`
- **System:** `performance.json`, `zip_lookup_comparison.json`, `pipeline_run.json`

`artefacts/models/`:
- **Price and rent models:** `sale_model.joblib`, `rent_model.joblib`,
  `rent_model_no_coords.joblib` (each with `*_meta.json`)
- **Forecasting:** `hpi_forecast_model*.joblib`, `rent_forecast_model*.joblib`,
  `forecasts.json`, `cpi_rent_factors.json`

Other outputs:
- `artefacts/engine_test_scenarios.xlsx`: the Excel check workbook
- `data/processed/panel_data_quality.json`: the data-quality log

*Indicative decision support only – not financial or valuation advice.*
