# Project: Intelligent Real Estate Investment Decision Support Using Market Forecasting

Final-year BSc (Hons) Computing research project (Coventry University / NIBM, 2024.2).
The code is the research artefact. Its results will be written up in Chapter 5 of the
dissertation, so correctness, reproducibility and honest reporting matter more than features.

## What the system does

For one US residential property (ZIP code, living area, beds, baths) the system:

1. predicts the current **sale price** (ML model 1);
2. predicts the current **monthly rent** of an equivalent property (ML model 2);
3. forecasts the next 12 months' **house price appreciation** (state level) and **rent growth**
   (census-region level) from macroeconomic indicators (ML model 3);
4. runs a **rent-versus-buy NPV engine**, using 1–3 plus user assumptions, and returns the
   yearly costs, break-even year, recommendation, sensitivity and macro scenarios;
5. explains predictions 1 and 2 with **SHAP**.

## Data (put the files in `data/raw/`)

| ID | File | Notes |
|----|------|-------|
| D1 | `Final Dataset.csv` (sale listings, 39,981 rows, 29 states) | USE ONLY: price, beds, baths, living_space, state, zip_code_density, median_household_income. **DROP** the six macro columns (state_unemployment_rate, state_building_permits, state_real_gdp, mortgage_rate, interest_rate, cpi). They are not real FRED values and they leak the price. If a version with Zip Code / Latitude / Longitude exists, prefer it. |
| D1-old | `house_train_proc.csv` | Do not use for training. Its `PPSq` column = price/area, which is target leakage. |
| D2 | `apartments_for_rent_classified_100K.csv` (UCI; `sep=';'`, `encoding='cp1252'`) | Keep category == housing/rent/apartment, price_type == Monthly, currency == USD. Listings are Dec 2018 – Dec 2019. |
| D3 | `fred_final.csv` | Real monthly state panel, 51 states, Jan 2012 – Sep 2025: state_unemployment_rate, state_building_permits, state_real_gdp, mortgage_rate, interest_rate, cpi, gdp, building_permits. |
| D4 | FHFA all-transactions HPI by state (quarterly) | Forecast target for appreciation. **Student to download.** |
| D5 | BLS CPI "Rent of primary residence": CUUR0000SEHA (US) + CUUR0100/0200/0300/0400SEHA (regions) | Forecast target for rent growth; also used to inflate 2019 rents to the D1 reference year. **Student to download.** |
| — | `transactions.csv` | Synthetic data. DO NOT USE. |

Agreed cleaning thresholds:
- D1: drop duplicates; price 30k–5M; beds and baths 1–10; living_space 300–15,000.
- D2: rent 200–15,000; square_feet 150–8,000.

## Non-negotiable research rules

- **No leakage.** No feature may be derived from the target. Fit any target or zip encoding on
  the training folds only. Every forecasting feature must be lagged so that it only uses
  information available at the forecast date.
- **No fabricated or "filled-in" data.** Never synthesise macro values for undated rows. If data
  is missing, stop and tell the student what to download.
- **Reproducible.** Fixed `random_state=42`. Record library versions in `requirements.txt`. Save
  every metric to `artefacts/metrics/*.json` and every figure to `artefacts/figures/*.png`.
- **Honest results.** Report the baseline even when the model beats it. If the macro features
  do NOT improve the forecast, report that; it is a valid answer to RQ4.
- The disclaimer "Indicative decision support only – not financial or valuation advice" must be
  shown with every result in the UI.

## Models (as specified in Chapter 4 of the dissertation)

**Sale and rent models:**
- Target: `log(price)`.
- Candidates: LinearRegression (baseline), RandomForest, XGBoost, CatBoost.
- 80/20 train/test split. 5-fold CV on train. RandomizedSearchCV for tuning. Select by the
  lowest mean CV RMSE.
- Report MAE, RMSE, MAPE and R² (in original $ units, plus R² on log scale) on the test set,
  and CV mean ± std.
- Use TreeSHAP for a global summary plot and per-prediction top-5 contributions.

**Forecast model:**
- Panel of state × quarter. Target y = 100 × (HPI[t+4] / HPI[t] − 1).
- Features: past 4-quarter and past 1-quarter growth; unemployment and its 12-month change;
  permits growth; state GDP growth; mortgage rate and its 12-month change; fed funds rate;
  CPI inflation; state (categorical).
- Model: gradient boosting.
- Baselines: (a) persistence, (b) ARIMA per state, (c) the same model without macro features.
- Evaluation: expanding-window rolling origin from 2018 onward. Report MAE, RMSE and
  directional accuracy.
- Uncertainty band: 10th/90th percentiles of out-of-sample errors.
- Rent growth: the same approach with regional CPI rent.

**Rent-vs-buy engine (pure Python, no ML, fully unit tested).** Notation: P price, R monthly
rent, g appreciation, q rent growth, d down payment, r mortgage rate, T term, τ tax,
m maintenance, h insurance, cb/cs buy/sell costs, k discount rate.

```
L = (1-d)P ; i = r/12 ; n = 12T ; M = L*i*(1+i)^n / ((1+i)^n - 1)
V_t = P(1+g)^t ; B_t = L(1+i)^(12t) - M((1+i)^(12t) - 1)/i
O_t = 12M*[t<=T] + (τ+m+h)*V_{t-1} ; Rent_t = 12R(1+q)^(t-1)
C_B(H) = (d+cb)P + Σ O_t/(1+k)^t - (V_H(1-cs) - B_H)/(1+k)^H
C_R(H) = Σ Rent_t/(1+k)^t ; Δ(H) = C_R - C_B ; break-even = min H with Δ ≥ 0 (search to 30 yrs)
User cost UC = (k+τ+m+h-g)*P, compared with 12R
```

Default assumptions: d 20%, T 30y, H 7y, τ 1.0%, m 1.0%, h 0.35%, cb 3%, cs 6%, k 5%,
r = latest mortgage_rate in D3, g and q from the forecast.

## Target folder structure

```
data/raw/  data/processed/
src/pipeline/   clean_sale.py  clean_rent.py  build_panel.py  leakage_audit.py
src/models/     train_price_models.py  train_forecast.py  explain.py
src/engine/     rent_vs_buy.py  sensitivity.py  scenario.py
src/api/        app.py  schemas.py       (keep the student's existing back-end framework)
frontend/       (keep the student's existing front end)
tests/          test_engine.py  test_api.py
artefacts/      models/  metrics/  figures/
notebooks/      eda.ipynb (optional)
```

## API contract

- `POST /api/analyse`
- `POST /api/compare`
- `POST /api/sensitivity`
- `POST /api/scenario`
- `GET /api/forecast/{state}`

Full analysis must respond in under 3 s. Load the models once at start-up.

## Outputs Chapter 5 needs (always save these)

- `artefacts/metrics/sale_models.json` and `rent_models.json`: every candidate × every metric,
  CV mean ± std, test set.
- `artefacts/metrics/forecast.json`: model vs persistence vs ARIMA vs no-macro, for HPI and rent.
- `artefacts/metrics/engine_tests.json`: pass/fail per manual test scenario.
- `artefacts/metrics/decision_flip.json`: for ≥500 test properties, the recommendation using a
  fixed 3% growth vs the forecast growth, plus the % of decisions that flip and the mean
  break-even shift. This is the key evidence for the research gap.
- `artefacts/metrics/performance.json`: API response time (mean/p95 over 100 requests).
- Figures:
  - predicted vs actual (sale, rent);
  - SHAP summary (sale, rent);
  - forecast backtest plot;
  - error by state;
  - tornado sensitivity chart;
  - cost-over-time example.

## Working style

- Work one step of `IMPLEMENTATION_PLAN.md` at a time. Show the plan before large changes.
- Explain what you did in plain English at the end of each step. The student writes the
  dissertation from these explanations.
- Ask before deleting or restructuring existing front-end/back-end code.
