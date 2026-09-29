# Implementation Plan

Work through these steps in order. For each step, copy the prompt into Claude Code in VS Code.
Only move on when the "Done when" check passes, then commit with Git.

---

## Step 0: Set up and inspect existing code
**Prompt:**
> Read CLAUDE.md and IMPLEMENTATION_PLAN.md. Inspect my existing front-end and back-end code
> and tell me: the frameworks used, what already works, and what is missing compared with
> CLAUDE.md. Propose how to fit the target folder structure without breaking my code. Create
> requirements.txt and a Python virtual environment setup. Don't change code yet.

**Done when:** you understand the report and have agreed the structure.

## Step 1: Data cleaning and leakage audit
**Prompt:**
> Do Step 1. Write src/pipeline/clean_sale.py and clean_rent.py using the rules in CLAUDE.md,
> saving to data/processed/. Write src/pipeline/leakage_audit.py that shows:
> (a) R² with and without PPSq on house_train_proc.csv;
> (b) the within-state correlation of the Final Dataset macro columns with log price;
> (c) R² before and after shuffling those columns within each state.
> Save the results to artefacts/metrics/leakage_audit.json and print a plain-English summary.

**Done when:**
- The sale data has about 38,000 rows and the rent data about 98,900.
- The audit shows R² of about 0.997 with PPSq, and about 0.978 → 0.84 when the macro columns
  are shuffled.

## Step 2: Exploratory data analysis
**Prompt:**
> Do Step 2. Create distribution, correlation and price-by-state figures for both datasets in
> artefacts/figures/ and a short summary table (count, mean, median, std) as JSON.

## Step 3: Sale price model
**Prompt:**
> Do Step 3. Write src/models/train_price_models.py for the sale target: Linear baseline, RF,
> XGBoost and CatBoost, with an 80/20 split, 5-fold CV and RandomizedSearchCV (n_iter≈30).
> Save the best model to artefacts/models/, the metrics to sale_models.json, a predicted-vs-actual
> plot and a SHAP summary plot.

**Done when:** the best model beats the baseline, with R² ≥ 0.80 on log price (a preliminary run
gave about 0.84).

## Step 4: Rent model
**Prompt:**
> Do Step 4. Same as Step 3 for the rent data (features aligned per CLAUDE.md), saving
> rent_models.json and figures.

## Step 5: Forecast model (needs D4 and D5 downloaded)
**Prompt:**
> Do Step 5. Build the state×quarter panel from fred_final.csv + FHFA state HPI, and a region×month
> panel with BLS CPI rent. Train the forecast model and baselines (persistence, ARIMA, no-macro)
> with expanding-window rolling origin from 2018. Save forecast.json, a backtest plot and the
> latest forecast per state with 10th/90th percentile bands to artefacts/models/forecasts.json.
> Also save the CPI rent ratio needed to inflate the 2019 rents.

**Done when:** you know honestly whether the macro features beat persistence (either answer is
fine).

## Step 6: Rent-vs-buy engine and tests
**Prompt:**
> Do Step 6. Implement src/engine/rent_vs_buy.py exactly as the formulas in CLAUDE.md, plus
> sensitivity.py (tornado data) and scenario.py (apply macro shocks → re-forecast → recompute).
> Write tests/test_engine.py with at least 8 hand-calculated scenarios (include zero growth,
> 100% down payment, H > T). Also export those scenarios to an Excel sheet so I can verify them
> manually. Save engine_tests.json.

**Done when:** all tests pass and the spreadsheet matches.

## Step 7: Decision-flip experiment (key evidence for the gap)
**Prompt:**
> Do Step 7. For ≥500 properties from the sale test set, compute the recommendation and
> break-even with (a) a fixed 3% appreciation and 3% rent growth, and (b) the forecast values.
> Report the % of decisions that flip, the mean break-even shift, and results by state. Save
> decision_flip.json and a chart.

## Step 8: API integration
**Prompt:**
> Do Step 8. Connect everything to my existing back end using the API contract in CLAUDE.md
> (ZIP → state/lat/long/income/density lookup, with a state-median fallback). Add input
> validation, the disclaimer, and tests/test_api.py. Measure response time over 100 requests →
> performance.json.

## Step 9: Front end
**Prompt:**
> Do Step 9. Update my front end to show: an input form; result cards (price, rent, forecasts
> with ranges); SHAP bar charts; assumption sliders; a cost-over-time chart; the break-even
> year and recommendation; sensitivity and scenario tabs; and the disclaimer on every result.

## Step 10: Final check and Chapter 5 pack
**Prompt:**
> Do Step 10. Run the full pipeline from scratch and all tests. Then write RESULTS_SUMMARY.md
> listing every metric, figure path and finding, mapped to objectives O1–O6 and the Chapter 5
> headings (Evaluation Criteria, Test Results, System Performance, Technical Measures,
> Comparison with Existing Solutions, Results Related to Objectives).

**Then:** send RESULTS_SUMMARY.md, the metrics JSON files and the figures back to me, and I'll
write Chapter 5.

User evaluation (SUS + TAM questionnaire with 20–30 users) happens after Step 9. Ask me for the
questionnaire when you're ready.
