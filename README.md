# EEG Cognitive State Prediction

## Person 2 to Person 3 handoff

Person 2 should save one row per EEG window to:

`data/processed/model_features.csv`

The table must contain:

- `condition`: cognitive-state label used as the prediction target
- Numeric EEG feature columns
- `subject_id` for subject-aware evaluation and personalized modeling
- Optional `session_id` and `window_id` identifiers

## Run modeling

Install dependencies with `pip install -r requirements.txt`, then open
`notebooks/03_modeling/modeling_evaluation.ipynb` and run the cells in order.
The workflow saves the selected general and personalized models in `models/`,
the comparison report in `results/reports/`, and confusion matrices in
`results/figures/`.
