# TTC Subway Delay Predictor

A machine learning project that predicts whether a reported TTC subway incident will cause an actual service delay, using real, public Toronto Transit Commission data.

## The question this project answers

Given basic information about a subway incident — which station, which line, direction, time of day, day of week, and the delay reason code logged by staff — can we predict whether it will result in a measurable delay before we know how long that delay will be?

This is framed as a **binary classification** problem: every incident is labeled `had_delay = 1` (caused a delay) or `had_delay = 0` (logged, but no measurable delay resulted).

## Why this matters

About 68.5% of logged subway incidents in the raw data caused zero minutes of measurable delay — they were recorded events (a door fault, a passenger assistance call, etc.) that didn't end up disrupting service. Knowing in advance which *types* of incidents are likely to escalate into real delays is useful for transit operations planning and rider communication.

## Data source

[TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/) — City of Toronto Open Data Portal, published by the Toronto Transit Commission. This project uses a historical extract of the same public dataset (2014–2020) sourced via a mirrored copy, since the live portal was unreachable from the build environment used for this project. See `data/_source_README.md` for provenance notes.

## Project structure

```
transit-delay-predictor/
├── data/                          # raw, cleaned, and processed data + outputs
├── src/
│   ├── 01_clean_data.py           # cleans raw data, defines the target label
│   ├── 02_explore_data.py         # exploratory charts (delay rate by hour/day/line)
│   ├── 03_encode_features.py      # one-hot encodes categorical features
│   ├── 04_train_model.py          # train/test split + trains 2 models
│   └── 05_evaluate_model.py       # evaluates both models, generates charts
└── README.md
```

Run the scripts in order (01 → 05) from the project root; each one reads the previous step's output from `data/`.

## Methodology

1. **Cleaning**: parsed dates/times into `hour` and `month` features, filled missing `bound` (direction) values as `"Unknown"` rather than dropping them (missingness turned out to be informative — see Results), dropped a small number of rows with missing/unparseable critical fields, and removed delay values above 180 minutes as likely data-entry errors.
2. **Target leakage avoidance**: `delay_minutes` (the value used to derive the label) was deliberately excluded from the model's input features, since including it would let the model trivially "cheat" by reading the answer instead of learning a pattern.
3. **Feature encoding**: categorical columns (`station`, `line`, `bound`, `day_of_week`, `delay_code`) were one-hot encoded. High-cardinality columns (`station`: 627 raw values, `delay_code`: 217 raw values) were bucketed to their most frequent categories plus an `"OTHER"` catch-all, to keep the feature space manageable. The `line` column was also standardized to fix inconsistent data entry (typos like `"BD LINE"`, `"YU/BD"`, stray bus-route numbers).
4. **Modeling**: an 80/20 train/test split, stratified to preserve the ~31.5%/68.5% class balance in both sets. Two models were trained and compared:
   - Logistic Regression (simple, interpretable baseline)
   - Random Forest (200 trees, class-balanced, to handle the class imbalance)
5. **Evaluation**: accuracy, precision, recall, F1, and confusion matrices — accuracy alone is misleading here since a model that always predicts "no delay" would already score 68.4% accuracy while catching zero real delays.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.850 | 0.715 | 0.875 | 0.786 |
| Random Forest | 0.836 | 0.681 | 0.907 | 0.778 |
| *Baseline (always predict "no delay")* | *0.684* | *–* | *0.000* | *–* |

Both models substantially beat the naive baseline in a way that's actually useful: the Random Forest correctly flags about 91% of real delays in advance (recall), at the cost of a somewhat higher false-alarm rate than Logistic Regression.

**Most predictive features** (Random Forest feature importance): whether the `bound` (direction) field was left blank by staff was the single strongest signal, followed by several specific delay reason codes (`MUSC`, `TUSC`, `MUIS`, and others) — suggesting *why* an incident occurred is a stronger predictor of escalation to a real delay than *when* or *where* it occurred.

## What I learned building this

- Real-world data requires deliberate decisions about missing values and ambiguous target definitions before any modeling can start — over two-thirds of the raw "delay" records had zero actual delay minutes, which reframed the whole problem.
- Data leakage is easy to introduce by accident (e.g. leaving a feature in that was used to derive the label) and can produce deceptively perfect-looking results.
- Accuracy is a poor standalone metric for imbalanced classification problems; precision and recall tell a more honest, decision-relevant story.
- One-hot encoding scales poorly with high-cardinality categorical columns, and bucketing rare categories is a standard, practical mitigation.

## Tools used

Python, pandas, scikit-learn, matplotlib.
