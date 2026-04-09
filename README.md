# ER-Triage-Prototype

> AI-assisted ER triage prototype predicting short-term patient deterioration for dynamic prioritization.

---

## Overview

This repository contains a prototype for an AI-assisted emergency room triage system. The system predicts the probability of short-term clinical deterioration for incoming patients and dynamically prioritizes the ER queue to reduce time-to-treatment for high-risk cases.

This project is designed as a **research & prototype tool**, using publicly available data from [MIMIC-IV](https://physionet.org/content/mimiciv/), and demonstrates an end-to-end pipeline:

```
Feature Extraction → Model Training → Risk Prediction → Prioritization → Simulation
```

> ⚠️ **Disclaimer:** This tool is a research prototype only. It is not validated for clinical use and must not be used to make real patient care decisions.

---

## Features

| Module | Description |
|---|---|
| `preprocess.py` | Cleans and normalizes raw MIMIC-IV vitals and demographics |
| `features.py` | Derives engineered features (shock index, O₂ flags, etc.) |
| `labels.py` | Creates binary deterioration labels within a configurable time window |
| `train.py` | Trains an XGBoost classifier with cross-validation |
| `predict.py` | Generates per-patient risk scores from trained model |
| `prioritize.py` | Ranks the ER queue dynamically by predicted risk |
| `simulation.py` | Simulates ER throughput and evaluates model impact on time-to-treatment |
| `streamlit_app/` | Optional interactive UI for queue visualization |

---

## Project Structure

```
ER-Triage-Prototype/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/               # Raw MIMIC-IV CSV/Parquet files (not committed)
│   ├── processed/         # Cleaned & feature-engineered data
│   └── features.csv       # Final feature matrix ready for modeling
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── labels.py
│   ├── train.py
│   ├── predict.py
│   ├── prioritize.py
│   └── simulation.py
├── notebooks/
│   └── exploration.ipynb  # EDA and quick experiments
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
└── tests/
    ├── test_preprocess.py
    └── test_features.py
```

---

## Data

Data files are **not included** in this repository due to privacy and licensing regulations.

This project uses **MIMIC-IV** (Medical Information Mart for Intensive Care). To obtain access:

1. Complete the required CITI training course
2. Sign the PhysioNet data use agreement
3. Apply for access at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)

Once approved, place the relevant CSV/Parquet exports under `data/raw/`.

### Expected Input Features

The following columns are expected in the raw input (extractable from MIMIC-IV `chartevents`, `admissions`, and `patients` tables):

| Feature | Description |
|---|---|
| `heart_rate` | Beats per minute |
| `sbp` / `dbp` | Systolic / diastolic blood pressure (mmHg) |
| `resp_rate` | Respiratory rate (breaths/min) |
| `spo2` | Oxygen saturation (%) |
| `temperature` | Body temperature (°C or °F) |
| `age` | Patient age at admission |
| `sex` | Biological sex |

### Labels

The deterioration label is derived from downstream events within a configurable window (default: **6 hours**):

- ICU transfer
- Mechanical ventilation initiation
- In-hospital death

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ER-Triage-Prototype.git
cd ER-Triage-Prototype
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
# Step 1 — Preprocess raw MIMIC-IV data
python src/preprocess.py --input data/raw/ --output data/processed/

# Step 2 — Engineer features
python src/features.py --input data/processed/ --output data/features.csv

# Step 3 — Create labels
python src/labels.py --input data/processed/ --window 6

# Step 4 — Train model
python src/train.py --features data/features.csv --output models/

# Step 5 — Run simulation
python src/simulation.py --model models/xgb_model.pkl
```

### 5. Launch the Streamlit UI (optional)

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## Model

- **Algorithm:** XGBoost (`xgboost.XGBClassifier`)
- **Target:** Binary — patient deteriorates within N hours (default: 6)
- **Evaluation metrics:** AUROC, AUPRC, sensitivity @ fixed specificity
- **Validation:** Stratified k-fold cross-validation (default: k=5)

Trained model artifacts are saved to `models/` (not committed to git).

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

Key pipeline parameters can be tuned via `config.yaml` (coming soon) or passed as CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `--window` | `6` | Deterioration label window in hours |
| `--threshold` | `0.5` | Risk score cutoff for high-risk classification |
| `--n_estimators` | `200` | Number of XGBoost trees |
| `--cv_folds` | `5` | Cross-validation folds |

---

## Roadmap

- [ ] Add SHAP explainability for individual predictions
- [ ] Support time-series vitals (LSTM / temporal model variant)
- [ ] Add `config.yaml` for centralized parameter management
- [ ] Docker container for reproducible deployment
- [ ] Evaluation dashboard in Streamlit

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. All contributions must include appropriate test coverage.

---

## License

[MIT](LICENSE)

---

## Citation

If you use this project in your research, please cite the MIMIC-IV dataset:

> Johnson, A., Bulgarelli, L., Shen, L., et al. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
