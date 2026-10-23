# PMSM Permanent Magnet Temperature Prediction

Predicts permanent-magnet tooth temperature (`pm`, °C) from electric motor sensor measurements in the SQLite `electric_motor_temperature` table.

## Files
- `train.py`: loads the SQLite table, trains and evaluates the model, and saves artifacts.
- `predict.py`: batch CSV or single-record predictions using the saved model.
- `preprocessing.py`: shared column normalization and feature selection used by both training and inference.
- `report.docx`: project report.

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `Regression.db` in this directory, or pass its path explicitly. The database must contain the `electric_motor_temperature` table.

## Train and evaluate
```bash
python train.py --db Regression.db
```
The script creates `artifacts/motor_temperature_model.joblib`, `artifacts/evaluation_metrics.json`, and `artifacts/feature_importance.csv`.

## Batch prediction
Create a CSV with these columns: `ambient,coolant,u_d,u_q,motor_speed,i_d,i_q`.
```bash
python predict.py --input new_motor_readings.csv --output predictions.csv
```

## Single prediction
```bash
python predict.py --json '{"ambient":25,"coolant":30,"u_d":-1,"u_q":100,"motor_speed":2000,"i_d":-2,"i_q":10}'
```

## Model and validation
The model is a scikit-learn `HistGradientBoostingRegressor` inside a pipeline with median imputation. Evaluation reports MAE (°C), RMSE (°C), and R². If `profile_id` has enough distinct groups, the holdout is split by profile to reduce leakage between measurements from the same session. Permutation importance is computed on a capped sample of the held-out set and saved to CSV.

The project ZIP supplied for implementation did not include `Regression.db`, so no trained metrics or empirical feature rankings are bundled. Run training against the real database to generate them. Feature importance must be interpreted as predictive association, not causation.
