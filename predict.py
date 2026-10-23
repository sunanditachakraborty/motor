"""Batch or single-record inference using the exact shared feature preparation.

Batch:   python predict.py --input new_motor_readings.csv --output predictions.csv
Single:  python predict.py --json '{"ambient":25,"coolant":30,"u_d":-1,"u_q":100,"motor_speed":2000,"i_d":-2,"i_q":10}'
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from preprocessing import prepare_features


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}. Run train.py first.")
    bundle = joblib.load(path)
    return bundle['model'], bundle['features']


def predict_frame(df: pd.DataFrame, model, expected_features):
    X, _, _ = prepare_features(df, training=False)
    # Ensure inference schema is aligned to the exact training order.
    X = X.reindex(columns=expected_features)
    predictions = model.predict(X)
    result = df.copy()
    result['predicted_pm_temperature_C'] = predictions
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='artifacts/motor_temperature_model.joblib')
    p.add_argument('--input', help='Input CSV containing the seven required predictor columns')
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--json', help='Single JSON object containing predictor values')
    args = p.parse_args()
    if bool(args.input) == bool(args.json):
        p.error('Provide exactly one of --input or --json.')
    model, features = load_model(Path(args.model))
    if args.json:
        frame = pd.DataFrame([json.loads(args.json)])
        result = predict_frame(frame, model, features)
        print(result[['predicted_pm_temperature_C']].to_string(index=False))
    else:
        frame = pd.read_csv(args.input)
        result = predict_frame(frame, model, features)
        result.to_csv(args.output, index=False)
        print(f"Predicted {len(result)} record(s). Results saved to {Path(args.output).resolve()}")

if __name__ == '__main__':
    main()
