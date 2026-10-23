"""Train and evaluate a PMSM permanent-magnet temperature regression model.

Example (Windows): python train.py --db Regression.db
"""
from __future__ import annotations
import argparse, json, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from preprocessing import prepare_features, normalize_columns, FEATURE_CANDIDATES, TARGET


def load_table(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Place Regression.db beside train.py or pass --db.")
    with sqlite3.connect(str(db_path)) as con:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
        if table not in tables:
            raise ValueError(f"Table '{table}' not found in {db_path}. Available tables: {tables}")
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='Regression.db', help='Path to SQLite Regression.db')
    parser.add_argument('--table', default='electric_motor_temperature')
    parser.add_argument('--output-dir', default='artifacts')
    parser.add_argument('--test-size', type=float, default=0.20)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--importance-sample', type=int, default=10000)
    args = parser.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    raw = load_table(Path(args.db), args.table)
    normalized = normalize_columns(raw)
    X, y, meta = prepare_features(normalized, training=True)
    # Remove rows with no predictor information at all. Partial missing values are imputed in pipeline.
    usable = ~X.isna().all(axis=1)
    X, y = X.loc[usable], y.loc[usable]
    if len(X) < 20:
        raise ValueError(f"Only {len(X)} usable records found; at least 20 are needed for a meaningful split.")

    # Hold out entire profiles when possible, reducing optimistic leakage between adjacent measurements.
    groups = None
    if 'profile_id' in normalized.columns:
        candidate_groups = normalized.loc[X.index, 'profile_id']
        if candidate_groups.nunique(dropna=True) >= 5:
            groups = candidate_groups.fillna('__missing_profile__').astype(str)
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_method = 'GroupShuffleSplit by profile_id'
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        split_method = 'Random train/test split (profile_id unavailable or insufficient unique groups)'

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('regressor', HistGradientBoostingRegressor(
            loss='squared_error', learning_rate=0.08, max_iter=250,
            max_leaf_nodes=31, l2_regularization=0.1, random_state=args.random_state,
            early_stopping=True, validation_fraction=0.1
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'target': TARGET,
        'model': 'HistGradientBoostingRegressor (median imputation pipeline)',
        'database': str(Path(args.db)), 'table': args.table,
        'rows_loaded': int(len(raw)), 'rows_used': int(len(X)),
        'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
        'split_method': split_method,
        'features': FEATURE_CANDIDATES,
        'MAE_C': float(mean_absolute_error(y_test, pred)),
        'RMSE_C': float(np.sqrt(mean_squared_error(y_test, pred))),
        'R2': float(r2_score(y_test, pred)),
        'training_seconds': round(time.time() - start, 2),
    }
    joblib.dump({'model': model, 'features': FEATURE_CANDIDATES, 'target': TARGET}, outdir / 'motor_temperature_model.joblib')
    (outdir / 'evaluation_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    # Permutation importance on a capped held-out sample keeps runtime practical for 1.33M rows.
    n_imp = min(args.importance_sample, len(X_test))
    X_imp = X_test.sample(n=n_imp, random_state=args.random_state) if n_imp < len(X_test) else X_test
    y_imp = y_test.loc[X_imp.index]
    imp = permutation_importance(model, X_imp, y_imp, n_repeats=5,
                                 random_state=args.random_state, scoring='neg_mean_absolute_error', n_jobs=-1)
    importance = pd.DataFrame({
        'feature': FEATURE_CANDIDATES,
        'importance_mean_MAE_increase': imp.importances_mean,
        'importance_std': imp.importances_std,
    }).sort_values('importance_mean_MAE_increase', ascending=False)
    importance.to_csv(outdir / 'feature_importance.csv', index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and reports to: {outdir.resolve()}")
    print("Feature importance (higher MAE increase means more important):")
    print(importance.to_string(index=False))

if __name__ == '__main__':
    main()
