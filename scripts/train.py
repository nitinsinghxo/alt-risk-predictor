from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, choices=['regression','classification'], default='regression')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    return p.parse_args()


def build_xy(df: pd.DataFrame, task: str):
    y = df['rv_future'] if task == 'regression' else df['vol_bucket']
    feature_cols = [c for c in df.columns if c not in {'date','ticker','rv_future','vol_bucket'} and df[c].dtype != 'O']
    X = df[feature_cols].ffill().bfill().fillna(0.0)
    if task == 'classification':
        y = y.dropna()
        X = X.loc[y.index]
    else:
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]
    return X, y


def main():
    args = parse_args()
    cfg = load_config()
    data_path = os.path.join(cfg.processed_dir, 'dataset.csv')
    if not os.path.exists(data_path):
        raise SystemExit('Missing processed dataset. Run scripts/build_dataset.py')

    df = pd.read_csv(data_path, parse_dates=['date'])

    X, y = build_xy(df, args.task)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, shuffle=False)

    if args.task == 'regression':
        models = {
            'rf': RandomForestRegressor(n_estimators=400, max_depth=None, random_state=args.random_state, n_jobs=-1),
            'gbr': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            # Compute RMSE without relying on 'squared' kwarg for compatibility
            mse = mean_squared_error(y_test, pred)
            rmse = float(np.sqrt(mse))
            print(f'{name} RMSE: {rmse:.4f}')
            os.makedirs(os.path.join(cfg.processed_dir, 'models'), exist_ok=True)
            model_path = os.path.join(cfg.processed_dir, 'models', f'{name}_reg.pkl')
            try:
                import joblib
                joblib.dump(model, model_path)
            except Exception:
                pass
    else:
        models = {
            'rf': RandomForestClassifier(n_estimators=400, max_depth=None, random_state=args.random_state, n_jobs=-1),
            'gbc': GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            f1 = f1_score(y_test, pred, average='weighted')
            print(f'{name} F1: {f1:.4f}')
            print(classification_report(y_test, pred))
            os.makedirs(os.path.join(cfg.processed_dir, 'models'), exist_ok=True)
            model_path = os.path.join(cfg.processed_dir, 'models', f'{name}_clf.pkl')
            try:
                import joblib
                joblib.dump(model, model_path)
            except Exception:
                pass


if __name__ == '__main__':
    main()
