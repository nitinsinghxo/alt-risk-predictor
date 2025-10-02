from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--random_state', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    data_path = os.path.join(cfg.processed_dir, 'dataset.csv')
    if not os.path.exists(data_path):
        raise SystemExit('Missing processed dataset. Run scripts/build_dataset.py')

    df = pd.read_csv(data_path, parse_dates=['date'])
    feature_cols = [c for c in df.columns if c not in {'date','ticker','rv_future','vol_bucket'} and df[c].dtype != 'O']
    X = df[feature_cols].ffill().bfill().fillna(0.0)
    y = df['vol_bucket']

    # Drop rows without labels
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]
    meta = df.loc[mask, ['date','ticker']]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=args.test_size, random_state=args.random_state, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=400, random_state=args.random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)

    out = meta_test.copy()
    out['true_bucket'] = y_test.values
    out['pred_bucket'] = pred
    for i, cls in enumerate(classes):
        out[f'proba_{cls}'] = proba[:, i]

    out_path = os.path.join(cfg.processed_dir, 'predictions_classification.csv')
    out.to_csv(out_path, index=False)
    print(f'Wrote {out_path} with shape {out.shape}')


if __name__ == '__main__':
    main()
