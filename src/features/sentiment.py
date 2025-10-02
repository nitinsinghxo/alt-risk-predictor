from __future__ import annotations

from typing import Iterable, Dict
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VSent

try:
    from transformers import pipeline
    _has_transformers = True
except Exception:
    _has_transformers = False


def vader_scores(texts: Iterable[str]) -> pd.DataFrame:
    sid = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        s = sid.polarity_scores(t or '')
        rows.append(s)
    return pd.DataFrame(rows)


def finbert_scores(texts: Iterable[str]) -> pd.DataFrame:
    if not _has_transformers:
        return pd.DataFrame()
    clf = pipeline('text-classification', model='ProsusAI/finbert', top_k=None, truncation=True)
    rows = []
    for t in texts:
        preds = clf(t or '')
        # preds is a list of dicts with labels POSITIVE/NEGATIVE/NEUTRAL
        label_to_score = {p['label'].lower(): float(p['score']) for p in preds}
        rows.append({
            'finbert_positive': label_to_score.get('positive', 0.0),
            'finbert_negative': label_to_score.get('negative', 0.0),
            'finbert_neutral': label_to_score.get('neutral', 0.0),
        })
    return pd.DataFrame(rows)


def aggregate_daily(df: pd.DataFrame, text_col: str, date_col: str) -> pd.DataFrame:
    vdf = vader_scores(df[text_col].fillna(''))
    vdf.index = df.index
    out = pd.concat([df[[date_col]], vdf], axis=1)
    daily = out.groupby(out[date_col].dt.date).mean(numeric_only=True)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    if _has_transformers:
        fdf = finbert_scores(df[text_col].fillna(''))
        fdf.index = df.index
        out2 = pd.concat([df[[date_col]], fdf], axis=1)
        daily2 = out2.groupby(out2[date_col].dt.date).mean(numeric_only=True)
        daily2.index = pd.to_datetime(daily2.index)
        daily = daily.join(daily2, how='outer')
    return daily
