from __future__ import annotations

import os
import argparse
import pandas as pd
from datetime import datetime, timedelta

from src.utils.config import load_config
from src.data.finance_yf import download_price_history
from src.data.trends_pytrends import fetch_trends
from src.data.twitter_snscrape import fetch_tweets_for_tickers
from src.data.reddit_praw import fetch_reddit_posts
from src.features.sentiment import aggregate_daily


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', type=str, default='AAPL,MSFT,NVDA,SPY')
    p.add_argument('--start', type=str, default=(datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d'))
    p.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'))
    p.add_argument('--region', type=str, default='US')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    os.makedirs(cfg.raw_dir, exist_ok=True)

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]

    prices = download_price_history(tickers, args.start, args.end)
    prices.to_csv(os.path.join(cfg.raw_dir, 'prices.csv'))

    trends = fetch_trends(tickers, args.start, args.end, geo=args.region)
    trends.to_csv(os.path.join(cfg.raw_dir, 'trends.csv'))

    tweets = fetch_tweets_for_tickers(tickers, args.start, args.end)
    if not tweets.empty:
        tweets.to_csv(os.path.join(cfg.raw_dir, 'tweets.csv'), index=False)
        tweets_daily = aggregate_daily(tweets.rename(columns={'date':'dt', 'content': 'text'}), text_col='text', date_col='dt')
        tweets_daily.to_csv(os.path.join(cfg.raw_dir, 'tweets_daily_sentiment.csv'))

    # Reddit uses epoch seconds
    try:
        start_ts = int(datetime.strptime(args.start, '%Y-%m-%d').timestamp())
        end_ts = int(datetime.strptime(args.end, '%Y-%m-%d').timestamp())
    except Exception:
        start_ts = int((datetime.today() - timedelta(days=365)).timestamp())
        end_ts = int(datetime.today().timestamp())

    reddit_posts = fetch_reddit_posts(tickers, start_ts, end_ts)
    if not reddit_posts.empty:
        reddit_posts.to_csv(os.path.join(cfg.raw_dir, 'reddit_posts.csv'), index=False)
        reddit_daily = aggregate_daily(reddit_posts.rename(columns={'created_dt':'dt', 'title':'text'}), text_col='text', date_col='dt')
        reddit_daily.to_csv(os.path.join(cfg.raw_dir, 'reddit_daily_sentiment.csv'))


if __name__ == '__main__':
    main()
