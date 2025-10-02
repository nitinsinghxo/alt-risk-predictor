from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    data_dir: str
    raw_dir: str
    processed_dir: str
    region: str = "US"
    language: str = "en"

    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None


def load_config(project_root: str | None = None) -> AppConfig:
    load_dotenv()
    root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(root, "data")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")

    return AppConfig(
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        region=os.getenv("REGION", "US"),
        language=os.getenv("LANGUAGE", "en"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT"),
    )
