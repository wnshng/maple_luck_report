from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .nexon_client import NexonMapleClient
from .preprocess import (
    normalize_cube_history,
    normalize_potential_history,
    normalize_starforce_history,
    prepare_uploaded_dataframe,
)
from .storage import processed_csv_path, raw_json_path, save_dataframe_csv, save_raw_json


@dataclass(frozen=True)
class LoadedHistory:
    raw_records: list[dict]
    dataframe: pd.DataFrame
    raw_path: Path
    csv_path: Path


def fetch_cube_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    records = client.get_all_cube_history(start_date, end_date)
    df = normalize_cube_history(records)
    raw_path = raw_json_path("cube", start_date, end_date, RAW_DATA_DIR)
    csv_path = processed_csv_path("cube", start_date, end_date, PROCESSED_DATA_DIR)
    save_raw_json(records, raw_path)
    save_dataframe_csv(df, csv_path)
    return LoadedHistory(records, df, raw_path, csv_path)


def fetch_starforce_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    records = client.get_all_starforce_history_by_date_range(start_date, end_date)
    df = normalize_starforce_history(records)
    raw_path = raw_json_path("starforce", start_date, end_date, RAW_DATA_DIR)
    csv_path = processed_csv_path("starforce", start_date, end_date, PROCESSED_DATA_DIR)
    save_raw_json(records, raw_path)
    save_dataframe_csv(df, csv_path)
    return LoadedHistory(records, df, raw_path, csv_path)


def fetch_potential_history_dataframe(
    client: NexonMapleClient,
    start_date: str,
    end_date: str,
) -> LoadedHistory:
    records = client.get_all_potential_history(start_date, end_date)
    df = normalize_potential_history(records)
    raw_path = raw_json_path("potential", start_date, end_date, RAW_DATA_DIR)
    csv_path = processed_csv_path("potential", start_date, end_date, PROCESSED_DATA_DIR)
    save_raw_json(records, raw_path)
    save_dataframe_csv(df, csv_path)
    return LoadedHistory(records, df, raw_path, csv_path)


def read_uploaded_csv(uploaded_file, kind: str) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return prepare_uploaded_dataframe(df, kind)
