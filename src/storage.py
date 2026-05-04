from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_raw_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
    return output_path


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def raw_json_path(kind: str, start_date: str, end_date: str, raw_dir: str | Path) -> Path:
    return Path(raw_dir) / f"{kind}_history_{start_date}_{end_date}.json"


def processed_csv_path(kind: str, start_date: str, end_date: str, processed_dir: str | Path) -> Path:
    return Path(processed_dir) / f"{kind}_history_{start_date}_{end_date}.csv"

