"""Fusion utilities."""

import pandas as pd


def format_date_col(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Format the date column in the DataFrame.

    Convert DataFrame column with date from `DD/MM/YYYY` format
    into `YYYY-MM-DD` format.
    """
    data[col_name] = pd.to_datetime(data[col_name], format="%d/%m/%Y")
    data[col_name] = data[col_name].dt.strftime("%Y-%m-%d")
    return data
