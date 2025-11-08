#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
):
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame")

    # 70% train, 15% val, 15% test (split via 70/30 -> 50/50)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[label_col],
        random_state=random_state
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[label_col],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
