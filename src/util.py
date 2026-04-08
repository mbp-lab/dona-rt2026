from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

def save_dict_as_csv(dictionary: Dict, filepath: Path, precision: int = 2) -> None:
    with open(filepath, "w") as f:
        f.writelines(
            [
                f"{key},{value}\n" if isinstance(value, int) else f"{key},{value:.{precision}f}\n"
                for key, value in dictionary.items()
            ]
        )

def get_key_index(search_str, input_dict: Dict) -> List[Optional[int]]:
    positions = []
    for i, key in enumerate(input_dict.keys()):
        if search_str in key:
            positions.append(i)
    return positions

def convert_timedelta_to_s(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for col_name in cols:
            df[col_name] = df[col_name].apply(lambda x: x.total_seconds())
            df[col_name] = df[col_name].astype(int)
        return df

def convert_to_min(df, cols):
    for col_name in cols:
        df[col_name] = df[col_name].apply(lambda x: x / 60 if x >= 60 else 1)
    return df

def generate_general_stats(df: pd.DataFrame, fp: Path = Path("./data/general.dat")) -> None:
    if not fp.exists():
        fp.parent.mkdir(parents=True, exist_ok=True)    
    timespan = (
        df.groupby("donor_id").datetime.max() - df.groupby("donor_id").datetime.min()
    ).dt.total_seconds() / (60 * 60 * 24)
    timespan_wa = (
        df[df.data_source_id == 2].groupby("donor_id").datetime.max()
        - df[df.data_source_id == 2].groupby("donor_id").datetime.min()
    ).dt.total_seconds() / (60 * 60 * 24)
    timespan_ig = (
        df[df.data_source_id == 3].groupby("donor_id").datetime.max()
        - df[df.data_source_id == 3].groupby("donor_id").datetime.min()
    ).dt.total_seconds() / (60 * 60 * 24)

    general_stats_dict = {
        "n_donations": df.donor_id.nunique(),
        "n_chats": df.conversation_id.nunique(),
        "n_messages": len(df),
        "n_donations_wa": df[df.data_source_id == 2].donor_id.nunique(),
        "n_chats_wa": df[df.data_source_id == 2].conversation_id.nunique(),
        "n_donations_ig": df[df.data_source_id == 3].donor_id.nunique(),
        "n_chats_ig": df[df.data_source_id == 3].conversation_id.nunique(),
        "mean_chats_donor": df.groupby("donor_id").conversation_id.nunique().mean(),
        "median_chats_donor": int(df.groupby("donor_id").conversation_id.nunique().median()),
        "sd_chats_donor": df.groupby("donor_id").conversation_id.nunique().std(),
        "min_chats_donor": int(df.groupby("donor_id").conversation_id.nunique().min()),
        "max_chats_donor": int(df.groupby("donor_id").conversation_id.nunique().max()),
        "mean_chats_donor_wa": df[df.data_source_id == 2]
        .groupby("donor_id")
        .conversation_id.nunique()
        .mean(),
        "median_chats_donor_wa": int(
            df[df.data_source_id == 2]
            .groupby("donor_id")
            .conversation_id.nunique()
            .median()
        ),
        "sd_chats_donor_wa": df[df.data_source_id == 2]
        .groupby("donor_id")
        .conversation_id.nunique()
        .std(),
        "min_chats_donor_wa": int(
            df[df.data_source_id == 2]
            .groupby("donor_id")
            .conversation_id.nunique()
            .min()
        ),
        "max_chats_donor_wa": int(
            df[df.data_source_id == 2]
            .groupby("donor_id")
            .conversation_id.nunique()
            .max()
        ),
        "mean_chats_donor_ig": df[df.data_source_id == 3]
        .groupby("donor_id")
        .conversation_id.nunique()
        .mean(),
        "median_chats_donor_ig": int(
            df[df.data_source_id == 3]
            .groupby("donor_id")
            .conversation_id.nunique()
            .median()
        ),
        "sd_chats_donor_ig": df[df.data_source_id == 3]
        .groupby("donor_id")
        .conversation_id.nunique()
        .std(),
        "min_chats_donor_ig": int(
            df[df.data_source_id == 3]
            .groupby("donor_id")
            .conversation_id.nunique()
            .min()
        ),
        "max_chats_donor_ig": int(
            df[df.data_source_id == 3]
            .groupby("donor_id")
            .conversation_id.nunique()
            .max()
        ),
        "mean_timespan_donor": int(timespan.mean()),
        "median_timespan_donor": int(timespan.median()),
        "sd_timespan_donor": int(timespan.std()),
        "min_timespan_donor": int(timespan.min()),
        "max_timespan_donor": int(timespan.max()),
        "mean_timespan_donor_wa": int(timespan_wa.mean()),
        "median_timespan_donor_wa": int(timespan_wa.median()),
        "sd_timespan_donor_wa": int(timespan_wa.std()),
        "min_timespan_donor_wa": int(timespan_wa.min()),
        "max_timespan_donor_wa": int(timespan_wa.max()),
        "mean_timespan_donor_ig": int(timespan_ig.mean()),
        "median_timespan_donor_ig": int(timespan_ig.median()),
        "sd_timespan_donor_ig": int(timespan_ig.std()),
        "min_timespan_donor_ig": int(timespan_ig.min()),
        "max_timespan_donor_ig": int(timespan_ig.max()),
        "mean_messages_donor": df.groupby("donor_id").size().mean(),
        "median_messages_donor": int(df.groupby("donor_id").size().median()),
        "sd_messages_donor": df.groupby("donor_id").size().std(),
        "min_messages_donor": int(df.groupby("donor_id").size().min()),
        "max_messages_donor": int(df.groupby("donor_id").size().max()),
        "mean_messages_donor_wa": df[df.data_source_id == 2]
        .groupby("donor_id")
        .size()
        .mean(),
        "median_messages_donor_wa": int(
            df[df.data_source_id == 2].groupby("donor_id").size().median()
        ),
        "sd_messages_donor_wa": df[df.data_source_id == 2]
        .groupby("donor_id")
        .size()
        .std(),
        "min_messages_donor_wa": int(
            df[df.data_source_id == 2].groupby("donor_id").size().min()
        ),
        "max_messages_donor_wa": int(
            df[df.data_source_id == 2].groupby("donor_id").size().max()
        ),
        "mean_messages_donor_ig": df[df.data_source_id == 3]
        .groupby("donor_id")
        .size()
        .mean(),
        "median_messages_donor_ig": int(
            df[df.data_source_id == 3].groupby("donor_id").size().median()
        ),
        "sd_messages_donor_ig": df[df.data_source_id == 3]
        .groupby("donor_id")
        .size()
        .std(),
        "min_messages_donor_ig": int(
            df[df.data_source_id == 3].groupby("donor_id").size().min()
        ),
        "max_messages_donor_ig": int(
            df[df.data_source_id == 3].groupby("donor_id").size().max()
        ),
    }
    save_dict_as_csv(dictionary=general_stats_dict, filepath=fp)
