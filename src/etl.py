from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from src.util import convert_timedelta_to_s, convert_to_min

Keep = Literal["first", "last"]


def add_response_time(df):
    # Ensure sorted
    df = df.sort_values(by="datetime").reset_index(drop=True)
    df["rt"] = pd.Series([pd.NaT] * len(df), dtype="timedelta64[ns]")

    # Work per conversation
    for conv_id, group in df.groupby("conversation_id"):
        sender_ids = group["sender_id"].values
        datetimes = group["datetime"].values
        indices = group.index.values

        last_time_by_sender = {}

        for i in range(len(group)):
            sender = sender_ids[i]

            # Candidates are last messages from *other* senders
            other_senders = [s for s in last_time_by_sender if s != sender]

            if other_senders:
                # Get the most recent message from other senders
                last_other_times = [last_time_by_sender[s] for s in other_senders]
                most_recent_time = max(last_other_times)
                df.at[indices[i], "rt"] = datetimes[i] - most_recent_time

            # Update last message time for current sender
            last_time_by_sender[sender] = datetimes[i]

    df["rt"] = pd.to_timedelta(df["rt"], errors="raise")
    return df


def add_sender(df):
    df["sender"] = "alter"
    df.loc[df.sender_id == df.donor_id, "sender"] = "ego"
    return df


def add_rt_ego(df: pd.DataFrame) -> pd.DataFrame:
    df["rt_ego"] = pd.Series(
        [pd.NaT] * len(df), dtype="timedelta64[ns]"
    )  # Circumvent broadcasting rules not allowing for timedelta dtype NaTs
    df.loc[df.sender_id == df.donor_id, "rt_ego"] = df.rt
    return df


def add_rt_prev_alter(df):
    df["rt_prev_alter"] = pd.Series([pd.NaT] * len(df), dtype="timedelta64[ns]")

    def process_group(group):
        # Create a mask for the start of each alter block
        alter_start_mask = (group["sender"] == "alter") & (
            (group["sender"].shift(1) != "alter") | (group["sender"].shift(1).isna())
        )
        # Create a column with the response time of the first alter message in each block
        group["alter_block_start_time"] = group.loc[alter_start_mask, "rt"]
        # Forward fill the alter block start times
        group["alter_block_start_time"] = group["alter_block_start_time"].ffill()
        # Assign the response time to the appropriate ego messages
        group.loc[group["sender"] == "ego", "rt_prev_alter"] = group["alter_block_start_time"]
        # Drop the temporary column
        group = group.drop(columns=["alter_block_start_time"])
        return group

    df = df.groupby("conversation_id").apply(process_group).reset_index(drop=True)
    return df


def add_wc_prev_alter(df):
    df["wc_prev_alter"] = None

    def process_group(group):
        # Create a mask for the start of each alter block
        alter_start_mask = (group["sender"] == "alter") & (
            (group["sender"].shift(1) != "alter") | (group["sender"].shift(1).isna())
        )
        # Sum the word counts for each alter block
        group["alter_block_id"] = alter_start_mask.cumsum()
        block_word_counts = (
            group[group["sender"] == "alter"].groupby("alter_block_id")["word_count"].sum()
        )
        # Map the summed word counts back to the group
        group["alter_block_word_count"] = (
            group["alter_block_id"].map(block_word_counts).ffill().infer_objects(copy=False)
        )

        # Assign the summed word counts to the appropriate ego messages
        group.loc[group["sender"] == "ego", "wc_prev_alter"] = group["alter_block_word_count"]

        # Drop the temporary columns
        group = group.drop(columns=["alter_block_id", "alter_block_word_count"])

        return group

    df = df.groupby("conversation_id").apply(process_group).reset_index(drop=True)
    return df


def add_time_of_day(df: pd.DataFrame, date_column: str = "datetime") -> pd.DataFrame:
    df["hour"] = df[date_column].dt.hour
    return df


def add_day_of_week(df: pd.DataFrame, date_column: str = "datetime") -> pd.DataFrame:
    df["weekday"] = df[date_column].dt.day_of_week
    return df


def merge_consecutive_sender_blocks(
    df: pd.DataFrame,
    *,
    keep: Keep = "last",
) -> pd.DataFrame:
    """
    Merge consecutive messages (within each conversation) written by the same sender.

    For each maximal block of consecutive rows (ordered by time) with identical
    sender_id within the same conversation_id, this function:
    - sums word_count across the block
    - aggregates all other columns by taking their first/last value (controlled by `keep`)

    Parameters
    ----------
    df:
        DataFrame with columns: conversation_id, sender_id, word_count, rt, datetime,
        plus any number of additional columns.
    keep:
        Whether to keep the "first" or "last" value of each column within a block
        (except word_count, which is always summed).

    Returns
    -------
    pd.DataFrame:
        One row per consecutive sender block, containing all input columns.
    """
    required = {"conversation_id", "sender_id", "word_count", "rt", "datetime"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")

    # If datetimes are tied, their relative order is unspecified and does not matter.
    d = df.sort_values(["conversation_id", "datetime"], kind="quicksort").copy()

    # Identify starts of new blocks: new conversation or sender changes vs previous row.
    prev_conv = d.conversation_id.shift(1)
    prev_sender = d.sender_id.shift(1)
    new_block = (d.conversation_id.ne(prev_conv)) | (d.sender_id.ne(prev_sender))

    # Block id across the whole frame (unique per block).
    # Use assign to guarantee a real column (attribute access can be brittle).
    d = d.assign(_block_id=new_block.cumsum())

    # Build aggregation dict dynamically
    agg_func = "first" if keep == "first" else "last"

    agg_dict = {}
    for col in d.columns:
        if col == "_block_id":
            continue
        elif col == "word_count":
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = agg_func

    out = d.groupby("_block_id", as_index=False).agg(agg_dict).drop(columns=["_block_id"])

    return out


def filter_unidirectional_chats(df: pd.DataFrame) -> pd.DataFrame:
    prev_len = len(df)
    size_per_id = df.groupby("conversation_id").conversation_id.size()
    ego_size_id = df[df.sender_id == df.donor_id].groupby("conversation_id").conversation_id.size()
    merged_df = pd.merge(
        size_per_id, ego_size_id, how="inner", left_index=True, right_index=True
    )  # Inner join filters out chats w/o ego messages already

    merged_df["fraction_ego_msg"] = merged_df.conversation_id_y.div(
        merged_df.conversation_id_x.replace(0, np.nan)
    )
    conv_ids_to_keep = (merged_df["fraction_ego_msg"] >= 0.1) & (
        merged_df["fraction_ego_msg"] <= 0.9
    )
    res_df = df[df["conversation_id"].isin(merged_df[conv_ids_to_keep].index)]
    print(
        f"Unidirectional chats filter: Conversation_ids to retain: "
        f"{len(merged_df[conv_ids_to_keep])}/{len(merged_df)}. This removes "
        f"{prev_len - len(res_df)}/{prev_len} messages ({(1 - len(res_df) / prev_len) * 100:.2f}%)"
    )
    return res_df


def aggregate(
    df: pd.DataFrame,
    agg_level: str,
    agg_type: Literal["mean", "median"],
    agg_period: Optional[int],
) -> pd.DataFrame:
    if agg_type == "mean":
        df = (
            df.groupby(agg_level)[
                [
                    "word_count",
                    "rt_prev_alter",
                    "wc_prev_alter",
                    "rt_ego",
                ]
            ]
            .mean()
            .reset_index()
        )
    elif agg_type == "median":
        df = (
            df.groupby(agg_level)[
                [
                    "word_count",
                    "rt_ego",
                    "rt_prev_alter",
                    "wc_prev_alter",
                ]
            ]
            .median()
            .reset_index()
        )
    elif agg_type == "probability":
        # Define a custom aggregation function for rt_ego and rt_alter
        def fraction_leq_period(series):
            return (series <= agg_period).mean()

        df = (
            df.groupby(agg_level)[
                ["word_count", "rt_prev_alter", "wc_prev_alter", "rt_ego", "data_source_id"]
            ]
            .agg(
                {
                    "word_count": np.median,
                    "rt_prev_alter": lambda x: fraction_leq_period(x),
                    "wc_prev_alter": np.median,
                    "rt_ego": lambda x: fraction_leq_period(x),
                    "data_source_id": np.median,
                }
            )
            .reset_index()
        )
    else:
        raise NotImplementedError(
            f"aggregate_conversations: agg_type must be one of "
            f"['mean', 'median', 'probability'] but was {agg_type}."
        )
    return df


def add_columns(df):
    res_df = df.copy()
    res_df.sort_values(by="datetime", inplace=True)
    # Set sender column to ego and alter
    res_df = add_sender(res_df)

    # Add response time ("rt") to most recent message by another sender
    res_df = add_response_time(res_df)

    # Set ego's response time
    res_df = add_rt_ego(res_df)

    # For each ego line, add the response time of the previous alter message
    res_df = add_rt_prev_alter(res_df)

    # For each ego line, sum up the words of the preceding block of alter message(s)
    res_df = add_wc_prev_alter(res_df)

    # Add time of day & day of week
    res_df = add_day_of_week(add_time_of_day(res_df))

    return res_df


def _transform_filtering(experiment_config, df: pd.DataFrame) -> pd.DataFrame:
    ### Since some filtering steps need to be done in between transformation steps
    ### Filter data before the scale might be changed
    if experiment_config.rt_minimum_threshold:
        df_size_before = df.count()[0]
        thresh = experiment_config.rt_minimum_threshold
        if "rt_ego" in df.columns and "rt_prev_alter" in df.columns:
            print(
                f"Percentage of ego rt >= {thresh}: {(1 - df[df.rt_ego >= thresh].count()[0] / df_size_before) * 100}"
            )
            print(
                f"Percentage of alter rt >= {thresh}: {(1 - df[df.rt_prev_alter >= thresh].count()[0] / df_size_before) * 100}"
            )
            df = df[(df.rt_ego >= thresh) & (df.rt_prev_alter >= thresh)]
            percentage_removed = (1 - df.count()[0] / df_size_before) * 100
            print(
                f"Response time requirement (lower threshold) of rt >= {thresh} filters {df.count()[0]}/{df_size_before}"
                f"({percentage_removed:.2f}%) of messages"
            )
        elif "rt" in df.columns:
            print(
                f"Percentage of rt >= {thresh}: {(1 - df[df.rt >= thresh].count()[0] / df_size_before) * 100}"
            )
            df = df[df.rt >= thresh]
            percentage_removed = (1 - df.count()[0] / df_size_before) * 100
            print(
                f"Response time requirement (lower threshold) of rt >= {thresh} filters {df.count()[0]}/{df_size_before}"
                f"({percentage_removed:.2f}%) of messages"
            )
    if experiment_config.conversation_gap.rt_max_gap_filter:
        df_size_before = df.count()[0]
        thresh = experiment_config.conversation_gap.rt_max_gap_filter
        if "rt_ego" in df.columns and "rt_prev_alter" in df.columns:
            max_rt_per_conversation = (
                df.groupby("conversation_id")[["rt_ego", "rt_prev_alter"]].max().max(axis=1)
            )
        else:
            max_rt_per_conversation = df.groupby("conversation_id")[["rt"]].max().max(axis=1)
        valid_conversations = max_rt_per_conversation[max_rt_per_conversation <= thresh].index
        df = df[df["conversation_id"].isin(valid_conversations)]
        percentage_removed = (1 - df.count()[0] / df_size_before) * 100
        print(
            f"Largest conversation gap of rt <= {thresh} filters {df_size_before - df.count()[0]}/{df_size_before}"
            f"({percentage_removed:.2f}%) of messages"
        )
    if experiment_config.conversation_min_wc_threshold:
        ### Reporting
        prev_len = len(df)
        prev_n_convs = df.conversation_id.nunique()
        prev_donors = df.donor_id.unique()

        ### Calculation
        thresh = experiment_config.conversation_min_wc_threshold
        df["total_wc"] = df.word_count + df.wc_prev_alter
        conv_ids = df.groupby("conversation_id")["total_wc"].sum() >= thresh
        df = df[df.conversation_id.isin(conv_ids[conv_ids].index)]
        df.drop(columns="total_wc", inplace=True)

        ### Reporting
        post_len = len(df)
        post_n_convs = df.conversation_id.nunique()
        post_donors = df.donor_id.unique()
        print(
            f"A WC limit of >= {thresh} words per chat removes {prev_len - post_len}/{prev_len} "
            f"rows ({(prev_len - post_len) / prev_len * 100:.2f}%) and "
            f"{(prev_n_convs - post_n_convs)}/{prev_n_convs} convs."
        )
        print(f"Removed donor IDs: {set(prev_donors).difference(set(post_donors))}")

    if experiment_config.rt_threshold:
        df_size_before = df.count()[0]
        thresh = experiment_config.rt_threshold
        if "rt_ego" in df.columns and "rt_prev_alter" in df.columns:
            df = df[(df.rt_ego <= thresh) & (df.rt_prev_alter <= thresh)]
        else:
            df = df[df.rt <= thresh]
        percentage_removed = (1 - df.count()[0] / df_size_before) * 100
        print(
            f"Response time requirement (upper threshold) of rt <= {thresh} filters {df_size_before - df.count()[0]}/{df_size_before}"
            f"({percentage_removed:.2f}%) of messages"
        )
    if experiment_config.wc_threshold:
        df_size_before = df.count()[0]
        mode = experiment_config.wc_threshold[0]
        thresh = experiment_config.wc_threshold[1]
        if mode == "remove":
            df = df[(df.word_count <= thresh) & (df.wc_prev_alter <= thresh)]
            percentage_removed = (1 - df.count()[0] / df_size_before) * 100
            print(
                f"Word count requirement (upper threshold) of wc <= {thresh} filters {df_size_before - df.count()[0]}/{df_size_before}"
                f"({percentage_removed:.2f}%) of messages"
            )
        elif mode == "impute_median":
            ego_wc_median = df.groupby("conversation_id")["word_count"].transform("median")
            alter_wc_median = df.groupby("conversation_id")["wc_prev_alter"].transform("median")

            num_ego_imputed = (df.word_count > thresh).sum()
            num_alter_imputed = (df.wc_prev_alter > thresh).sum()

            df.loc[df.word_count > thresh, "word_count"] = ego_wc_median
            df.loc[df.wc_prev_alter > thresh, "wc_prev_alter"] = alter_wc_median

            print(
                f"Word count values above {thresh} imputed with the conversation-wise median. "
                f"Rows affected: word_count={num_ego_imputed}, wc_prev_alter={num_alter_imputed}."
            )
    if experiment_config.exclude_donations_lt_n_chats:
        thresh = experiment_config.exclude_donations_lt_n_chats
        prev_len = len(df)
        uniques = (
            df.groupby("donor_id")["conversation_id"].unique().apply(lambda x: len(x) >= thresh)
        )
        df = df[df.donor_id.isin(uniques[uniques].index)]
        post_len = len(df)
        print(
            f"Removing donations with < {thresh} chats removes {prev_len - post_len}/{prev_len} "
            f"rows ({(prev_len - post_len) / prev_len * 100:.2f}%) or "
            f"{len(uniques) - len(uniques[uniques])}/{len(uniques)} donations"
        )
    return df


def filter_groups(df: pd.DataFrame) -> pd.DataFrame:
    ### Filter group conversations
    prev_len = len(df)
    group_chat_conv_ids = df.groupby("conversation_id").sender_id.nunique() > 2
    print(
        f"N group chats (before actual filtering): {len(group_chat_conv_ids[group_chat_conv_ids])}"
    )
    # Actual filtering
    df = df[~df.conversation_id.isin(group_chat_conv_ids[group_chat_conv_ids].index)]

    print(
        f"Filtering group chats removes {prev_len - len(df)}/{prev_len} messages "
        f"({(1 - len(df) / prev_len) * 100:.2f}%)"
    )
    group_chat_conv_ids = df.groupby("conversation_id").sender_id.nunique() > 2
    print(
        f"N group chats (after actual filtering): {len(group_chat_conv_ids[group_chat_conv_ids])}"
    )
    return df


def filter_system_messages(
    df: pd.DataFrame, message_count_threshold: int = 5, triads_only: bool = True
) -> pd.DataFrame:
    prev_len = len(df)
    prev_n_groups = (df.groupby("conversation_id").sender_id.nunique() > 2).sum()
    relevant_df = df[df.data_source_id == 2].copy()

    if triads_only:
        participants_per_conversation = relevant_df.groupby("conversation_id").sender_id.nunique()
        triad_conversation_ids = participants_per_conversation[
            participants_per_conversation == 3
        ].index
        relevant_df = relevant_df[relevant_df.conversation_id.isin(triad_conversation_ids)]

    # Step 2: Count messages per (sender_id, conversation_id)
    message_counts = relevant_df.groupby(["sender_id", "conversation_id"]).size()

    # Step 3: Identify system senders
    max_messages_per_sender = message_counts.groupby("sender_id").max()

    system_sender_ids = max_messages_per_sender[
        max_messages_per_sender <= message_count_threshold
    ].index

    print(f"Identified {len(system_sender_ids)} system senders to remove.")

    # Step 4: Remove all messages from these senders (globally)
    df = df[~df.sender_id.isin(system_sender_ids)].reset_index(drop=True)

    post_n_groups = (df.groupby("conversation_id").sender_id.nunique() > 2).sum()

    print(f"Group conversations went from {prev_n_groups} to {post_n_groups}")
    print(
        f"Filtering system messages removes {prev_len - len(df)}/{prev_len} messages ({(prev_len - len(df)) / prev_len:.2f}%)"
    )
    print(
        f"Remaining WA groups: {(df[df.data_source_id == 2].groupby('conversation_id').sender_id.nunique() > 2).sum()}"
    )
    return df


def transform_data(
    experiment_config,
    df: pd.DataFrame,
    plotting_transforms: bool = False,
    skip_merge: bool = False,
) -> pd.DataFrame:
    res_df = df.copy()
    if experiment_config.conversation_min_msg_threshold:
        ### Reporting
        prev_len = len(res_df)
        prev_n_convs = res_df.conversation_id.nunique()
        prev_donors = res_df.donor_id.unique()

        ### Calculation
        thresh = experiment_config.conversation_min_msg_threshold
        conv_ids = res_df.groupby("conversation_id").conversation_id.count() >= thresh
        res_df = res_df[res_df.conversation_id.isin(conv_ids[conv_ids].index)]

        ### Reporting
        post_len = len(res_df)
        post_n_convs = res_df.conversation_id.nunique()
        post_donors = res_df.donor_id.unique()
        print(
            f"A message limit of >= {thresh} messages per chat removes "
            f"{prev_len - post_len}/{prev_len} rows "
            f"({(prev_len - post_len) / prev_len * 100:.2f}%) and "
            f"{(prev_n_convs - post_n_convs)}/{prev_n_convs} convs."
        )
        print(f"Removed donors: {set(prev_donors).difference(set(post_donors))}")

    # Merge multiple messages by the same person into one line
    prev_len = len(res_df)
    if not skip_merge:
        res_df = merge_consecutive_sender_blocks(df=res_df, keep=experiment_config.block_merge)
        print(
            f"Reducing block messages to one per block removes {prev_len - len(res_df)}/{prev_len} messages "
            f"({(1 - len(res_df) / prev_len) * 100:.2f}%)"
        )

    prev_len = len(res_df)
    if plotting_transforms:
        res_df = res_df[~res_df.rt.isna()]
    else:
        # Remove rows that do not have both ego and alter rt
        res_df = res_df[~res_df.rt_ego.isna() & ~res_df.rt_prev_alter.isna()]
    print(
        f"Removing NaN (due to data transform) removes {prev_len - len(res_df)}/{prev_len} messages "
        f"({(1 - len(res_df) / prev_len) * 100:.2f}%)"
    )

    if experiment_config.filter_temporal_resolution == "seconds_only":
        conv_ids_to_retain = (
            res_df.groupby("conversation_id")
            .filter(lambda x: any(x["datetime"].dt.second.ne(0)))["conversation_id"]
            .unique()
        )
        res_df = res_df[res_df["conversation_id"].isin(conv_ids_to_retain)]

    # Remove columns that are not required
    if plotting_transforms:
        res_df = res_df.drop(
            columns=[
                "rt_ego",
                "rt_prev_alter",
            ],
            errors="ignore",
        )
    else:
        res_df = res_df.drop(
            columns=[
                "sender_id",
                "datetime",
                "is_group_conversation",
                "rt",
                "sender",
            ],
            errors="ignore",
        )

    ### Set zero response times to 1.
    if experiment_config.round_to_minute:

        def round_to_next_minute(td):
            rounded = td.round("min")
            return max(rounded, pd.Timedelta(minutes=1))

        columns_to_round = [
            "rt",
            "rt_ego",
            "rt_prev_alter",
        ]
        for col in columns_to_round:
            if col in res_df.columns:
                print(f"Rounding {col}")
                res_df[col] = res_df[col].apply(round_to_next_minute)
    else:
        print("Not rounding minutes")
        if "rt_ego" in res_df.columns:
            res_df.loc[
                res_df["rt_ego"] < pd.Timedelta(seconds=1),
                "rt_ego",
            ] += pd.Timedelta(seconds=1)
        if "rt_prev_alter" in res_df.columns:
            res_df.loc[
                res_df["rt_prev_alter"] < pd.Timedelta(seconds=1),
                "rt_prev_alter",
            ] += pd.Timedelta(seconds=1)

    ### Convert time to int before filtering
    if plotting_transforms:
        res_df = convert_timedelta_to_s(res_df, cols=["rt"])
    else:
        res_df = convert_timedelta_to_s(res_df, cols=["rt_ego", "rt_prev_alter"])

    ### Config-based optional filtering steps that need to be done after transformation
    res_df = _transform_filtering(experiment_config=experiment_config, df=res_df)

    ### Timescale transformations
    if experiment_config.temporal_resolution == "minute":
        if plotting_transforms:
            res_df = convert_to_min(res_df, cols=["rt"])
        else:
            res_df = convert_to_min(res_df, cols=["rt_ego", "rt_prev_alter"])

    ### Type transformations
    res_df.conversation_id = res_df.conversation_id.astype(str)
    res_df.donor_id = res_df.donor_id.astype(str)
    res_df.weekday = res_df.weekday.astype(int)
    res_df.hour = res_df.hour.astype(int)
    res_df.wc_prev_alter = pd.to_numeric(res_df.wc_prev_alter)
    res_df.word_count = pd.to_numeric(res_df.word_count)
    res_df.hour = res_df.hour.astype("category")
    res_df.weekday = res_df.weekday.astype("category")

    if experiment_config.aggregation.message_aggregation == "conversation":
        return aggregate(
            df=res_df,
            agg_level=[
                "donor_id",
                "conversation_id",
            ],
            agg_type=experiment_config.aggregation.aggregation_type,
            agg_period=experiment_config.aggregation.aggregation_period,
        )
    if experiment_config.aggregation.message_aggregation == "donor":
        return aggregate(
            df=res_df,
            agg_level="donor_id",
            agg_type=experiment_config.aggregation.aggregation_type,
            agg_period=experiment_config.aggregation.aggregation_period,
        )
    return res_df
