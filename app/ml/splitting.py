"""Chronological, temporally-non-overlapping, candidate-group-preserving dataset splitting.

Shared by the VIDEO and LIVE trainers. A candidate group (e.g. one user-day slate, or one
LIVE ranking window) is never cut across a split boundary. That alone is not enough to
guarantee chronology, though: two different groups can have overlapping time ranges (e.g.
two users each posting throughout the same calendar day), and if such groups were assigned
to different splits by group-start-time alone, the resulting splits would overlap in time --
violating the property that every row in an earlier split must precede every row in a later
one.

This module fixes that by first merging groups into maximal "chronological blocks": runs of
groups whose time ranges transitively overlap (`_build_chronological_blocks`). Any two groups
in different blocks are guaranteed to be strictly ordered in time, so blocks -- never
individual groups -- are the atomic unit assigned to named splits. This guarantees, for any
two non-empty splits A and B produced in that order:

    A["timestamp"].max() < B["timestamp"].min()

If a heavily-overlapping dataset collapses into fewer chronological blocks than requested
splits, some split would necessarily end up empty; `InsufficientSplitDataError` is raised
in that case rather than silently returning an empty (and therefore useless) partition.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


class InsufficientSplitDataError(Exception):
    """Raised when the dataset's chronological block structure cannot support the requested
    number of non-empty, group-preserving, strictly-ordered splits.

    A group is never split, and overlapping groups are always merged into one indivisible
    chronological block (see module docstring), so this is not a failure to *find* a valid
    partition -- a dataset simply does not contain enough non-overlapping time structure to
    populate every requested split without leaving at least one of them empty.
    """


def _build_chronological_blocks(
    df: pd.DataFrame, *, group_col: str, timestamp_col: str,
) -> list[dict[str, Any]]:
    """Merge groups into maximal blocks of transitively-overlapping time ranges.

    Two groups conflict (must share a block) if their [min, max] timestamp ranges overlap or
    touch, directly or transitively through a chain of other groups. Blocks are returned in
    chronological order with the guarantee that `blocks[i]["max"] < blocks[i + 1]["min"]` --
    exactly the property needed to assign whole blocks to strictly-ordered named splits.
    """
    grouped = df.groupby(group_col, sort=False)[timestamp_col].agg(["min", "max", "size"])
    grouped = grouped.sort_values("min", kind="stable")

    blocks: list[dict[str, Any]] = []
    for group_name, row in grouped.iterrows():
        group_min, group_max, group_size = row["min"], row["max"], int(row["size"])
        if blocks and group_min <= blocks[-1]["max"]:
            block = blocks[-1]
            block["groups"].append(group_name)
            block["max"] = max(block["max"], group_max)
            block["size"] += group_size
        else:
            blocks.append({"groups": [group_name], "min": group_min, "max": group_max, "size": group_size})
    return blocks


def chronological_group_split(
    df: pd.DataFrame,
    *,
    ratios: Mapping[str, float],
    group_col: str = "candidate_group",
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """Split `df` chronologically into named partitions without breaking a group or letting
    any two splits overlap in time.

    `ratios` is an ordered mapping of split name -> target row fraction (must sum to 1.0);
    iteration order determines chronological split order (e.g. train, validation, ...).
    Returns one DataFrame per split name, each sorted by `timestamp_col`.

    Raises `InsufficientSplitDataError` if the dataset's chronological block structure (see
    module docstring) cannot support every named split being non-empty.
    """
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    if not len(df):
        return {name: df.iloc[0:0].copy() for name in ratios}

    blocks = _build_chronological_blocks(df, group_col=group_col, timestamp_col=timestamp_col)
    names = list(ratios.keys())
    total_rows = len(df)
    boundaries: list[int] = []
    cumulative = 0.0
    for name in names:
        cumulative += ratios[name]
        boundaries.append(round(total_rows * cumulative))
    boundaries[-1] = total_rows  # absorb rounding drift into the final split

    assignment: dict[object, str] = {}
    running = 0
    split_index = 0
    for block in blocks:
        midpoint = running + block["size"] / 2
        while split_index < len(names) - 1 and midpoint > boundaries[split_index]:
            split_index += 1
        for group in block["groups"]:
            assignment[group] = names[split_index]
        running += block["size"]

    labels = df[group_col].map(assignment)
    splits = {name: df[labels == name].sort_values(timestamp_col, kind="stable").copy() for name in names}

    empty = [name for name in names if len(splits[name]) == 0]
    if empty:
        total_groups = sum(len(block["groups"]) for block in blocks)
        raise InsufficientSplitDataError(
            f"Cannot produce non-empty splits {empty} out of {len(names)} requested "
            f"({list(ratios)}): the {total_rows} rows across {total_groups} candidate group(s) "
            f"collapse into only {len(blocks)} strictly-ordered chronological block(s) after "
            "merging groups with overlapping time ranges (a candidate group is never split, and "
            "overlapping groups can never be placed in different splits without violating "
            "chronology). Provide more data, fewer overlapping groups, or request fewer splits."
        )
    return splits
