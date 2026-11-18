from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from force2020_reliability.config import ExperimentConfig
from force2020_reliability.paths import INTERIM_DIR

SPLITS_DIR = INTERIM_DIR / "splits"


def well_coordinate_table(frame: pd.DataFrame) -> pd.DataFrame:
    coordinates = (
        frame.groupby("well_id", observed=True)
        .agg(
            x=("x_loc", lambda values: values.replace(0, np.nan).median()),
            y=("y_loc", lambda values: values.replace(0, np.nan).median()),
            rows=("well_id", "size"),
        )
        .reset_index()
        .sort_values("well_id")
    )
    invalid = coordinates[["x", "y"]].isna().any(axis=1)
    if invalid.any():
        wells = coordinates.loc[invalid, "well_id"].tolist()
        raise ValueError(f"Wells without valid coordinates: {wells}")
    return coordinates


def _assign_well_folds(
    frame: pd.DataFrame,
    target_column: str,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    well_assignment: dict[str, int] = {}
    groups = frame["well_id"].to_numpy()
    y = frame[target_column].to_numpy()
    for fold, (_, test_index) in enumerate(splitter.split(frame, y, groups)):
        for well_id in pd.unique(groups[test_index]):
            if well_id in well_assignment:
                raise RuntimeError(f"Well assigned to multiple folds: {well_id}")
            well_assignment[str(well_id)] = fold
    missing = sorted(set(frame["well_id"].unique()) - set(well_assignment))
    if missing:
        raise RuntimeError(f"Wells missing from grouped split: {missing}")

    assigned = pd.DataFrame(
        {"well_id": list(well_assignment.keys()), "fold": list(well_assignment.values())}
    ).sort_values("well_id")
    empty_folds = sorted(set(range(n_folds)) - set(assigned["fold"]))
    if empty_folds:
        raise ValueError(f"Grouped assignment left empty folds: {empty_folds}")
    return assigned.reset_index(drop=True)

def make_random_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path = SPLITS_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed,
    )
    fold_assignment = np.full(len(frame), -1, dtype=np.int8)
    y = frame[config.target_column].to_numpy()
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(frame)), y)):
        fold_assignment[test_index] = fold
    if (fold_assignment < 0).any():
        raise RuntimeError("Random split did not assign every row")
    manifest = frame[["well_id", "row_id"]].copy()
    manifest["fold"] = fold_assignment
    manifest.to_parquet(output_dir / "random_folds.parquet", index=False)
    return manifest


def make_well_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path = SPLITS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = _assign_well_folds(
        frame=frame,
        target_column=config.target_column,
        n_folds=config.n_folds,
        seed=config.seed,
    )
    assignment.to_csv(output_dir / "well_folds.csv", index=False, encoding="utf-8-sig")
    wells = well_coordinate_table(frame)
    coordinates = wells[["x", "y"]].to_numpy(dtype=float)
    distance = cdist(coordinates, coordinates, metric="euclidean")

    roles: list[pd.DataFrame] = []
    for test_fold in range(config.n_folds):
        test_mask = assignment["fold"].to_numpy() == test_fold
        train_mask = ~test_mask
        rows = assignment.copy()
        rows["evaluation_fold"] = test_fold
        rows["role"] = np.where(test_mask, "test", "train")
        rows["distance_to_nearest_test_m"] = distance[:, test_mask].min(axis=1)
        rows["distance_to_nearest_train_m"] = distance[:, train_mask].min(axis=1)
        roles.append(
            rows[
                [
                    "evaluation_fold",
                    "well_id",
                    "role",
                    "distance_to_nearest_test_m",
                    "distance_to_nearest_train_m",
                ]
            ]
        )
    role_frame = pd.concat(roles, ignore_index=True)
    role_frame.to_csv(output_dir / "well_roles.csv", index=False, encoding="utf-8-sig")
    return assignment, role_frame


def make_spatial_split(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path = SPLITS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wells = well_coordinate_table(frame)
    coordinates = wells[["x", "y"]].to_numpy(dtype=float)
    n_clusters = min(config.spatial_clusters, len(wells))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.seed,
        n_init=50,
    )
    cluster = kmeans.fit_predict(coordinates, sample_weight=np.sqrt(wells["rows"].to_numpy()))
    wells["fold"] = cluster.astype(int)
    wells = wells.sort_values(["fold", "well_id"]).reset_index(drop=True)
    wells.to_csv(output_dir / "spatial_well_folds.csv", index=False, encoding="utf-8-sig")

    distance = cdist(coordinates, coordinates, metric="euclidean")
    roles: list[pd.DataFrame] = []
    for test_fold in range(n_clusters):
        test_mask = wells["fold"].to_numpy() == test_fold
        nearest_test = distance[:, test_mask].min(axis=1)
        role = np.full(len(wells), "train", dtype=object)
        role[test_mask] = "test"
        role[(~test_mask) & (nearest_test < config.spatial_buffer_m)] = "buffer_excluded"
        train_mask = role == "train"
        nearest_train = distance[:, train_mask].min(axis=1)
        fold_roles = wells[["well_id", "fold"]].copy()
        fold_roles["evaluation_fold"] = test_fold
        fold_roles["role"] = role
        fold_roles["distance_to_nearest_test_m"] = nearest_test
        fold_roles["distance_to_nearest_train_m"] = nearest_train
        roles.append(
            fold_roles[
                [
                    "evaluation_fold",
                    "well_id",
                    "role",
                    "distance_to_nearest_test_m",
                    "distance_to_nearest_train_m",
                    "fold",
                ]
            ]
        )
    role_frame = pd.concat(roles, ignore_index=True)
    role_frame.to_csv(
        output_dir / "spatial_well_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return wells, role_frame


def split_summary(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    random_manifest: pd.DataFrame,
    well_assignment: pd.DataFrame,
    spatial_assignment: pd.DataFrame,
    spatial_roles: pd.DataFrame,
) -> dict[str, object]:
    target = config.target_column
    well_counts = frame.groupby("well_id", observed=True).size()
    random_summary = (
        random_manifest.assign(rows=1)
        .groupby("fold", observed=True)["rows"]
        .sum()
        .astype(int)
        .to_dict()
    )
    well_summary = (
        well_assignment.merge(well_counts.rename("rows"), on="well_id", how="left")
        .assign(n_classes=lambda data: data["well_id"].map(frame.groupby("well_id")[target].nunique()))
        .groupby("fold", observed=True)
        .agg(n_wells=("well_id", "size"), rows=("rows", "sum"), mean_classes=("n_classes", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    spatial_summary = (
        spatial_roles.groupby(["evaluation_fold", "role"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "dataset": {
            "n_wells": int(frame["well_id"].nunique()),
            "n_rows": len(frame),
            "n_classes": int(frame[target].nunique()),
        },
        "random_point_folds_rows": random_summary,
        "well_grouped_folds": well_summary,
        "spatial_fold_wells": spatial_summary,
        "spatial_buffer_m": config.spatial_buffer_m,
    }


def generate_all_splits(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path = SPLITS_DIR,
) -> dict[str, object]:
    random_manifest = make_random_split(frame, config, output_dir)
    well_assignment, _ = make_well_split(frame, config, output_dir)
    spatial_assignment, spatial_roles = make_spatial_split(frame, config, output_dir)
    summary = split_summary(
        frame,
        config,
        random_manifest,
        well_assignment,
        spatial_assignment,
        spatial_roles,
    )
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def resolve_fold_members(
    protocol: str,
    fold: int,
    frame: pd.DataFrame,
    splits_dir: Path = SPLITS_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    if protocol == "random_point":
        manifest = pd.read_parquet(splits_dir / "random_folds.parquet")
        fold_values = manifest.loc[manifest["fold"] == fold]
        row_keys = pd.MultiIndex.from_frame(frame[["well_id", "row_id"]])
        fold_keys = pd.MultiIndex.from_frame(fold_values[["well_id", "row_id"]])
        test_mask = row_keys.isin(fold_keys)
        return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    if protocol == "well_grouped":
        roles = pd.read_csv(splits_dir / "well_roles.csv")
        selected = roles.loc[roles["evaluation_fold"] == fold]
    elif protocol == "spatial_block":
        roles = pd.read_csv(splits_dir / "spatial_well_roles.csv")
        selected = roles.loc[roles["evaluation_fold"] == fold]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    train_wells = set(selected.loc[selected["role"] == "train", "well_id"])
    test_wells = set(selected.loc[selected["role"] == "test", "well_id"])
    is_train = frame["well_id"].isin(train_wells).to_numpy()
    is_test = frame["well_id"].isin(test_wells).to_numpy()
    return np.flatnonzero(is_train), np.flatnonzero(is_test)