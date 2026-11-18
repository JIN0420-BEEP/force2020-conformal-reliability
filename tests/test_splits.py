from __future__ import annotations

from force2020_reliability.splits.generate import (
    _assign_well_folds,
    make_random_split,
    make_spatial_split,
    resolve_fold_members,
)


def test_grouped_folds_keep_wells_disjoint(synthetic_frame, config):
    assignment = _assign_well_folds(
        synthetic_frame,
        target_column=config.target_column,
        n_folds=config.n_folds,
        seed=config.seed,
    )
    assert assignment["well_id"].is_unique
    assert assignment["well_id"].nunique() == synthetic_frame["well_id"].nunique()
    assert set(assignment["fold"]) == set(range(config.n_folds))
    assert assignment.groupby("fold").size().max() - assignment.groupby("fold").size().min() <= 1


def test_random_split_is_complete_and_well_rows_can_cross_folds(
    synthetic_frame, config, tmp_path
):
    manifest = make_random_split(synthetic_frame, config, output_dir=tmp_path)
    assert len(manifest) == len(synthetic_frame)
    assert set(manifest["fold"]) == set(range(config.n_folds))
    assert not manifest[["well_id", "row_id"]].duplicated().any()
    assert manifest.groupby("well_id")["fold"].nunique().max() > 1

    train_index, test_index = resolve_fold_members(
        "random_point", 0, synthetic_frame, splits_dir=tmp_path
    )
    assert len(train_index) + len(test_index) == len(synthetic_frame)
    assert set(train_index).isdisjoint(test_index)


def test_spatial_folds_apply_buffer(synthetic_frame, config, tmp_path):
    wells, roles = make_spatial_split(synthetic_frame, config, output_dir=tmp_path)
    assert wells["well_id"].is_unique
    assert roles["well_id"].nunique() == wells.shape[0]
    assert set(roles["role"]).issubset({"train", "test", "buffer_excluded"})
    for fold, fold_roles in roles.groupby("evaluation_fold"):
        assert (fold_roles["role"] == "test").sum() > 0
        assert (fold_roles["role"] == "train").sum() > 0
        excluded = fold_roles["role"] == "buffer_excluded"
        if excluded.any():
            assert (fold_roles.loc[excluded, "distance_to_nearest_test_m"] < config.spatial_buffer_m).all()
    train_index, test_index = resolve_fold_members(
        "spatial_block", 0, synthetic_frame, splits_dir=tmp_path
    )
    train_wells = set(synthetic_frame.iloc[train_index]["well_id"])
    test_wells = set(synthetic_frame.iloc[test_index]["well_id"])
    assert train_wells.isdisjoint(test_wells)