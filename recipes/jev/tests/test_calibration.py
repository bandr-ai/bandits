from __future__ import annotations

import pytest

from bandits.store import DerivedStore
from bandits_jev.calibration import (
    MAX_TEMPERATURE,
    calibrate,
    compute_calibration_id,
    fit_temperature,
    load_calibration,
    save_calibration,
    tempered_probabilities,
)
from tests.eval_fixtures import ADAPTER, build_dataset, make_run


@pytest.fixture
def store(tmp_path) -> DerivedStore:
    return DerivedStore(tmp_path)


def _examples(dataset, split):
    return [e for e in dataset.examples if e.split == split]


def test_fitted_temperature_undoes_a_known_overconfidence(store) -> None:
    """Logits are 3x the true logits the labels were sampled from, so the
    NLL-optimal temperature is 3 up to sampling noise."""
    dataset_id, dataset, truths = build_dataset(store, per_split={"calibration": 3000})
    run = make_run(dataset_id, dataset, truths, split="calibration", scale=3.0, adapter_digest=ADAPTER)

    calibration = calibrate(run, "run-1", _examples(dataset, "calibration"))

    assert calibration.temperature == pytest.approx(3.0, rel=0.12)
    assert not calibration.at_bound
    assert calibration.nll_after < calibration.nll_before
    assert calibration.ece_after < calibration.ece_before
    assert calibration.adapter_digest == ADAPTER


def test_a_well_calibrated_model_keeps_temperature_near_one(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"calibration": 3000}, seed=4)
    run = make_run(dataset_id, dataset, truths, split="calibration", scale=1.0)

    assert calibrate(run, "r", _examples(dataset, "calibration")).temperature == pytest.approx(1.0, rel=0.12)


def test_temperature_is_fitted_only_on_the_calibration_split(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"dev": 20, "test": 20})
    for split in ("dev", "test"):
        run = make_run(dataset_id, dataset, truths, split=split, scale=2.0)
        with pytest.raises(ValueError, match="only on the 'calibration' split"):
            calibrate(run, "r", _examples(dataset, split))


def test_temperature_one_reproduces_the_stored_probabilities(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"calibration": 10})
    for two_order in (False, True):
        run = make_run(
            dataset_id, dataset, truths, split="calibration", scale=2.0, two_order=two_order, order_bias=1.0
        )
        for result in run.results:
            rebuilt = tempered_probabilities(result, 1.0)
            for score in result.scores:
                assert rebuilt[score.option_id] == pytest.approx(score.probability, abs=1e-12)


def test_fit_is_deterministic_and_the_artifact_round_trips(store) -> None:
    dataset_id, dataset, truths = build_dataset(store, per_split={"calibration": 200})
    run = make_run(dataset_id, dataset, truths, split="calibration", scale=2.5)
    first = calibrate(run, "r", _examples(dataset, "calibration"))
    second = calibrate(run, "r", _examples(dataset, "calibration"))

    assert first == second
    envelope = save_calibration(first, store)
    assert envelope.artifact_id == compute_calibration_id(first)
    assert load_calibration(envelope.artifact_id, store) == first


def test_a_fit_pinned_at_the_search_bound_is_flagged(store) -> None:
    """Logits that carry no signal but are wildly confident: the best the
    fit can do is flatten them as far as the range allows."""
    dataset_id, dataset, truths = build_dataset(store, per_split={"calibration": 300})
    run = make_run(
        dataset_id, dataset, truths, split="calibration", scale=0.0, noise=200.0, noise_seed=3
    )
    calibration = calibrate(run, "r", _examples(dataset, "calibration"))

    assert calibration.temperature == pytest.approx(MAX_TEMPERATURE, rel=1e-3)
    assert calibration.at_bound


def test_fit_refuses_nothing_to_fit() -> None:
    with pytest.raises(ValueError):
        fit_temperature([], {})
