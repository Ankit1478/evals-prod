"""The store is append-only, and finished runs are frozen."""

import pytest

from evalkit.schema.score import CaseResult
from evalkit.stats.gate import load_results
from evalkit.store import JsonlStore, RunSummary, StoreError
from tests.conftest import make_traj


def _start(store, run_id="2026-01-01T00-00-00_aaaaaa", suite="s"):
    return store.start_run({"run_id": run_id, "suite": suite,
                            "started_at": "2026-01-01T00:00:00+00:00",
                            "case_count": 1, "trials_per_case": 1})


def _result(case_id="t_case_01"):
    return CaseResult(case_id=case_id, trial_index=0, scores=[], passed=True,
                      stop_reason="completed")


def test_round_trip(tmp_path):
    store = JsonlStore(tmp_path)
    run_id = _start(store)
    traj = make_traj().model_copy(update={"run_id": run_id})
    store.write_trajectory(traj)
    store.write_result(_result(), kind="capability")
    store.finish_run(RunSummary(run_id=run_id, suite="s", started_at="x",
                                finished_at="y", results=1, passed=1))

    run = store.load_run(run_id)
    assert run.trajectories[0].case_id == "t_case_01"
    assert run.results[0].passed and run.kinds["t_case_01"] == "capability"
    assert run.summary.finished_at == "y"


def test_gate_still_reads_what_the_store_writes(tmp_path):
    store = JsonlStore(tmp_path)
    run_id = _start(store)
    store.write_result(_result(), kind="regression")
    results, kinds = load_results(tmp_path / run_id)
    assert results[0].case_id == "t_case_01" and kinds == {"t_case_01": "regression"}


def test_run_ids_are_never_reused(tmp_path):
    store = JsonlStore(tmp_path)
    _start(store)
    with pytest.raises(StoreError):
        _start(store)


def test_a_trajectory_is_never_overwritten(tmp_path):
    store = JsonlStore(tmp_path)
    run_id = _start(store)
    traj = make_traj().model_copy(update={"run_id": run_id})
    store.write_trajectory(traj)
    with pytest.raises(StoreError):
        store.write_trajectory(traj)


def test_finished_run_is_frozen(tmp_path):
    store = JsonlStore(tmp_path)
    run_id = _start(store)
    store.finish_run(RunSummary(run_id=run_id, suite="s", started_at="x", finished_at="y"))
    with pytest.raises(StoreError):
        store.write_result(_result())
    with pytest.raises(StoreError):
        store.finish_run(RunSummary(run_id=run_id, suite="s", started_at="x"))


def test_list_runs_newest_first_and_filters_by_suite(tmp_path):
    store = JsonlStore(tmp_path)
    _start(store, "2026-01-01T00-00-00_aaaaaa", suite="a")
    _start(store, "2026-01-02T00-00-00_bbbbbb", suite="b")
    _start(store, "2026-01-03T00-00-00_cccccc", suite="a")
    assert [r.run_id for r in store.list_runs()][0] == "2026-01-03T00-00-00_cccccc"
    assert [r.suite for r in store.list_runs(suite="a")] == ["a", "a"]
    assert store.list_runs(limit=1)[0].finished_at is None     # still open
