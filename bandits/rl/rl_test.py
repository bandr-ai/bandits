"""Tests for the RL environment layer.

The load-bearing pair is :func:`test_golden_episode_scores_one` and
:func:`test_cheating_episode_scores_zero`: the same final state, the same
verifier, reached two different ways, must score 1.0 and 0.0. If that pair ever
stops holding, the environment pays for cheating and every number downstream is
worthless.

Everything is built through the real pipeline (``load_corpus`` ->
``build_surface`` -> ``infer_schema`` -> ``mine_tasks`` -> ``synthesize_verifier``).
Hand-built fixtures would let this suite keep passing while the pipeline that
actually feeds it drifts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bandits.contracts import ToolClass
from bandits.fidelity.replay import tool_classes_from_surface
from bandits.ingest import load_corpus_and_registry
from bandits.rl import EnvSpec, StepResult, TaskSuite, TraceEnv, make_env
from bandits.rl.episode import EpisodeNotStartedError
from bandits.rl.spec import digest_of
from bandits.state import infer_schema
from bandits.surface import build_surface
from bandits.task import mine_tasks
from bandits.verify import synthesize_verifier
from bandits.verify.run import UnreviewedVerifierError

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

REVIEWER = "test-reviewer"
GOLDEN_TASK = "task-ep-refund-ok"

#: The golden solution, as production performed it.
SOLUTION = (
    ("get_order", {"order_id": 7741}),
    ("refund_order", {"order_id": 7741, "amount_cents": 4200}),
    ("send_email", {"to_customer_id": 88, "subject": "Your refund",
                    "body": "Refunded 42.00 for order 7741."}),
)


# ------------------------------------------------------------------ pipeline


@pytest.fixture(scope="module")
def pipeline():
    corpus, registry = load_corpus_and_registry(
        FIXTURES / "traces.otlp.jsonl", "otlp", FIXTURES / "tools.json"
    )
    surface = build_surface(corpus, declared_tools=registry)
    schema = infer_schema(corpus, surface)
    mining = mine_tasks(corpus, schema)
    return {
        "corpus": corpus,
        "surface": surface,
        "schema": schema,
        "tasks": tuple(mining.tasks),
        "traces": {t.trace_id: t for t in corpus.traces},
        "tool_classes": tool_classes_from_surface(surface),
    }


def _task(pipeline, task_id: str):
    for task in pipeline["tasks"]:
        if task.task_id == task_id:
            return task
    raise AssertionError(f"{task_id} not mined from the golden corpus")


def _verifier(pipeline, task_id: str, *, reviewed: bool = True):
    task = _task(pipeline, task_id)
    externals = [
        n for n, k in pipeline["tool_classes"].items() if k is ToolClass.EXTERNAL
    ]
    verifier = synthesize_verifier(
        task, pipeline["traces"][task.trace_id], pipeline["schema"],
        external_tools=externals,
    )
    if reviewed:
        verifier = verifier.model_copy(update={"reviewed_by": REVIEWER})
    return verifier


@pytest.fixture
def golden_env(pipeline):
    env = make_env(
        pipeline["schema"],
        _task(pipeline, GOLDEN_TASK),
        _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"],
        max_steps=12,
    )
    yield env
    env.close()


def _run(env: TraceEnv, actions, *, message="done", seed=0):
    """reset -> actions -> finish. Returns ``(steps, terminal_step)``."""
    env.reset(seed=seed)
    steps = []
    for name, arguments in actions:
        steps.append(env.step({"name": name, "arguments": arguments}))
    terminal = env.step({"name": env.finish_tool, "arguments": {"message": message}})
    return steps, terminal


# ------------------------------------------------------------------ the pair


def test_golden_episode_scores_one(golden_env):
    """The full successful episode on ep-refund-ok: reward 1.0, clean."""
    obs = golden_env.reset(seed=0)
    assert obs["instruction"].startswith("I want a refund")
    assert obs["task_id"] == GOLDEN_TASK
    assert "get_order" in {t["name"] for t in obs["tools"]}

    steps, terminal = _run(golden_env, SOLUTION)
    assert [s.reward for s in steps] == [0.0, 0.0, 0.0], "no shaped intermediate reward"
    assert all(not s.done for s in steps)
    assert terminal.reward == 1.0
    assert terminal.done and not terminal.truncated
    assert terminal.info["terminal_reason"] == "finish"
    assert terminal.info["verification"].passed
    assert terminal.info["anticheat_clean"]
    assert terminal.info["final_message"] == "done"
    # send_email is recorded, never performed.
    assert [e["tool"] for e in terminal.info["effects"]] == ["send_email"]


def test_cheating_episode_scores_zero(golden_env):
    """Same final state, reached by writing the store directly: reward 0.0.

    This is the rollout that reaches around ``EnvSession.execute`` -- exactly
    what ``bandits/env/interface.py``'s boundary rule exists to make detectable.
    The assertions all pass; the reward is still zero, and the report says why.
    """
    golden_env.reset(seed=0)
    golden_env.step({"name": "get_order", "arguments": {"order_id": 7741}})
    # The cheat: mutate the store without a WRITE-class tool call.
    assert golden_env.session is not None
    golden_env.session.store.update(
        "orders", "order_id", 7741,
        {"status": "refunded", "refund_amount_cents": 4200},
    )
    golden_env.step({
        "name": "send_email",
        "arguments": {"to_customer_id": 88, "subject": "Your refund", "body": "Refunded."},
    })
    terminal = golden_env.step({"name": "respond", "arguments": {"message": "done"}})

    assert terminal.reward == 0.0
    assert terminal.info["reward_zeroed_by_anticheat"] is True
    assert not terminal.info["anticheat_clean"]
    guards = {f["guard"] for f in terminal.info["anticheat_findings"] if f["severity"] == "fail"}
    assert guards & {"state_changed_without_write_call", "unattributed_row_change"}
    assert terminal.info["verification"].reward == 0.0


def test_forbidden_read_is_caught_as_cheating(golden_env):
    """Probing for the answer key inside an argument trips the guard too."""
    steps, terminal = _run(
        golden_env,
        (
            *SOLUTION,
            ("get_order", {"order_id": "expected_state.json"}),
        ),
    )
    assert terminal.reward == 0.0
    assert terminal.info["reward_zeroed_by_anticheat"] is True
    assert any(
        f["guard"] == "forbidden_read"
        for f in terminal.info["anticheat_findings"]
    )


# ------------------------------------------------------------------ failures


def test_missing_refund_scores_zero(golden_env):
    """An episode that reads and emails but never refunds gets nothing."""
    steps, terminal = _run(
        golden_env,
        (
            ("get_order", {"order_id": 7741}),
            ("send_email", {"to_customer_id": 88, "subject": "hi", "body": "soon"}),
        ),
    )
    assert terminal.reward == 0.0
    assert terminal.info["anticheat_clean"], "a lazy episode is wrong, not cheating"
    result = terminal.info["verification"]
    assert not result.passed
    failed = [r for r in result.results if not r.passed]
    assert failed and failed[0].assertion.field == "status"


def test_refunding_the_wrong_order_scores_zero(golden_env):
    """Right tool, wrong row. Still zero: reward is a state assertion."""
    _steps, terminal = _run(
        golden_env,
        (
            ("get_order", {"order_id": 7741}),
            ("refund_order", {"order_id": 7742, "amount_cents": 4200}),
            ("send_email", {"to_customer_id": 88, "subject": "s", "body": "b"}),
        ),
    )
    assert terminal.reward == 0.0
    assert not terminal.info["verification"].passed


def test_known_gap_the_refund_amount_is_not_asserted(golden_env):
    """Pins a real hole in the synthesized verifier, rather than hiding it.

    ``ep-refund-ok`` never read ``orders[7741]`` before the refund with a
    ``refund_amount_cents`` column populated, so synthesis has no pre-state value
    to diff against and emits no assertion for it. A one-cent refund therefore
    still scores 1.0. That is an upstream verifier-coverage limitation, not an
    RL-layer one -- this layer's job is to report whatever the verifier says, and
    it does. Written down here so the hole is visible; if stage 5 ever asserts
    the amount, this test should be inverted, not deleted.
    """
    _steps, terminal = _run(
        golden_env,
        (
            ("get_order", {"order_id": 7741}),
            ("refund_order", {"order_id": 7741, "amount_cents": 1}),
            ("send_email", {"to_customer_id": 88, "subject": "s", "body": "b"}),
        ),
    )
    assert terminal.reward == 1.0
    assert not any(
        r.assertion.field == "refund_amount_cents"
        for r in terminal.info["verification"].results
    )


def test_known_gap_collateral_damage_to_a_partial_row(pipeline):
    """A second, partially-known row can be written without cost. Also upstream.

    ``orders[7742]`` was only ever *named* in a ``search_orders`` id list, so its
    STATE_UNCHANGED assertion covers exactly the two fields production proved
    (``order_id``, ``customer_id``) and nothing else -- deliberately narrow, see
    ``synthesize.partial_key_ids``. Refunding it is therefore free. Collateral
    damage to a *fully* observed row is not; that is what the STATE_UNCHANGED on
    ``customers`` and on ``orders[7741]`` covers.
    """
    env = make_env(
        pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"],
    )
    with env:
        _steps, terminal = _run(
            env, (*SOLUTION, ("refund_order", {"order_id": 7742, "amount_cents": 100})),
        )
    assert terminal.reward == 1.0


def test_collateral_damage_to_an_observed_row_scores_zero(pipeline):
    """Refund the right order, then wreck a row the verifier does know."""
    env = make_env(
        pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"],
    )
    with env:
        _steps, terminal = _run(
            env,
            (
                *SOLUTION,
                ("update_order_status", {"order_id": 7741, "status": "cancelled"}),
            ),
        )
    assert terminal.reward == 0.0
    assert not terminal.info["verification"].passed


# ------------------------------------------------------------------ truncation


def test_max_steps_truncates_with_zero_reward(pipeline):
    env = make_env(
        pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"], max_steps=3,
    )
    with env:
        env.reset(seed=0)
        first = env.step({"name": "get_order", "arguments": {"order_id": 7741}})
        assert not first.done
        env.step({"name": "get_order", "arguments": {"order_id": 7741}})
        last = env.step({"name": "get_order", "arguments": {"order_id": 7741}})
        assert last.done and last.truncated
        assert last.reward == 0.0
        assert last.info["terminal_reason"] == "max_steps"
        assert last.info["verification"] is None
        with pytest.raises(EpisodeNotStartedError):
            env.step({"name": "respond", "arguments": {}})


def test_truncation_is_zero_even_when_the_task_was_solved(pipeline):
    """The step budget is a hard boundary, not a discount factor."""
    env = make_env(
        pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"], max_steps=len(SOLUTION), expose_progress=True,
    )
    with env:
        env.reset(seed=0)
        for name, arguments in SOLUTION:
            last = env.step({"name": name, "arguments": arguments})
    assert last.truncated and last.done
    assert last.reward == 0.0
    # The diagnostic knows the state is right. The reward still does not care.
    assert last.info["progress"]["assertions_passing_fraction"] == 1.0
    assert "must never be summed" in last.info["progress"]["WARNING"]


# ------------------------------------------------------------------ tool errors


def test_unsupported_tool_returns_an_observation(golden_env):
    """escalate_to_human is UNKNOWN, so it is unsupported. It must not crash."""
    golden_env.reset(seed=0)
    step = golden_env.step({"name": "escalate_to_human", "arguments": {"reason": "help"}})
    assert not step.done
    assert step.reward == 0.0
    assert step.observation["status"] == "error"
    assert step.observation["error_kind"] == "unsupported_tool"
    assert step.info["unsupported_tool_attempt_count"] == 1
    assert step.info["unsupported_tool_attempts"][0]["tool"] == "escalate_to_human"
    # And the episode is still usable afterwards.
    _steps, terminal = _run(golden_env, ())
    assert terminal.done


def test_unsupported_tool_is_absent_from_the_action_space(golden_env):
    obs = golden_env.reset(seed=0)
    assert "escalate_to_human" not in {t["name"] for t in obs["tools"]}
    assert "escalate_to_human" in obs["context"]["unsupported_tools"]


def test_unsupported_attempt_still_lets_a_correct_episode_score_one(golden_env):
    """A wasted exploratory call is not cheating and must not cost reward."""
    golden_env.reset(seed=0)
    golden_env.step({"name": "escalate_to_human", "arguments": {}})
    for name, arguments in SOLUTION:
        golden_env.step({"name": name, "arguments": arguments})
    terminal = golden_env.step({"name": "respond", "arguments": {"message": "done"}})
    assert terminal.reward == 1.0
    assert terminal.info["unsupported_tool_attempt_count"] == 1
    assert any(
        f["guard"] == "unsupported_tool_called" and f["severity"] == "warn"
        for f in terminal.info["anticheat_findings"]
    )


def test_observed_tool_error_is_ordinary_dynamics(golden_env):
    """A missing row is an ERROR observation, not an env failure."""
    golden_env.reset(seed=0)
    step = golden_env.step({"name": "get_order", "arguments": {"order_id": 999999}})
    assert not step.done
    assert step.observation["status"] == "error"
    assert step.observation["error_kind"] != "unsupported_tool"
    assert step.info["unsupported_tool_attempt_count"] == 0


def test_malformed_action_does_not_raise(golden_env):
    golden_env.reset(seed=0)
    step = golden_env.step({"arguments": {"order_id": 7741}})
    assert step.observation["error_kind"] == "malformed_action"
    assert step.reward == 0.0 and not step.done
    step = golden_env.step({"name": "get_order", "arguments": [1, 2]})
    assert step.observation["error_kind"] == "malformed_action"


def test_step_before_reset_raises(pipeline):
    env = make_env(
        pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
        surface=pipeline["surface"],
    )
    with pytest.raises(EpisodeNotStartedError):
        env.step({"name": "get_order", "arguments": {}})


# ------------------------------------------------------------------ determinism


def test_same_seed_same_actions_are_identical(pipeline):
    runs = []
    for _ in range(2):
        env = make_env(
            pipeline["schema"], _task(pipeline, GOLDEN_TASK), _verifier(pipeline, GOLDEN_TASK),
            surface=pipeline["surface"],
        )
        with env:
            first = env.reset(seed=17)
            steps, terminal = _run(env, SOLUTION, seed=17)
            runs.append(
                {
                    "reset": first,
                    "observations": [s.observation for s in steps],
                    "reward": terminal.reward,
                    "digest": terminal.info["digest"],
                    "effects": terminal.info["effects"],
                }
            )
    assert runs[0]["observations"] == runs[1]["observations"]
    assert runs[0]["reset"] == runs[1]["reset"]
    assert runs[0]["reward"] == runs[1]["reward"] == 1.0
    assert runs[0]["digest"] == runs[1]["digest"]
    assert runs[0]["effects"] == runs[1]["effects"]


def test_reset_rebuilds_a_fresh_world(golden_env):
    """A second episode must not inherit the first one's writes."""
    _steps, terminal = _run(golden_env, SOLUTION)
    assert terminal.reward == 1.0
    solved = terminal.info["digest"]

    golden_env.reset(seed=0)
    obs = golden_env.step({"name": "get_order", "arguments": {"order_id": 7741}})
    assert obs.observation["response"]["status"] == "delivered"
    _steps, second = _run(golden_env, SOLUTION)
    assert second.info["digest"] == solved


# ------------------------------------------------------------------ spec


def test_spec_is_serializable_and_stable(golden_env):
    spec = golden_env.spec()
    payload = spec.to_json()
    assert payload["task_id"] == GOLDEN_TASK
    assert payload["reward_range"] == [0.0, 1.0]
    assert payload["reward_mode"] == "all_or_nothing"
    assert payload["max_steps"] == 12
    assert payload["schema_digest"] and payload["verifier_digest"]
    assert payload["verifier_reviewed_by"] == REVIEWER
    assert "escalate_to_human" in payload["unsupported_tools"]
    assert "store_policy" in payload["static_entities"]

    round_tripped = EnvSpec.from_json(payload)
    assert round_tripped.to_json() == payload
    assert round_tripped.spec_digest == spec.spec_digest
    assert golden_env.spec().spec_digest == spec.spec_digest


def test_spec_tool_schemas_come_from_the_declared_registry(golden_env):
    spec = golden_env.spec()
    refund = next(t for t in spec.tools if t.name == "refund_order")
    assert refund.tool_class == "write"
    assert refund.parameters["properties"]["order_id"]["type"] == "integer"
    assert refund.parameters["x-bandits-source"] == "declared"
    email = next(t for t in spec.tools if t.name == "send_email")
    assert email.tool_class == "external"
    assert "never performed" in email.description
    assert spec.action_space["properties"]["name"]["enum"][-1] == spec.finish_tool


def test_verifier_digest_changes_with_the_verifier(pipeline):
    v = _verifier(pipeline, GOLDEN_TASK)
    other = v.model_copy(update={"assertions": v.assertions[:1]})
    assert digest_of(v.model_dump(mode="json")) != digest_of(other.model_dump(mode="json"))


# ------------------------------------------------------------------ the review gate


def test_unreviewed_verifier_is_refused_at_construction(pipeline):
    with pytest.raises(UnreviewedVerifierError):
        make_env(
            pipeline["schema"],
            _task(pipeline, GOLDEN_TASK),
            _verifier(pipeline, GOLDEN_TASK, reviewed=False),
            surface=pipeline["surface"],
        )


def test_verifier_for_another_task_is_refused(pipeline):
    with pytest.raises(ValueError, match="grades task"):
        make_env(
            pipeline["schema"],
            _task(pipeline, "task-ep-refund-ok-2"),
            _verifier(pipeline, GOLDEN_TASK),
            surface=pipeline["surface"],
        )


def test_suite_excludes_unreviewed_verifiers_by_default(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
    )
    assert len(suite) == 0
    reasons = {e["reason"] for e in suite.excluded}
    assert "unreviewed_verifier" in reasons
    assert all(e["task_id"] for e in suite.excluded)


def test_suite_includes_reviewed_tasks(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        reviewed_by={GOLDEN_TASK: REVIEWER},
    )
    assert suite.task_ids == (GOLDEN_TASK,)
    assert suite[0].reviewed
    assert any(e["reason"] == "unreviewed_verifier" for e in suite.excluded)


def test_suite_allow_unreviewed_opts_back_in(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        allow_unreviewed=True,
    )
    assert len(suite) > 1
    assert GOLDEN_TASK in suite.task_ids
    # Unlabeled / negative traces never produced a verifier at all.
    assert {e["reason"] for e in suite.excluded} <= {
        "unlabeled_trace", "no_assertions", "synthesis_failed", "missing_trace",
    }
    assert suite.filter(reviewed=True).task_ids == ()


def test_suite_filters_by_outcome(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        allow_unreviewed=True,
    )
    passing = suite.filter(outcome=True)
    assert len(passing) == len(suite), "only positively labeled traces yield verifiers"
    assert len(suite.filter(outcome=False)) == 0
    assert len(suite.filter(task_ids=[GOLDEN_TASK])) == 1


def test_suite_sampling_is_deterministic(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        allow_unreviewed=True,
    )
    assert [suite.sample(s).task_id for s in range(8)] == [
        suite.sample(s).task_id for s in range(8)
    ]
    assert suite.sample(None).task_id == suite[0].task_id
    assert len({suite.sample(s).task_id for s in range(50)}) > 1


def test_suite_make_env_runs_an_episode(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        reviewed_by={GOLDEN_TASK: REVIEWER},
    )
    env = suite.make_env(seed=3)
    with env:
        _steps, terminal = _run(env, SOLUTION)
    assert terminal.reward == 1.0
    assert isinstance(terminal, StepResult)


def test_empty_suite_sampling_says_why(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
    )
    with pytest.raises(IndexError, match="unreviewed verifier"):
        suite.sample(0)


def test_suite_serializes(pipeline):
    suite = TaskSuite.from_pipeline(
        corpus=pipeline["corpus"], schema=pipeline["schema"],
        tasks=pipeline["tasks"], surface=pipeline["surface"],
        reviewed_by={GOLDEN_TASK: REVIEWER},
    )
    payload = suite.to_json()
    assert payload["task_count"] == 1
    assert payload["tasks"][0]["task_id"] == GOLDEN_TASK
    assert payload["excluded"]


# ------------------------------------------------------------------ step tuple


def test_step_result_unpacks_as_a_gym_tuple(golden_env):
    golden_env.reset(seed=0)
    obs, reward, done, truncated, info = golden_env.step(
        {"name": "get_order", "arguments": {"order_id": 7741}}
    )
    assert reward == 0.0 and not done and not truncated
    assert obs["status"] == "ok" and info["step"] == 1
