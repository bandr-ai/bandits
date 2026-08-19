"""Tests for the serving layer.

The fixtures here are hand-written contract objects modelling the same retail
world as the rest of the repo, deliberately independent of stages 1-3 and of
``bandits.rl``: if schema inference or the RL wrapper regresses, these tests
still say whether *serving* is correct. Sessions come from
``bandits.env.build_session`` via :class:`SessionBackend`, which is the real
runtime -- nothing here is mocked, so a passing isolation test is a statement
about actual SQLite stores.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from bandits.contracts import (
    Assertion,
    AssertionKind,
    EntityRows,
    EntitySchema,
    FieldProfile,
    ForeignKey,
    StateSchema,
    TaskCase,
    ToolClass,
    ToolProfile,
    ToolSurface,
    Verifier,
    WriteEffect,
)
from bandits.serve import (
    EnvClient,
    EnvClientError,
    ErrorCode,
    McpEnvServer,
    SessionBackend,
    serve,
)

# ---------------------------------------------------------------- fixtures


def f(name: str, *types: str, samples=(), ident: bool = False) -> FieldProfile:
    return FieldProfile(
        name=name,
        json_types=tuple(types),
        occurrences=len(samples) or 1,
        distinct_values=len(set(map(repr, samples))),
        sample_values=tuple(samples),
        looks_like_identifier=ident,
    )


ORDER_IDS = (7741, 7742, 7743, 7744)


def retail_schema() -> StateSchema:
    customers = EntitySchema(
        name="customers",
        primary_key="customer_id",
        fields=(
            f("customer_id", "integer", samples=[88], ident=True),
            f("name", "string", samples=["Ada Lovelace"]),
            f("email", "string", samples=["ada@example.com"]),
        ),
        read_by=("get_customer",),
        evidence_count=2,
    )
    orders = EntitySchema(
        name="orders",
        primary_key="order_id",
        fields=(
            f("order_id", "integer", samples=list(ORDER_IDS), ident=True),
            f("customer_id", "integer", samples=[88], ident=True),
            f("status", "string", samples=["delivered", "refunded", "in_transit"]),
            f("total_cents", "integer", samples=[4200]),
        ),
        foreign_keys=(
            ForeignKey(
                field="customer_id",
                references_entity="customers",
                references_field="customer_id",
                confidence=0.9,
            ),
        ),
        read_by=("get_order", "search_orders"),
        written_by=("refund_order", "update_order_status"),
        write_effects=(
            WriteEffect(
                tool="update_order_status",
                key_argument="order_id",
                argument_columns={"status": "status"},
                evidence_count=3,
                confidence=0.9,
            ),
            WriteEffect(
                tool="refund_order",
                key_argument="order_id",
                sets_constants={"status": "refunded"},
                evidence_count=3,
                confidence=0.9,
            ),
        ),
        evidence_count=8,
    )
    return StateSchema(entities=(customers, orders))


TOOL_CLASSES = {
    "get_customer": ToolClass.READ,
    "get_order": ToolClass.READ,
    "search_orders": ToolClass.READ,
    "refund_order": ToolClass.WRITE,
    "update_order_status": ToolClass.WRITE,
    "send_email": ToolClass.EXTERNAL,
    "escalate_to_human": ToolClass.UNKNOWN,  # no evidence: must never be faked
}


def retail_task(task_id: str = "ep-refund-ok") -> TaskCase:
    return TaskCase(
        task_id=task_id,
        trace_id="trace-0",
        instruction="I want a refund for order 7741, my customer id is 88.",
        pre_state=(
            EntityRows(
                entity="customers",
                rows=({"customer_id": 88, "name": "Ada Lovelace", "email": "ada@example.com"},),
            ),
            EntityRows(
                entity="orders",
                rows=tuple(
                    {
                        "order_id": oid,
                        "customer_id": 88,
                        "status": "delivered",
                        "total_cents": 4200,
                    }
                    for oid in ORDER_IDS
                ),
            ),
        ),
        tools=tuple(TOOL_CLASSES),
        outcome=True,
    )


def retail_surface() -> ToolSurface:
    """A declared registry, so the served tool schemas are the real ones."""
    declared = {
        "get_order": {
            "type": "object",
            "properties": {"order_id": {"type": "integer"}},
            "required": ["order_id"],
        },
        "refund_order": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "amount_cents": {"type": "integer"},
            },
            "required": ["order_id"],
        },
        "update_order_status": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "status": {"type": "string"},
            },
            "required": ["order_id", "status"],
        },
    }
    return ToolSurface(
        tools=tuple(
            ToolProfile(name=name, declared_schema=declared.get(name), tool_class=cls, call_count=3)
            for name, cls in TOOL_CLASSES.items()
        )
    )


def refund_verifier(task_id: str = "ep-refund-ok") -> Verifier:
    return Verifier(
        verifier_id="v-refund",
        task_id=task_id,
        assertions=(
            Assertion(
                kind=AssertionKind.STATE_EQUALS,
                entity="orders",
                row_key={"order_id": 7741},
                field="status",
                expected="refunded",
                description="the order the customer asked about is refunded",
            ),
            Assertion(
                kind=AssertionKind.STATE_UNCHANGED,
                entity="orders",
                row_key={"order_id": 7742},
                expected={"status": "delivered"},
                description="no collateral damage to the customer's other order",
            ),
        ),
        reviewed_by="test",
    )


class TrackingBackend(SessionBackend):
    """A backend that remembers every session it built.

    Only used to prove teardown: after shutdown, every session ever handed out
    must report ``is_open is False``. An open SQLite handle per abandoned
    rollout is how a long-lived env server dies at 3am.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sessions: list = []
        self._sessions_lock = threading.Lock()

    def _build_session(self, task):
        session = super()._build_session(task)
        with self._sessions_lock:
            self.sessions.append(session)
        return session


def make_backend(*, verifiers=True, max_steps=32, tasks=None, cls=SessionBackend):
    return cls(
        retail_schema(),
        tasks or [retail_task()],
        TOOL_CLASSES,
        surface=retail_surface(),
        verifiers={"ep-refund-ok": refund_verifier()} if verifiers else {},
        max_steps=max_steps,
    )


@pytest.fixture()
def server():
    srv = serve(make_backend()).start()
    try:
        yield srv
    finally:
        srv.shutdown()


@pytest.fixture()
def client(server):
    return EnvClient(server.base_url)


# ---------------------------------------------------------------- round trip


def test_spec_reset_step_close_round_trip(server, client):
    """The whole contract, end to end, over a real socket."""
    assert client.health().ok

    spec = client.spec()
    assert spec.task_id == "ep-refund-ok"
    assert spec.instruction.startswith("I want a refund")
    assert spec.max_steps == 32
    assert spec.env_digest and spec.verifier_digest
    names = {t.name for t in spec.tools}
    assert names == set(TOOL_CLASSES)
    # A declared schema is served verbatim, not re-derived.
    get_order = next(t for t in spec.tools if t.name == "get_order")
    assert get_order.input_schema["required"] == ["order_id"]

    started = client.reset(seed=7)
    assert started.episode_id.startswith("ep-")
    assert started.seed == 7
    assert started.observation.response["instruction"] == spec.instruction

    read = client.step(started.episode_id, "get_order", {"order_id": 7741})
    assert read.observation.status.value == "ok"
    assert read.observation.response["status"] == "delivered"
    assert read.reward == 0.0 and not read.done
    assert read.step == 1

    write = client.step(started.episode_id, "refund_order", {"order_id": 7741})
    assert write.observation.response["status"] == "refunded"
    assert write.reward == 1.0
    assert write.done is True and write.truncated is False
    assert write.info["assertions"][0]["passed"] is True

    closed = client.close(started.episode_id)
    assert closed.closed and not closed.already_closed
    assert closed.steps == 2 and closed.final_digest


def test_server_shuts_down_cleanly_and_port_is_released():
    srv = serve(make_backend()).start()
    url = srv.base_url
    EnvClient(url).reset()
    srv.shutdown()
    srv.shutdown()  # idempotent
    with pytest.raises(EnvClientError):
        EnvClient(url, timeout=2).health()


def test_error_observation_is_not_a_protocol_error(server, client):
    """A failure the real tool would also have returned is 200 + status=error."""
    ep = client.reset()
    client.step(ep.episode_id, "refund_order", {"order_id": 7743})
    again = client.step(ep.episode_id, "refund_order", {"order_id": 7743})
    assert again.observation.status.value == "error"
    assert again.observation.error_kind == "already_refunded"


# ---------------------------------------------------------------- isolation


def _final_status(client: EnvClient, episode_id: str, order_id: int) -> str:
    return client.step(episode_id, "get_order", {"order_id": order_id}).observation.response[
        "status"
    ]


def test_concurrent_episodes_do_not_share_state(server):
    """The test that matters: two episodes, interleaved writes, no leakage.

    Both episodes reset from the same task, so they start byte-identical. Each
    thread then stamps every order with a value only it uses, round by round,
    synchronized on a barrier so the writes genuinely interleave inside the
    server rather than lining up one episode after the other. If the two shared
    a store, the last writer would win and one episode would read the other's
    value -- with no error anywhere, which is exactly why this is tested rather
    than assumed.
    """
    backend = make_backend(verifiers=False)
    srv = serve(backend, max_episodes=8).start()
    try:
        client = EnvClient(srv.base_url)
        ep_a = client.reset(seed=1).episode_id
        ep_b = client.reset(seed=2).episode_id
        assert ep_a != ep_b

        rounds = 4
        barrier = threading.Barrier(2)
        errors: list[Exception] = []
        emails = {ep_a: 3, ep_b: 1}

        def run(episode_id: str, tag: str) -> None:
            try:
                for r in range(rounds):
                    barrier.wait(timeout=20)
                    for oid in ORDER_IDS:
                        out = client.step(
                            episode_id,
                            "update_order_status",
                            {"order_id": oid, "status": f"{tag}-{r}"},
                        )
                        assert out.observation.status.value == "ok"
                for _ in range(emails[episode_id]):
                    client.step(episode_id, "send_email", {"to_customer_id": 88, "tag": tag})
            except Exception as exc:  # surfaced on the main thread
                errors.append(exc)
                try:
                    barrier.abort()
                except Exception:
                    pass

        threads = [
            threading.Thread(target=run, args=(ep_a, "alpha")),
            threading.Thread(target=run, args=(ep_b, "beta")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors, errors

        # Each episode's state reflects only its own actions.
        for oid in ORDER_IDS:
            assert _final_status(client, ep_a, oid) == f"alpha-{rounds - 1}"
            assert _final_status(client, ep_b, oid) == f"beta-{rounds - 1}"

        # Effect ledgers are separate too: effects are per-session, not global.
        a_effects = client.step(ep_a, "get_order", {"order_id": 7741}).info["effects"]
        b_effects = client.step(ep_b, "get_order", {"order_id": 7741}).info["effects"]
        assert (a_effects, b_effects) == (3, 1)

        # And the two worlds hash differently, which is the state-level statement.
        a_digest = client.step(ep_a, "get_order", {"order_id": 7741}).info["env_digest"]
        b_digest = client.step(ep_b, "get_order", {"order_id": 7741}).info["env_digest"]
        assert a_digest != b_digest

        client.close(ep_a)
        client.close(ep_b)
    finally:
        srv.shutdown()


def test_many_parallel_episodes_stay_independent(server):
    """Same property under real contention: 8 rollouts, 8 threads, one server."""
    backend = make_backend(verifiers=False)
    srv = serve(backend, max_episodes=16).start()
    try:
        client = EnvClient(srv.base_url)
        results: dict[int, str] = {}
        lock = threading.Lock()
        start = threading.Barrier(8)

        def rollout(i: int) -> None:
            with client.episode(seed=i) as ep:
                start.wait(timeout=20)
                for oid in ORDER_IDS:
                    client.step(
                        ep.episode_id,
                        "update_order_status",
                        {"order_id": oid, "status": f"w{i}"},
                    )
                status = _final_status(client, ep.episode_id, ORDER_IDS[i % len(ORDER_IDS)])
                with lock:
                    results[i] = status

        threads = [threading.Thread(target=rollout, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert results == {i: f"w{i}" for i in range(8)}
        # Every episode closed itself on the way out of the context manager.
        assert len(srv.registry) == 0
    finally:
        srv.shutdown()


def test_two_threads_stepping_one_episode_are_serialized(server, client):
    """Sharing an episode is a caller error, but it must not corrupt a store."""
    ep = client.reset().episode_id
    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def hammer(tag: str) -> None:
        try:
            barrier.wait(timeout=20)
            for oid in ORDER_IDS:
                client.step(ep, "update_order_status", {"order_id": oid, "status": tag})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(t,)) for t in ("x", "y")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    # Every row holds one of the two values, whole. No torn write, no crash.
    for oid in ORDER_IDS:
        assert _final_status(client, ep, oid) in {"x", "y"}
    client.close(ep)


# ---------------------------------------------------------------- errors


def test_unknown_episode_id_is_a_structured_error(server, client):
    with pytest.raises(EnvClientError) as exc:
        client.step("ep-does-not-exist", "get_order", {"order_id": 7741})
    assert exc.value.status == 404
    assert exc.value.code is ErrorCode.NOT_FOUND
    assert "ep-does-not-exist" in exc.value.response.message
    assert "Traceback" not in exc.value.response.message


def test_unknown_episode_id_returns_json_not_html(server):
    """Straight off the socket: the body is our error model, not an HTML page."""
    req = urllib.request.Request(
        f"{server.base_url}/step",
        data=json.dumps(
            {"episode_id": "nope", "action": {"name": "get_order", "arguments": {}}}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    body = json.loads(exc.value.read())
    assert exc.value.headers["Content-Type"] == "application/json"
    assert body["error"] == "not_found"
    assert set(body) >= {"error", "message", "episode_id", "protocol_version"}


def test_closed_episode_is_409_not_404(server, client):
    ep = client.reset().episode_id
    client.close(ep)
    with pytest.raises(EnvClientError) as exc:
        client.step(ep, "get_order", {"order_id": 7741})
    assert exc.value.status == 409
    assert exc.value.code is ErrorCode.EPISODE_CLOSED
    # Closing again is fine; a crashed trainer re-issues closes on restart.
    assert client.close(ep).already_closed is True


def test_malformed_bodies_are_400_not_500(server):
    for payload in (b"{not json", json.dumps({"episode_id": 3}).encode()):
        req = urllib.request.Request(
            f"{server.base_url}/step",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 400
        assert json.loads(exc.value.read())["error"] == "bad_request"


def test_unknown_route_and_task_are_404(server, client):
    with pytest.raises(EnvClientError) as exc:
        client.spec(task_id="no-such-task")
    assert exc.value.code is ErrorCode.NOT_FOUND
    with pytest.raises(EnvClientError) as exc:
        client.reset(task_id="no-such-task")
    assert exc.value.code is ErrorCode.NOT_FOUND


def test_unsupported_tool_fails_loudly_and_is_still_advertised(server, client):
    """A tool we cannot reimplement never returns a plausible success."""
    spec = client.spec()
    escalate = next(t for t in spec.tools if t.name == "escalate_to_human")
    assert escalate.supported is False
    assert "escalate_to_human" in spec.unsupported_tools
    assert escalate.unsupported_reason

    ep = client.reset().episode_id
    with pytest.raises(EnvClientError) as exc:
        client.step(ep, "escalate_to_human", {"reason": "angry"})
    assert exc.value.status == 422
    assert exc.value.code is ErrorCode.UNSUPPORTED_TOOL
    assert exc.value.detail["tool"] == "escalate_to_human"


def test_step_after_termination_is_refused(server, client):
    ep = client.reset().episode_id
    done = client.step(ep, "refund_order", {"order_id": 7741})
    assert done.done
    with pytest.raises(EnvClientError) as exc:
        client.step(ep, "get_order", {"order_id": 7741})
    assert exc.value.code is ErrorCode.EPISODE_DONE


def test_truncation_is_not_termination(server):
    """max_steps sets truncated, never done. A trainer bootstraps differently."""
    srv = serve(make_backend(verifiers=False, max_steps=3)).start()
    try:
        client = EnvClient(srv.base_url)
        ep = client.reset().episode_id
        for _ in range(2):
            out = client.step(ep, "get_order", {"order_id": 7741})
            assert not out.done and not out.truncated
        last = client.step(ep, "get_order", {"order_id": 7741})
        assert last.truncated is True and last.done is False
        assert last.info["truncation_reason"] == "max_steps"
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- capacity


def test_live_episode_cap_is_enforced(server):
    srv = serve(make_backend(), max_episodes=2).start()
    try:
        client = EnvClient(srv.base_url)
        a = client.reset().episode_id
        client.reset()
        with pytest.raises(EnvClientError) as exc:
            client.reset()
        assert exc.value.status == 429
        assert exc.value.code is ErrorCode.EPISODE_LIMIT
        assert exc.value.detail["limit"] == 2
        assert len(srv.registry) == 2

        # Capacity is genuinely returned, not just accounted for.
        client.close(a)
        assert client.reset().episode_id
        assert len(srv.registry) == 2
    finally:
        srv.shutdown()


def test_cap_holds_under_concurrent_resets(server):
    """N threads racing to reset must not all slip past the check."""
    srv = serve(make_backend(), max_episodes=3).start()
    try:
        client = EnvClient(srv.base_url)
        refused = []
        lock = threading.Lock()
        gate = threading.Barrier(10)

        def try_reset() -> None:
            gate.wait(timeout=20)
            try:
                client.reset()
            except EnvClientError as exc:
                with lock:
                    refused.append(exc.code)

        threads = [threading.Thread(target=try_reset) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert len(srv.registry) == 3
        assert refused == [ErrorCode.EPISODE_LIMIT] * 7
    finally:
        srv.shutdown()


def test_evict_lru_mode_frees_capacity_and_closes_the_victim():
    backend = make_backend(verifiers=False, cls=TrackingBackend)
    srv = serve(backend, max_episodes=2, on_full="evict_lru").start()
    try:
        client = EnvClient(srv.base_url)
        oldest = client.reset().episode_id
        client.reset()
        client.reset()  # evicts `oldest` instead of refusing
        assert len(srv.registry) == 2
        assert srv.registry.evicted == 1
        with pytest.raises(EnvClientError) as exc:
            client.step(oldest, "get_order", {"order_id": 7741})
        assert exc.value.code is ErrorCode.EPISODE_CLOSED
        # The evicted episode's session was actually torn down.
        assert not backend.sessions[0].is_open
    finally:
        srv.shutdown()


def test_every_session_is_closed_after_shutdown():
    """No leaked SQLite handles: shutdown closes what /close did not."""
    backend = make_backend(cls=TrackingBackend)
    srv = serve(backend, max_episodes=8).start()
    client = EnvClient(srv.base_url)
    client.spec()  # the spec probe builds and closes a session too
    keep = [client.reset().episode_id for _ in range(3)]
    client.close(keep[0])
    assert any(not s.is_open for s in backend.sessions)
    assert len(srv.registry) == 2

    srv.shutdown()

    assert len(backend.sessions) >= 4
    assert all(not s.is_open for s in backend.sessions)
    assert len(srv.registry) == 0


# ---------------------------------------------------------------- MCP


def test_mcp_list_tools_returns_the_reconstructed_schemas():
    backend = make_backend()
    mcp = McpEnvServer(backend)
    tools = mcp.list_tools()
    by_name = {t["name"]: t for t in tools}
    assert set(by_name) == set(TOOL_CLASSES)
    assert by_name["update_order_status"]["inputSchema"]["required"] == ["order_id", "status"]
    assert by_name["refund_order"]["_meta"]["bandits/tool_class"] == "write"
    # Unsupported tools are advertised, with the refusal in the description.
    escalate = by_name["escalate_to_human"]
    assert escalate["_meta"]["bandits/supported"] is False
    assert "UNSUPPORTED" in escalate["description"]
    mcp.close_episode()


def test_mcp_jsonrpc_shapes():
    backend = make_backend()
    mcp = McpEnvServer(backend)
    init = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "bandits-env"
    assert init["result"]["instructions"].startswith("I want a refund")

    listed = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert len(listed["result"]["tools"]) == len(TOOL_CLASSES)

    called = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_order", "arguments": {"order_id": 7741}},
        }
    )
    assert called["result"]["isError"] is False
    assert json.loads(called["result"]["content"][0]["text"])["status"] == "delivered"

    unknown = mcp.handle({"jsonrpc": "2.0", "id": 4, "method": "no/such"})
    assert unknown["error"]["code"] == -32601
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert mcp.handle({"id": 5, "method": "tools/list"})["error"]["code"] == -32600
    mcp.close_episode()


def test_mcp_call_tool_mutates_only_its_own_episode():
    backend = make_backend(verifiers=False)
    mcp = McpEnvServer(backend)
    ep_a = mcp.open_episode()
    ep_b = mcp.open_episode()

    mcp.call_tool("update_order_status", {"order_id": 7741, "status": "mcp-a"}, episode_id=ep_a)
    mcp.call_tool("update_order_status", {"order_id": 7741, "status": "mcp-b"}, episode_id=ep_b)

    a = mcp.call_tool("get_order", {"order_id": 7741}, episode_id=ep_a)
    b = mcp.call_tool("get_order", {"order_id": 7741}, episode_id=ep_b)
    assert a["structuredContent"]["response"]["status"] == "mcp-a"
    assert b["structuredContent"]["response"]["status"] == "mcp-b"
    assert a["structuredContent"]["episode_id"] == ep_a

    # A wrong explicit episode_id errors; it never falls back to the bound one.
    missing = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_order", "arguments": {}, "episode_id": "ep-nope"},
        }
    )
    assert missing["error"]["code"] == -32602

    mcp.close_episode(ep_a)
    mcp.close_episode(ep_b)


def test_mcp_unsupported_tool_is_an_error_result_not_a_fake_success():
    mcp = McpEnvServer(make_backend())
    result = mcp.call_tool("escalate_to_human", {"reason": "angry"})
    assert result["isError"] is True
    assert result["structuredContent"]["error"] == "unsupported_tool"
    mcp.close_episode()


def test_mcp_shares_the_http_server_sessions(server, client):
    """One environment, two doors: an MCP call lands in the HTTP episode."""
    ep = client.reset().episode_id
    # Write over MCP...
    written = client.rpc(
        "tools/call",
        {
            "name": "update_order_status",
            "arguments": {"order_id": 7743, "status": "boxed"},
            "episode_id": ep,
        },
    )
    assert written["result"]["isError"] is False
    # ...read it back over HTTP: same episode, same store.
    assert client.step(ep, "get_order", {"order_id": 7743}).observation.response["status"] == "boxed"
    # And the verifier grades an MCP action exactly as it grades an HTTP one.
    graded = client.rpc(
        "tools/call",
        {"name": "refund_order", "arguments": {"order_id": 7741}, "episode_id": ep},
    )
    assert graded["result"]["structuredContent"]["reward"] == 1.0
    assert graded["result"]["structuredContent"]["done"] is True
    client.close(ep)
