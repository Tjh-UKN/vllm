# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_trace import BlockAllocated, KVTrace, create_kv_trace


@pytest.mark.parametrize(
    "additional_config",
    [{}, {"other_feature": True}, {"kv_trace": {}}, {"kv_trace": {"enabled": False}}],
)
def test_disabled_configuration_creates_no_recorder(additional_config):
    assert create_kv_trace(additional_config) is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (None, "must be a mapping"),
        (True, "must be a mapping"),
        ({"enabled": "true"}, "must be a boolean"),
        ({"enabled": 1}, "must be a boolean"),
        ({"enabled": False, "enable": True}, "Unknown kv_trace options"),
    ],
)
def test_invalid_configuration_is_not_silently_disabled(options, message):
    with pytest.raises(ValueError, match=message):
        create_kv_trace({"kv_trace": options})


def test_enabled_engines_have_independent_histories_and_identities():
    first = create_kv_trace({"kv_trace": {"enabled": True}})
    second = create_kv_trace({"kv_trace": {"enabled": True}})
    assert first is not None and second is not None
    first.record_allocation("gpu", [42])
    first.record_allocation("gpu", [42])
    second.record_allocation("gpu", [42])

    first_batch = first.take_events()
    second_batch = second.take_events()
    assert first_batch.trace_id != second_batch.trace_id
    assert first_batch.events == (
        BlockAllocated(0, "gpu", 42, 1),
        BlockAllocated(1, "gpu", 42, 2),
    )
    assert second_batch.events == (BlockAllocated(0, "gpu", 42, 1),)


def test_reallocation_keeps_pool_identity_and_survives_drain():
    trace = KVTrace()
    trace.record_allocation("gpu", [42])
    first = trace.take_events()
    trace.record_allocation("cpu", [42])
    trace.record_allocation("gpu", [42])
    second = trace.take_events()

    assert second.trace_id == first.trace_id
    assert second.events == (
        BlockAllocated(1, "cpu", 42, 1),
        BlockAllocated(2, "gpu", 42, 2),
    )
    assert second.dropped_sequences is None


def test_overflow_reports_exact_gap_without_resetting_generations():
    trace = KVTrace(max_events=2)
    for _ in range(5):
        trace.record_allocation("gpu", [42])
    batch = trace.take_events()

    assert batch.dropped_sequences == (0, 3)
    assert batch.events == (
        BlockAllocated(3, "gpu", 42, 4),
        BlockAllocated(4, "gpu", 42, 5),
    )
    empty = trace.take_events()
    assert empty.events == ()
    assert empty.dropped_sequences is None

    for _ in range(3):
        trace.record_allocation("gpu", [42])
    resumed = trace.take_events()
    assert resumed.dropped_sequences == (5, 6)
    assert resumed.events == (
        BlockAllocated(6, "gpu", 42, 7),
        BlockAllocated(7, "gpu", 42, 8),
    )


def test_retained_events_do_not_borrow_mutable_inputs():
    trace = KVTrace()
    block_ids = [41, 42]
    trace.record_allocation("gpu", block_ids)
    block_ids[:] = [99]
    batch = trace.take_events()
    trace.record_allocation("gpu", [41])

    assert batch.events == (
        BlockAllocated(0, "gpu", 41, 1),
        BlockAllocated(1, "gpu", 42, 1),
    )


@pytest.mark.parametrize("max_events", [0, -1, True, 1.5])
def test_retention_requires_a_positive_integer_capacity(max_events):
    with pytest.raises(ValueError, match="must be a positive integer"):
        KVTrace(max_events=max_events)
