# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-local collection and logging of native KV allocation events."""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from uuid import uuid4

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class BlockAllocated:
    """One observed allocation, not a claim about KV contents or references."""

    sequence: int
    pool_id: str
    block_id: int
    generation: int


@dataclass(frozen=True, slots=True)
class KVTraceBatch:
    """Retained events and an optional half-open range of dropped sequences."""

    trace_id: str
    events: tuple[BlockAllocated, ...]
    dropped_sequences: tuple[int, int] | None


class KVTrace:
    """Single-producer recorder with bounded history and engine-local identity.

    Call `record_allocation` only after the native allocator hands blocks out
    for a new lifetime. Prefix sharing and reference release do not allocate.
    Reference and execution-plan events are introduced separately.
    """

    def __init__(self, max_events: int = 4096) -> None:
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self.trace_id = uuid4().hex
        self._events: deque[BlockAllocated] = deque(maxlen=max_events)
        self._generations: dict[tuple[str, int], int] = {}
        self._next_sequence = 0
        self._dropped_events = 0

    def record_allocation(self, pool_id: str, block_ids: Iterable[int]) -> None:
        """Record new allocation identities in native allocation order."""
        for block_id in block_ids:
            key = (pool_id, block_id)
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            if len(self._events) == self._events.maxlen:
                self._dropped_events += 1
            self._events.append(
                BlockAllocated(
                    sequence=self._next_sequence,
                    pool_id=pool_id,
                    block_id=block_id,
                    generation=generation,
                )
            )
            self._next_sequence += 1

    def take_events(self) -> KVTraceBatch:
        """Drain retained history without resetting allocation identities."""
        events = tuple(self._events)
        dropped_sequences = None
        if self._dropped_events:
            first_retained = events[0].sequence
            dropped_sequences = (
                first_retained - self._dropped_events,
                first_retained,
            )
        batch = KVTraceBatch(self.trace_id, events, dropped_sequences)
        self._events.clear()
        self._dropped_events = 0
        return batch

    def log_events(self) -> None:
        """Drain pending events to the engine's existing log output."""
        batch = self.take_events()
        if batch.events or batch.dropped_sequences is not None:
            logger.info("KV_TRACE %s", batch)


def create_kv_trace(additional_config: Mapping[str, object]) -> KVTrace | None:
    """Create a recorder only for explicit `kv_trace.enabled=true`."""
    if "kv_trace" not in additional_config:
        return None
    options = additional_config["kv_trace"]
    if not isinstance(options, Mapping):
        raise ValueError("kv_trace must be a mapping")
    unknown = options.keys() - {"enabled"}
    if unknown:
        raise ValueError(f"Unknown kv_trace options: {sorted(unknown)}")
    enabled = options.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("kv_trace.enabled must be a boolean")
    return KVTrace() if enabled else None
