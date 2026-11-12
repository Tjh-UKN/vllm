# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_llm_configuration_reaches_native_allocation_logs(
    vllm_runner, caplog_mp_spawn, enabled
):
    with (
        caplog_mp_spawn("INFO") as logs,
        vllm_runner(
            "vllm-project/tiny-random-LlamaForCausalLM",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            gpu_memory_utilization=0.1,
            additional_config={"kv_trace": {"enabled": enabled}},
        ) as model,
    ):
        outputs = model.generate_greedy(["Hello"], max_tokens=2)
        assert outputs[0][0]
    text = logs.text
    assert ("KV_TRACE KVTraceBatch(" in text) == enabled
    if enabled:
        assert "BlockAllocated(" in text
        assert "generation=1" in text
