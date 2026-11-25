#!/usr/bin/env bash
# l34 reference preset: Qwen3.8-Flash-Next on 4 Intel Arc Pro B70 cards.
# x2 W4A8 checkpoint, INT8 execution, INT8 per-token-head KV, DFlash2 with a W8A8 drafter (7 tokens),
# 16 sequences, 262,144-token context, n-gram table in host RAM, 128 GiB host KV tier, EPLB observing.
# This is the command line the reference engine served with; host paths are placeholders:
#   VENV           venv with torch 2.15.0+xpu, the l34 kernels wheel, triton_xpu, intel-sycl-rt 2026.2.0
#   VLLM_TREE      checkout of Lumnus/vllm l34
#   MODEL_DIR      the x2 checkpoint
#   DRAFTER_DIR    the W8A8 DFlash2 drafter
#   PLE_INT8_TABLE ple_ngram_int8_rowscale.safetensors
#   CHAT_TEMPLATE  the model's chat template
#   STATE_DIR      a writable directory for the EPLB placement (kept across restarts)
#   CACHE_DIR      a writable cache directory
# Run it under compute-runtime 26.35 (put its libraries first on LD_LIBRARY_PATH).
# A first start has no saved expert placement and decodes at less than half speed until EPLB has moved under load:
#   python $VLLM_TREE/tools/runtime_control.py --url http://127.0.0.1:8000 eplb set --step-interval 300
# for a few minutes of traffic, then back to observing (--step-interval 2000000000). The placement is saved in STATE_DIR.
set -euo pipefail
: "${VENV:?}" "${VLLM_TREE:?}" "${MODEL_DIR:?}" "${DRAFTER_DIR:?}" "${PLE_INT8_TABLE:?}" "${CHAT_TEMPLATE:?}" "${STATE_DIR:?}" "${CACHE_DIR:?}"

export PATH="$VENV/bin:$PATH" LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="$VLLM_TREE"
# engine paths (all on in the reference engine)
export VLLM_XPU_MOE_ENTRY_QUANT=1 VLLM_XPU_MOE_SHARED_IN_GATHER=1 VLLM_XPU_HC_UP_FUSED=3 VLLM_XPU_HC_DOWN_FOLD=1
export VLLM_XPU_W4A8_FUSED_MOE=1 VLLM_XPU_ATTN_META_ONCE=1 VLLM_XPU_MOE_TOPK_ROUTE=1 VLLM_XPU_MOE_ROUTE_FUSED=1
# runtime control routes (/runtime_control) and the per-layer inspection log
export VLLM_SERVER_DEV_MODE=1 VLLM_LOG_MODEL_INSPECTION=1
# Han mask, decided per request from the user's latest message
export B70_CJK_MASK=1 B70_CJK_MASK_DEFAULT=auto
# cards and collectives
export ZE_AFFINITY_MASK=0,1,2,3 ZE_FLAT_DEVICE_HIERARCHY=FLAT CCL_ZE_IPC_EXCHANGE=sockets CCL_SYCL_ALLREDUCE_ARC=1
export CCL_SYCL_ALLGATHERV_TMP_BUF=1 CCL_SYCL_ALLREDUCE_TMP_BUF=1 CCL_SYCL_REDUCE_SCATTER_TMP_BUF=1 CCL_SYCL_BROADCAST_TMP_BUF=1
export CCL_SYCL_ALLTOALL_TMP_BUF=1 CCL_SYCL_ALLREDUCE_LL_THRESHOLD=65536
# host memory: exact pinned sizes capped at 1 GiB per cached block; lift the 30 GiB pinned-allocation limit
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:1024
export NEOReadDebugKeys=1 EnableDeferBacking=0 AllowUnrestrictedSize=1 ExperimentalCopyThroughLock=0
export ExperimentalH2DCpuCopyThreshold=0 ExperimentalD2HCpuCopyThreshold=0
export VLLM_TARGET_DEVICE=xpu VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$CACHE_DIR/tmp" TRITON_CACHE_DIR="$CACHE_DIR/triton" VLLM_CACHE_ROOT="$CACHE_DIR/vllm" XDG_CACHE_HOME="$CACHE_DIR/xdg"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$STATE_DIR"

exec python -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
  --served-model-name qwen38-flash-next \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 6144 \
  --kv-cache-dtype int8_per_token_head --attention-backend FLASH_ATTN \
  --attention-config '{"int8_prefill_dot":"sycl_qk_pv"}' \
  --enable-prompt-tokens-details \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template "$CHAT_TEMPLATE" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,256,512,1024]}' \
  --cudagraph-metrics \
  --kv-offloading-size 128 --kv-offloading-backend native \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFTER_DIR\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":4,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"attention_backend\":\"TRITON_ATTN\"}" \
  --engram-config "{\"cpu_offload\":true,\"int8_table_path\":\"$PLE_INT8_TABLE\"}" \
  --hf-overrides '{"text_config": {"indexer_n_heads": null}}' \
  --quantization-config '{"targets":{"re:.*lm_head$":"int8_per_channel","re:.*\\.mlp\\.experts$":{"activation":"int8_per_token"}}}' \
  --override-generation-config '{"temperature":0.7}' \
  --reasoning-config '{"reasoning_end_str": "\n\nI have thought enough; time to answer.\n</think>\n\n"}' \
  --kv-cache-release-chunk-mib 2 \
  --enable-eplb \
  --eplb-config "{\"policy\":\"incremental\",\"use_async\":true,\"max_moves_per_rank\":8,\"step_interval\":2000000000,\"window_size\":1000,\"communicator\":\"torch_gloo\",\"log_balancedness\":true,\"log_balancedness_interval\":100,\"state_path\":\"$STATE_DIR/eplb.json\"}" \
  --mm-encoder-host-towers '{"vision_tower": {"threads": 16, "residency": "dynamic", "promote_after": 2, "promote_window_s": 600, "demote_idle_s": 1800, "idle_location": "host", "host_available_floor_gib": 8}}'
