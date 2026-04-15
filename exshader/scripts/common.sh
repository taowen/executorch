#!/usr/bin/env bash

exshader_find_default_qwen3_pte() {
  local pte_dir="$1"
  local preferred
  preferred=$(
    ls -1 "$pte_dir"/qwen3_0_6b_vulkan_silu_emb4bit_8da4w*.pte 2>/dev/null | head -n1 || true
  )
  if [[ -n "${preferred:-}" ]]; then
    basename "$preferred"
    return 0
  fi

  local latest
  latest=$(ls -1t "$pte_dir"/qwen3_0_6b_vulkan*.pte 2>/dev/null | head -n1 || true)
  if [[ -n "${latest:-}" ]]; then
    basename "$latest"
    return 0
  fi
  return 1
}

exshader_find_qwen3_tokenizer() {
  ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1
}

exshader_resolve_qwen3_pte_name() {
  local pte_dir="$1"
  local cli_name="${2:-}"
  if [[ -n "${cli_name:-}" ]]; then
    echo "$cli_name"
    return 0
  fi
  exshader_find_default_qwen3_pte "$pte_dir"
}
