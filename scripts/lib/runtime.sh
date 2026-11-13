#!/usr/bin/env bash

# Shared runtime helper for portable launchers. Source this file after ROOT is
# resolved, then call ovb_run or ovb_run_python. USE_CONTAINER defaults to 0;
# setting it to 1 requires SIF. Existing callers may still set PYTHON to the
# interpreter path inside their container. OVB_CONTAINER_GPU=1 requests GPU
# passthrough when the selected container engine supports it.

ovb_is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

ovb_require() {
  local name=$1
  local value=${2:-}
  if [[ -z "$value" ]]; then
    echo "ERROR: set $name before running this command." >&2
    return 2
  fi
}

ovb_container_exec() {
  local use_gpu=0
  if [[ "${1:-}" == "--nv" ]]; then
    use_gpu=1
    shift
  fi

  local sif=${1:-}
  [[ -n "$sif" ]] || {
    echo "ERROR: ovb_container_exec requires a container image path." >&2
    return 2
  }
  shift
  [[ -f "$sif" ]] || {
    echo "ERROR: container image not found: $sif" >&2
    return 2
  }

  local -a flags=(exec)
  if (( use_gpu )); then
    flags+=(--nv)
  fi

  # Apptainer is the current project name; Singularity remains common on older
  # clusters. Direct SIF execution is retained as a final compatibility path.
  if command -v apptainer >/dev/null 2>&1; then
    apptainer "${flags[@]}" "$sif" "$@"
  elif command -v singularity >/dev/null 2>&1; then
    singularity "${flags[@]}" "$sif" "$@"
  elif [[ -x "$sif" ]]; then
    "$sif" "$@"
  else
    echo "ERROR: USE_CONTAINER=1, but neither apptainer nor singularity is available." >&2
    echo "Install a container runtime, make the SIF directly executable, or set USE_CONTAINER=0." >&2
    return 127
  fi
}

ovb_run() {
  if ovb_is_true "${USE_CONTAINER:-0}"; then
    ovb_require SIF "${SIF:-}" || return
    local command root_q
    local -a container_args=()
    printf -v command '%q ' "$@"
    printf -v root_q '%q' "$ROOT"
    if ovb_is_true "${OVB_CONTAINER_GPU:-0}"; then
      container_args+=(--nv)
    fi
    ovb_container_exec "${container_args[@]}" "$SIF" /bin/bash -lc \
      "set -euo pipefail; cd $root_q; $command"
  else
    (cd "$ROOT" && "$@")
  fi
}

ovb_run_python() {
  ovb_run "${PYTHON:-python}" "$@"
}
