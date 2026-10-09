#!/usr/bin/env bash
# 用source载入当前Shell；不安装环境，不启动推理，不改变NCCL传输。
_slo_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V0_PYTHON="${V0_PYTHON:-$_slo_project_root/.venv/bin/python}"
if ! command -v "$V0_PYTHON" >/dev/null 2>&1; then
    echo "未找到解释器：$V0_PYTHON，请先创建环境或显式设置V0_PYTHON" >&2
    return 2
fi
if [[ -z "${V0_MODEL:-}" ]]; then
    if ! V0_MODEL="$("$V0_PYTHON" "$_slo_project_root/benchmarks/defaults.py" --field model_path)"; then
        return 2
    fi
fi
if [[ -z "${V0_MODEL_REVISION:-}" ]]; then
    if ! V0_MODEL_REVISION="$("$V0_PYTHON" "$_slo_project_root/benchmarks/defaults.py" --revision-for "$V0_MODEL")"; then
        return 2
    fi
fi
export V0_PYTHON V0_MODEL V0_MODEL_REVISION
unset _slo_project_root
