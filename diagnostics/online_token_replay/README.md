# 在线Token捕获与固定Batch重放

用途：诊断V2在线Trace第23个输出Token的分歧。只支持单卡、Eager、BF16 KV、Greedy。不是数值门槛替代品，不改变V1未通过状态，不用于吞吐或延迟测量。

## 与冻结实验隔离

工具位于`diagnostics/`，不在既有`benchmarks.v2_stall.protocol.PATTERNS`中；不会修改引擎、配置、测试门槛或旧产物。独立plan/capture/replay均记录工具源码哈希，运行前后检查原suite、模型元数据及工具哈希。新增诊断文件不应改变旧suite的精确源码集合。

使用临时进程内ModelRunner包装和模型/LMHead钩子，不改变采样结果；结束时恢复包装与钩子。在线复制、CPU同步、磁盘写入会扰动下一Step的到达与Batch组成，因此捕获的是**新的诊断运行**，不是旧性能实验的历史重放。启用逐Step状态检查。

## 保存与重放

- 捕获目标输出前的完整输入、位置、Context、Block Table、Slot Mapping、采样行和各请求历史。
- 保存所有参与Batch所需物理页的有效历史KV；当前Step待写位置、未初始化尾部清零，不导出未初始化字节。单快照默认上限512MiB，超限报错，不截断。
- 捕获目标LMHead完整词表向量及实际采样Token，核验二者对应；保存Top-2差距和Logit[13]-Logit[382]。
- 每次固定Batch前向都恢复相同的有效KV快照及Context，防止前次执行写入污染。
- 目标请求单独执行保留其历史与KV，但改变Batch形状，且可能改变Attention接口；不能单凭差异定位到某个GEMM或Attention kernel。
- 重放要求实际KV池形状/类型与捕获一致，不静默重映射物理页。不运行在线Driver或采样。
- 跨捕获运行只有完整Token历史相同才计算Logits差异；历史不同单独记录，不删除。
- CPU报告核验所有产物哈希、冻结Trace、目标历史与原始输出，以及原始Logits重算指标。不将小Top-2差距自动豁免为正确。

## 验证范围

CPU测试覆盖逻辑/物理元数据、有效KV快照与逐次恢复、非有限向量拒绝、采样行映射、钩子生命周期、固定前向和共同历史比较。硬件边界使用明确标记的微型CPU张量模型；它们不是Qwen3、FlashAttention或GPU正确性证据。GPU捕获与重放由用户执行。

## 完整命令

在服务器项目根目录、GPU0空闲的tmux会话中执行。共4轮在线捕获、4个新进程的重放；每个目标执行3次完整Batch和3次单请求前向。失败即停止并保留现场，不覆盖、不自动重试。

```bash
(
set -euo pipefail
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh
export CUDA_VISIBLE_DEVICES=0

bash diagnostics/online_token_replay/test.sh --cpu

V2_REPLAY_SUITE="$PWD/results/raw/v2-coarse.msrEeU/rps-1-seed-53"
V2_REPLAY_ROOT="$(mktemp -d "$PWD/results/raw/v2-online-replay.XXXXXX")"
exec > >(tee "${V2_REPLAY_ROOT}.log") 2>&1
echo "诊断目录：$V2_REPLAY_ROOT"

bash diagnostics/online_token_replay/test.sh --prepare \
  --suite "$V2_REPLAY_SUITE" --mode poisson \
  --target-ids r00000014 r00000018 --output-token 23 \
  --repeats 3 --out "$V2_REPLAY_ROOT/plan"

for repeat in 1 2; do
  if [[ "$repeat" == 1 ]]; then
    policies=(baseline chunk-512)
  else
    policies=(chunk-512 baseline)
  fi
  for policy in "${policies[@]}"; do
    base="$V2_REPLAY_ROOT/repeat-$repeat/$policy"
    mkdir -p "$base"
    bash diagnostics/online_token_replay/test.sh --capture \
      --suite "$V2_REPLAY_SUITE" --mode poisson --policy "$policy" \
      --target-ids r00000014 r00000018 --output-token 23 \
      --out "$base/capture"
    bash diagnostics/online_token_replay/test.sh --replay \
      --source "$base/capture" --repeats 3 --out "$base/replay"
  done
done

bash diagnostics/online_token_replay/test.sh --report --root "$V2_REPLAY_ROOT"
echo "完成：$V2_REPLAY_ROOT"
)
```

结果：`plan/plan.json`、`repeat-*/<policy>/capture/snapshot-*.pt`、同目录JSON元数据、`replay/report.json`及原始向量、`report.*/report.md/summary.json/cross_capture.json`。固定重复不稳定、无法复现捕获Top1、单请求差异、跨运行分歧分别报告，退出0只表示诊断采集/审计完成，不表示数值正确性通过。

如果诊断未复现原Token分歧，只能说明此次插桩时序未复现；不能写为修复。若固定Batch重放无法复现捕获向量，优先检查快照完整性、环境和重放路径，不先推断为模型浮点误差。必要时继续最小化或加入逐层独立KV/Attention参考检查。
