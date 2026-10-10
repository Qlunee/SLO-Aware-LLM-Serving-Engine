# V1首次分叉：逐层KV与Attention取证

此入口只用于排查`curated-6`第11个输出Token的首次分叉，不修改模型计算、采样器、分页KV或验收阈值。新增目录`benchmarks/v1_diagnostics`与独立CPU测试，不改动旧验证入口的源码哈希集合。

## 已有证据

`results/raw/v1-first-divergence.005ytr8a`已复现原质量实验。三条路径第11个Token之前的历史相同：原始批处理选择`logical`，原始串行与Mixed选择`data`。后两条路径的两个候选Logits并列；这解释了Greedy为何分叉，但还没有证明产生Logits差异的底层原因。

## 执行

服务器已有`.venv`和Qwen3-4B权重，不需要安装依赖。不要并行启动其他引擎，不要在运行期间保存源码修改。

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh

# CPU探针自检：包含页重排、旧KV破坏检测、Hook不改变输出。
"$V0_PYTHON" -B -m unittest discover -s tests -p test_layer_replay_diagnostic.py -v

# 用户执行GPU重放。保留原8条请求、每条128-token输出及执行顺序。
CUDA_VISIBLE_DEVICES=0 "$V0_PYTHON" -B -m benchmarks.v1_diagnostics.layer_replay \
  --source results/raw/v1-quality.jp5c_dbl \
  --target-id curated-6 \
  --output-token 11
```

`--output-token`从1开始计数。默认CPU线程数为1，以降低小矩阵参考计算的线程开销；GPU精度设置不变。这是带同步、Hook与CPU拷贝的正确性取证，不能使用其时间衡量性能。

## 比较边界

1. 从Attention输入捕获已做Q/K Norm、RoPE的K和V，按请求的**逻辑位置**写入独立CPU账本，不使用生产Slot Mapping来构建参考。
2. 每次该请求执行后，按页表还原有效KV，与独立账本逐元素精确比较。未知Prefix条目会直接报错，不能用缓存自己填充参考。本案例预计从冷Prefill开始。
3. 检查输入Token、位置、计算进度、Slot、Attention Context长度与页表。第11个Token逐层保存Q、投影K/V、有效缓存、实际Attention输出。
4. 用相同Q及独立账本，在CPU上做显式FP32 GQA与因果Attention，对照实际BF16输出；不调用FlashAttention，不改变生产输出。
5. 跨路径只在Token历史相同时比较各层张量；原始批处理、原始串行与Mixed分别拥有独立账本。
6. 验证加入探针后三条路径的完整输出仍与旧报告一致，否则不做原实验归因。

## 产物与阅读顺序

自动新建`results/raw/v1-layer-replay.*`，不覆盖旧结果。

- `report.json`：逐步状态、逐层KV精确性、Attention参考误差、跨路径逐层差异与重放一致性。
- `{legacy_batch,legacy_serial,mixed}/layer_00.pt`至`layer_35.pt`：第11个Token的逐层证据，层编号从0开始。
- 各路径`logits.pt`：固定该步的输入历史与Logits。
- `environment.json`、`config.json`、`artifact_hashes.json`：源码、环境与递归产物摘要。
- `error.json`：若有异常，保留已采集证据并记录错误。

先看`source_unchanged`和`reproduced_saved_outputs`，再看各路径：

| 字段 | 含义 |
|---|---|
| `all_state_checks_match` | 本请求已检查的Token/位置/Slot/上下文等与逻辑状态一致 |
| `all_shadow_kv_exact` | 已执行目标请求的有效缓存与捕获投影值完全一致 |
| `all_attention_within_existing_probe_tolerance` | 第11个Token所有层落在已有Attention探针的`atol=.02,rtol=.02`容差内 |

任何`false`都要回到具体Step/层分析；退出码0仅表示采集完成，不表示所有字段为真，更不表示V1通过。源码变化或未能复现旧输出会以非零状态退出。

## 不能得出的结论

即使本案例全部匹配，也仅缩小该分叉的原因范围，不证明所有请求、跨页、抢占、Prefix、Graph或TP=2正确。独立账本验证KV写入/保留，不验证投影本身的数学正确性。跨路径KV值不同本身不等于寻址错误：投影可能已有BF16数值差异。

没有替换旧Logits门槛，没有将Top-1分歧改成“通过”，没有因并列而更改采样器。是否调整正式跨形状验收协议，需另行解释、冻结规则并重新验证。
