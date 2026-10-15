# V3 第一检查点：Step 时延采集与 P95 查表预测器

输入版本：服务器 `665914e`。本次新增独立入口，不修改既有 V0/V1/V2 源码、调度规则或历史实验。

本检查点实现受控真实批次采集、P95 分桶建表、不同 seed 的独立受控评测。尚未实现 Slack、Hysteresis、Aging、KV Guard；不同 seed 的同矩阵评测不代表已验证独立在线负载分布。接入控制器前还须在独立在线 Trace 上验证覆盖率和欠预测风险。所有 GPU 结果待实际运行，CPU 测试不能作为 GPU 性能或完整模型数值验收证据。

## 1. 文件与接口

| 文件 | 职责 |
|---|---|
| `nanovllm/engine/step_predictor.py` | 纯标准库特征、分桶、P95、支持范围及预测接口 |
| `benchmarks/profile_step_time.py` | collect / fit / evaluate / inspect 命令 |
| `benchmarks/v3_profile/timing.py` | 与未来 Slack 时钟一致的计时边界 |
| `benchmarks/v3_profile/driver.py` | 通过真实调度器构造有限受控场景并 drain |
| `benchmarks/v3_profile/collect.py` | 单卡 Eager/BF16 采集与错误保留 |
| `benchmarks/v3_profile/protocol.py` | 配置检查、运行身份、数据完整性检查 |
| `benchmarks/v3_profile/report.py` | 建表与独立误差评测 |
| `scripts/test_v3_profile.sh` | 测试入口 |
| `tests/test_v3_profile.py` | CPU 计时契约、真实调度状态、估计器及产物测试 |

```python
predictor = StepPredictor.load("predictor.json", runtime_identity)
prediction = predictor.predict(batch_features)
# prediction.predicted_ms / supported / sample_count / reason
```

模型、模型元数据、软件包、GPU 属性、驱动、TP、KV 类型、执行后端、KV 容量、源码集合和计时契约必须匹配，禁止跨执行环境直接套用表。模型权重沿用冻结 revision 与文件大小清单，不额外对全部权重做内容哈希；使用者应保持本地权重未变。

## 2. 计时与特征

```text
start_ns：进入 schedule() 前
scheduled_ns：schedule()/build_batch() 返回
run_end_ns：run_mixed() 返回，KV 写入完成，采样 token 已到 CPU
visible_ns：postprocess() 返回
step_time_ms = (visible_ns - start_ns) / 1e6
```

包含调度、metadata、H2D、Forward、采样和 postprocess；不包含请求提交、统计、状态扫描、JSON 写盘。`run_mixed()` 原路径在采样 `.tolist()` 时形成 CPU 可见边界，无采样中间 Chunk 则已有显式同步，不额外在计时区间添加同步。阶段耗时不是纯 GPU kernel 耗时。

记录实际 batch 特征：`decode_count / prefill_tokens / prefill_request_count / sum_context_tokens / max_context_len / sample_count / prefill_has_history / graph_mode`。context 使用每行 `end_pos`，不是 Prompt 原始长度；recompute 计入 Prefill 工作量。

增加 `sample_count` 是因为中间 Chunk 会跳过 LMHead；增加 `prefill_has_history` 是因为有历史时走 KV Cache 读取路径。桶宽默认为 P=256、上下文总量=2048、最大上下文=256，向上取整；其他特征精确匹配。聚合特征无法完全描述所有 query/context 分布，须以独立误差评测检验，不能把分桶视为时延上界。

建表使用 nearest-rank P95：排序后取第 `ceil(0.95*N)` 个样本。默认每桶至少 30 个非 warmup 样本。稀疏或未知桶返回 `supported=False`，诊断估时使用训练桶 P95 的最大值；该回退值不保证保守，后续控制器必须显式处理不支持的候选。即使 `supported=True` 也不保证 SLO。

## 3. 受控场景

默认矩阵：D=0/1/8，P=0/256/512，Decode 初始上下文=256/1024，Prefill 请求数=1/2。无意义组合去重后共 24 个场景。每场景先执行 3 个 warmup episode，再测量 50 个 episode；场景执行顺序按 seed 打乱。

对有 Prefill 的场景，每个新请求的 Prompt 为 `2*(P/M)+1`，输出 2 token，通常形成四个 Step：

1. 首个中间 Chunk，无 Prefill 采样。
2. 带历史的中间 Chunk，无 Prefill 采样。
3. 最后一个短 Chunk，产生 Prefill 首 token。
4. 新请求 Decode 一次并结束。

因此第四轮实际 Decode 数可能为 D+M，第三轮实际 P 小于配置 P。必须以保存的实际特征建表，不以配置值代替。基础 Decode 请求用真实 Prefill 引导；引导和最终 drain 不进入训练样本。基础请求的 context 随生成增长，不假装每轮都精确等于初始长度。

所有 KV 都由真实模型计算，不手工填充缓存。随机 token 排除 tokenizer special ids、使用 greedy 与固定输出长度；这是调度性能输入，不是自然语言质量测试。测试容量固定为 256 个 KV 块；不足、抢占、基础请求意外结束、drain 失败、源码变化均保留错误并阻止使用该数据建表。256 页是本检查点受控容量，不是 V2 的服务饱和容量。

## 4. 执行命令

以下全部在服务器同一个 bash 会话运行。选择一张空闲 GPU；示例使用 GPU 0。新目录不可覆盖，失败后创建新的 suite，不删除或改写原证据。

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
source scripts/env_qwen3_4b.sh

# 无 GPU 计算：CPU 回归、默认矩阵可行性检查。
bash scripts/test_v3_profile.sh --cpu
bash scripts/test_v3_profile.sh --collect \
  --split train --seed 411 --out /tmp/v3-plan-not-created --plan-only

mkdir -p results/raw
V3_SUITE="$(mktemp -d "$PWD/results/raw/v3-step-profile.XXXXXX")"
printf 'V3_SUITE=%s\n' "$V3_SUITE"

# GPU 小规模烟雾：5 场景，1 warmup + 2 measured episode。
bash scripts/test_v3_profile.sh --smoke --out "$V3_SUITE/smoke"
bash scripts/test_v3_profile.sh --inspect --profiles "$V3_SUITE/smoke"
```

烟雾预期：三个 phase 都出现、`has_intermediate_prefill=true`、`has_prefill_history=true`，共 28 个测量 Step（含 warmup 共 42 个）。它只检查执行闭环；禁止用 smoke 建正式预测器。

烟雾通过后，固定同一份代码与环境采集正式数据：

```bash
# 默认 24 场景；50 measured + 3 warmup episode。
bash scripts/test_v3_profile.sh --collect \
  --split train --seed 411 --out "$V3_SUITE/train-411"

bash scripts/test_v3_profile.sh --fit \
  --profiles "$V3_SUITE/train-411" \
  --min-samples 30 --out "$V3_SUITE/predictor"

# 新进程、不同 seed；不向预测器追加这些样本。
bash scripts/test_v3_profile.sh --collect \
  --split eval --seed 511 --out "$V3_SUITE/eval-511"

bash scripts/test_v3_profile.sh --evaluate \
  --predictor "$V3_SUITE/predictor/predictor.json" \
  --profiles "$V3_SUITE/eval-511" --out "$V3_SUITE/evaluation"

cat "$V3_SUITE/predictor/fit_summary.json"
cat "$V3_SUITE/evaluation/report.json"
```

默认每个正式 profile 有 4200 个测量 Step，含 warmup 共 4452 个；另有不入样本的引导/drain。运行期间不要修改源码或同时启动其他 GPU 作业。不承诺固定运行时长。

需要增大覆盖范围时，采集前为训练和评测同时指定同一矩阵，例如 `--decode-counts 0,1,8,16 --prefill-budgets 0,256,512,1024 --contexts 256,1024,3072`。程序会先检查长度、全局预算、请求数与页容量；若检查失败，需要事先调整矩阵/测试页容量，不能静默跳过失败场景。新的矩阵或参数必须使用新 suite。

若需要多 seed 聚合，可采集 train-412/train-413，再将多个路径一起传给 `--fit --profiles ...`；评测相应使用 512/513。训练与评测 seed 必须不重叠；也禁止将相同 Step 文件改名用于评测。

## 5. 读取结果和失败处理

主要关注：

- `coverage`：独立测量样本中有充分训练支持的比例。
- `supported.underprediction_rate`：支持范围内实际耗时超过预测值的比例。
- `supported.absolute_error_p95_ms`：绝对误差 P95。
- `supported.underprediction_p95_ms / max_underprediction_ms`：低估幅度。
- `unsupported` 和 `all_including_unsupported`：保留未覆盖样本，避免只报告好看的子集。
- `per_phase`：区分纯 Decode、纯 Prefill、Mixed。
- `predict_cpu_p95_us`：本机预测调用 CPU 成本，包含特征校验/查表，不包含后续候选构造。

低覆盖率说明矩阵或桶支持不足；欠预测率高、低估幅度大说明需要调整特征、分桶或采集覆盖。不要用测试集反复调参后继续称其独立评测：改表后应使用新的独立 seed/Trace。

数据存于 `results/raw/`，已被现有 `.gitignore` 排除；不保存完整 logits，不上传权重或原始实验产物。profile 包含 protocol/environment/steps/summary 及 SHA256 清单；失败保留 `error.json`，不会写有效 summary，也不能参与 fit/evaluate。

本检查点退出需要 CPU 测试、GPU 烟雾、正式采集与独立误差报告齐全。预测误差是否足以支撑选定 SLO，还需结合下一阶段的安全余量与在线 Trace 评测判断；本阶段不宣称 SLO 收益。
