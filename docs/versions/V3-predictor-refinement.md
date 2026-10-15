# V3 预测器补采与独立在线分布复评

本检查点延续 `V3-step-predictor.md`，不新增正式版本，不改变 V1/V2 执行源码，不实现 Slack 控制器。GPU 测试由用户执行。

## 动机与范围

原实验 `results/raw/v3-step-profile.QuNaVO`：单卡 RTX 3090、Eager、BF16；seed411 训练、seed511 评测；评测覆盖率 86.43%，570 个 Step 因训练桶样本少于 30 而不受支持。支持范围内欠预测率 6.64%，最大低估 2.03ms。这些是该实验的观测，不是延迟保证。

本检查点保持原矩阵、分桶和最低 30 样本要求：补采 412/413，复用 411 聚合建表，用新 seed512 独立受控评测。预计仍可能存在少量稀疏桶，不能预先宣称覆盖率达标。

另外固定三份新的在线 Trace：Poisson612、Burst613、长短混合614；每份 2s warmup 到达、8s测量到达、60s有限 drain、平均2req/s、固定32输出token，输入256/1024token。沿用 V2 Trace/Driver，通过固定 Chunk256、全局4096 Token Budget 的 V1 Mixed 路径执行。不同 Trace 只用于评测，不参与建表。

这是预测器对在线分布的检查，不比较自适应调度与基线，不构成 SLO/Goodput 收益证据。每种在线分布仅一个 seed；受控矩阵只含部分并发数、P<=512和约1K上下文，在线低覆盖率可能暴露新的形状缺口。后续需要有针对性的训练数据，并再换独立 seed/Trace 复评。

## 源码兼容性与文件

原采集器将 `nanovllm / benchmarks / scripts / tests / configs` 源码集合冻结入运行身份；改动后不允许直接聚合旧411。因此新增编排和在线适配器位于 `tools/`，冻结入本次 `plan.json.workflow_sources`；原执行源码集合仍须与411完全一致。两份源码冻结分别检查，未放宽已有预测器的身份校验。

| 新文件 | 职责 |
|---|---|
| `tools/v3_refine.py` | 冻结输入、串行采集/聚合/复评、阶段恢复、最终报告 |
| `tools/v3_online_profile.py` | 固定策略在线执行与时延预测影子评测 |
| `tools/test_v3_refine.sh` | CPU / prepare / run / verify 入口 |
| `tools/tests/test_v3_refine.py` | CPU 冻结、恢复、完整编排与真实调度适配器测试 |

在线仍使用原 `timed_step()`：从 schedule 前至 postprocess 后计时。诊断预测在计时终点之后执行，不影响本轮选择；这些结果不能代替未来 Slack 控制器的候选构造与决策成本标定。token 时间戳使用 postprocess 完成时刻，统计处理、预测和写盘不会延后这个可见时刻，但它们的 CPU 开销会影响下一轮开始时间。

Step 按开始时刻归入 warmup/window/drain，窗口与 drain 的误差、覆盖率分别报告。全部请求、超时、拒绝、未完成与错误均保留。

## 命令

在服务器同一个 bash 会话运行，选择空闲 GPU。父实验保持原样，新的 suite 需要空目录。

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

bash tools/test_v3_refine.sh --cpu

V3_REFINE="$(mktemp -d "$PWD/results/raw/v3-refine.XXXXXX")"
echo "$V3_REFINE"
bash tools/test_v3_refine.sh --prepare \
  --parent "$PWD/results/raw/v3-step-profile.QuNaVO" \
  --suite "$V3_REFINE"

bash tools/test_v3_refine.sh --verify --suite "$V3_REFINE"
time bash tools/test_v3_refine.sh --run --suite "$V3_REFINE"

bash tools/test_v3_refine.sh --verify --suite "$V3_REFINE"
cat "$V3_REFINE/final-report/report.json"
```

prepare/verify/CPU 不执行 GPU 推理。run 串行执行：train412 → train413 → 聚合411/412/413 → collect eval512 → 受控评测 → 在线Poisson → 在线Burst → 在线长短混合 → 最终报告。三次新受控采集合计12,600个测量 Step，另有warmup、引导/drain和三次在线执行。新预测器与具体评测产物路径在最终报告和 `checkpoints/` 中。

如果已经由代理准备了实际 suite，只需将 `V3_REFINE` 设置成其绝对路径，从 verify/run 继续；不要重复 prepare。目录路径应保存到终端外，避免重连后丢失变量。

## 恢复与结果解释

重复执行 `--run --suite SAME_PATH` 会验证并复用带成功 checkpoint 的阶段。失败的 attempt 和日志保留，重试创建新的 attempt；已完成但产物被修改会报错，不静默忽略或覆盖。OS 文件锁防止同一 suite 并发运行，并在进程退出后释放。

报告字段：

- `fit`：聚合后训练样本数、总桶数与受支持桶数；原建表摘要中的 `independent_evaluation_pending` 属于建表时快照，独立评测完成状态查看当前最终报告与受控评测产物。
- `parent_coverage / controlled_coverage`：原511诊断与新512受控覆盖率。
- `controlled_supported`：受控支持范围内欠预测与误差。
- `online.<mode>.window`：在线测量窗口的覆盖率、欠预测、误差及各phase覆盖率。
- `online.<mode>.drain`：drain单列；未覆盖样本不删除。
- `review_targets_met / needs_attention`：预先冻结的复评目标是否达到及具体缺项。
- `predictor`：新预测表的绝对路径。

本轮复评目标在新数据运行前固定：支持覆盖率>=95%、支持范围内欠预测率<=10%、最大低估<=5ms、每类评测至少30个测量 Step，在线请求全部完成。这些是工程复评目标，尚未与指定 ITL SLO 的安全余量完成联合标定；不能据此宣称硬性 SLO 保证或完整 V3 验收。

流程完成后 `workflow_completed=true`；目标未达到时正常保留最终报告，`review_targets_met=false` 并列出原因，不伪造通过。执行异常、身份不匹配或损坏产物则以非零退出并保留日志。全部运行数据仍位于已忽略的 `results/raw/`，不上传原始实验结果或权重。
