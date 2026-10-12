# V2有界预热、可恢复批处理与独立协议

本入口位于`diagnostics/`，刻意不修改旧`benchmarks/`、`scripts/`、`tests/`或引擎源码，旧实验源码冻结仍可验证。它不是量化、调度算法修复或V1数值门槛放宽。

## 新协议

- 默认54轮：6/10req/s × seed71/72/73 × Poisson/Burst/长短混合 × baseline/Chunk256/Chunk512。
- TP1、Eager、BF16、full观测；短输入256、长输入1024、输出32Token；在线预热窗口4秒、测量60秒、有限Drain60秒。
- 独立的模型前预热固定8请求：4短＋4长，固定seed12345、同时到达、各输出16Token，截止120秒；数量不随测量Trace改变。每个策略和到达率用同一冻结预热输入。
- 预热只覆盖代表性Prefill、Decode及适用策略的Mixed路径，不声称覆盖所有batch/Graph shape。正式运行前验证完成和状态、重建冷Prefix/页元数据、清空队列相关状态、观测记录、计数器；不重建GPU KV张量。
- `warmup/`保留Trace、requests、steps、backlog和包含原始driver_error的report；失败不进入测量。
- 新协议冻结原源码集合、独立工具文件（包含测试/本文档）、模型元数据、全部Trace、策略顺序、预热规则。修改这些内容后必须新建suite，不能改旧协议继续跑。
- 旧200轮只是探索参考，不导入本协议，不将新旧预热的数据直接合并统计。
- 使用不同seed是相对于旧探索Trace独立，不意味着这里存在自动的“最佳策略”选择；默认同时报告256/512。纯短Prompt长度256时这些Chunk通常等价，不能凭微小波动排名；最长Prompt1024也无法充分区分1024/2048。

## 使用

在服务器tmux中：

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh
export CUDA_VISIBLE_DEVICES=0
bash diagnostics/v2_bounded/test.sh --cpu

V2_BOUNDED_SUITE="$(mktemp -d "$PWD/results/raw/v2-bounded.XXXXXX")"
echo "保存这个路径：$V2_BOUNDED_SUITE"
bash diagnostics/v2_bounded/test.sh --prepare --suite "$V2_BOUNDED_SUITE"
bash diagnostics/v2_bounded/test.sh --verify --suite "$V2_BOUNDED_SUITE"

# 3轮同协议子集：6req/s、seed71、Poisson下的三策略，各测量60秒。
bash diagnostics/v2_bounded/test.sh --smoke --suite "$V2_BOUNDED_SUITE"

# 补其余51轮；已通过审计的3轮直接复用。GPU0上不要同时运行其他任务。
bash diagnostics/v2_bounded/test.sh --batch --suite "$V2_BOUNDED_SUITE"
bash diagnostics/v2_bounded/test.sh --report --suite "$V2_BOUNDED_SUITE"
bash diagnostics/v2_bounded/test.sh --verify --suite "$V2_BOUNDED_SUITE"
```

断开终端后，在tmux内重设打印出的suite绝对路径，重复`--batch`即为续跑；不要重新`mktemp`或`--prepare`。`--max-new N`可限制本次新启动轮数，仍按冻结顺序执行。

若希望新的独立矩阵覆盖Decode-first等全部策略，可在**首次prepare**增加：

```bash
--policies baseline chunk-256 chunk-512 chunk-1024 chunk-2048 decode_first decode_first_wait
```

此时默认6/10req/s共126轮；不要在已有suite中修改策略。默认54轮是按前期探索缩小范围的独立验证，不声称补完旧261轮。

## 恢复与失败规则

- 每个job在`runs/<job-id>/attempt-0001/`独立封存，日志为同级`.log`。
- `VALID_COMPLETE`：测量有效且有限Drain内全部完成。
- `VALID_INCOMPLETE`：测量有效但有未完成请求；作为容量观测保留并继续，不能仅据此判定稳态或确定过载原因。
- `SETUP_FAILED/WARMUP_FAILED/MEASUREMENT_FAILED/CLEANUP_FAILED/INTERRUPTED/PARTIAL`：保留失败/中断，不当作性能数据。
- 已有成功、失败、中断尝试均不会被自动覆盖或重试；未执行的其他job可以继续。新失败不会让整个矩阵必然停止，最终报告仍标记不完整；协议/哈希异常立即中止。批次之间用文件锁防止同suite并发。
- 如需显式重试，先定位原因，再执行`--retry --suite PATH --job JOB_ID --reason "原因"`。它创建新attempt；不能重试有效测量来挑好结果。报告保留所有attempt，多attempt的job不自动择优纳入汇总，需单独审查或新协议复验。
- Ctrl+C会保留现场；如果子进程被强杀而未封存，该attempt标记PARTIAL。再次batch不会覆盖它。

## 审计与报告

`--verify`验证协议和已有运行，不要求矩阵全部执行；`--report`总是输出新`report.*/`，数据缺失返回2，绝不把部分完成当验收通过。

- `coverage.json`：全部job/attempt，包括缺失和失败。
- `runs.json`：唯一有效attempt的逐轮原始指标审计结果，含窗口积压和末30秒等间隔采样的积压斜率。
- `aggregate.json/report.md`：各策略均值±样本标准差、样本数；P99是各轮P99的均值，不是合并分位数。
- `pairs.json`：同Trace的完整对照，环境/实际KV容量匹配检查、配对变化量、每个请求的生成比较与共同历史。
- 所有时延/SLO统计继承原口径：默认针对测量窗口到达cohort；输出吞吐统计窗口可见Token，包含在线warmup期间到达请求的窗口输出；goodput统计窗口内完成且满足TTFT与请求最大ITL阈值的请求。等待/抢占/重算及allocator峰值包括Drain。KV水位为Step后采样，不是瞬时分配峰值。
- 未完成请求不从SLO分母丢弃；删失输出比较与完整输出分歧分别标注，不把长度不同自动归因于数值错误，也不丢弃共同前缀中已有的分歧证据。
- `measurement_matrix_complete=true`只说明指定矩阵及对照数据齐全有效，不代表V1数值正确性通过；数值验收始终PENDING。Drain全部完成不代表在线稳定，报告不自动宣布容量上限。

## 验证边界

CPU测试使用真实调度器、BlockManager、Driver、指标和审计；只替换GPU/进程/时间外部边界，常数Token设备不是Qwen3数值oracle。实际GPU预热、BF16前向、显存与性能均需用户执行上面的命令验证。
