# V2第1批：固定策略、在线Trace与烟雾测试

本阶段新增入口，不替代`benchmarks/v2_stall`的受控注入结果，不代表V2整版完成。复用Qwen3-4B/BF16、V1统一Mixed Forward和V0开放环Driver。V1完整模型数值门槛仍未通过；本阶段只能支持探索性推进。

## 交付范围

- `nanovllm/engine/fixed_scheduler.py`：新增可选FixedScheduler，保留V1原文件。
- `benchmarks/v2_baselines/workload.py`：Poisson、Burst、长短混合Trace。
- `observed_engine.py`：复用V1执行与观测，安装新固定策略。
- `protocol.py`：独立来源/模型/Trace冻结，旧源码清单作为子集核验。
- `run.py`：单进程在线烟雾运行，性能和逐Step诊断分开。
- `regress.py`：新策略Mixed、Prefix、强制Preemption GPU状态场景。
- `metrics.py`、`audit.py`：按原始记录重算，检查时间、输出长度和统计范围。
- `suite.py`、`scripts/test_v2_baselines.sh`：准备、回归、烟雾与审计入口。
- `tests/test_v2_baselines.py`：用户确认的策略、Trace及运行/审计三个公开边界。

不实现容量搜索、最佳配置选择、SLO自适应、量化或Graph/TP扩展；这些另建阶段。

## 策略定义

| 标识 | 行为 |
|---|---|
| baseline | 冻结原始ObservedEngine，不改选择顺序 |
| chunk-256/512/1024/2048 | V1 Decode优先，剩余全局预算按每请求Chunk上限分配 |
| decode_first | 有可执行Decode时不安排Prefill；全部Decode被抢占后允许重算推进 |
| decode_first_wait | 基于纯Decode-first；等待段超过固定阈值时，先尝试安排一个Prefill Chunk，剩余预算用于Decode |

等待计时覆盖初始入队、分块间等待和抢占后的等待，实际Prefill进展后重置。等待保护不是Slack/Aging自适应控制器，也不保证绝对最大等待；KV不足等冲突写入Step日志。强制选择Prefill后，仍按Decode在前的既有BatchMetadata契约排列，避免改变Attention执行接口。max_num_seqs和Token Budget均约束每Step。

## Trace与默认参数

- Warmup到达2s，测量到达8s，有限Drain60s；各窗口停止点均为半开边界。
- 基础平均到达率2req/s，seed41，仅烟雾用途。
- Poisson：全部256Token输入；相邻间隔来自指数分布。
- Burst：同样的输入组成，周期2s，其中0.5s为6req/s、1.5s为2/3req/s，周期平均2req/s。
- 长短混合：Poisson到达，256/1024Token输入，长请求概率20%；实际比例写入Trace，不强制恰好20%。
- 所有请求固定生成32Token，Greedy、ignore_eos=True。
- 全局Token Budget4096，max_model_len4096，每Step最多32个请求，显存比例0.8。
- 等待保护和饥饿统计阈值默认均为1000ms，分别冻结，不根据结果调整。

到达与长度抽样使用独立随机流。Burst按累积强度映射Poisson到达，完整周期平均强度相同；默认Warmup/测量窗口均包含完整周期。同seed的Poisson和Burst请求总数及输入相同，但到达节奏不同。真实Token来自带请求标识的重复语言fixture，不是ShareGPT，不用于自然任务质量评价。

max_num_seqs=32不是“固定32并发”，只约束每个Step的请求数量；在途请求数以Backlog记录为准。本批不承诺256路活跃Decode。

## CPU验证记录

新增13项CPU测试通过；复用V1状态25项、V0测量25项、V2受控注入13项均通过。CLI帮助、Shell语法与Python语法已检查。CPU归档集成测试中的GPU输出是明确标注的硬件边界fixture，不是GPU实验完成证据。

GPU回归与21轮烟雾尚需用户执行。不得将CPU通过写成GPU通过或简历性能数据。

## 执行命令

确认GPU0无其他计算任务，在服务器tmux会话中执行。不要同时运行旧10seed重复命令或其他实验。

```bash
(
set -euo pipefail
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh
export CUDA_VISIBLE_DEVICES=0

bash scripts/test_v2_baselines.sh --cpu

V2_BASE_SUITE="$(mktemp -d "$PWD/results/raw/v2-baselines.XXXXXX")"
echo "实验目录：$V2_BASE_SUITE"
exec > >(tee "${V2_BASE_SUITE}.log") 2>&1

bash scripts/test_v2_baselines.sh --prepare --suite "$V2_BASE_SUITE" \
  --parent-suite "$PWD/results/raw/v2-stall.DFlCEw" \
  --seed 41 --rps 2 --warmup-s 2 --measurement-s 8 \
  --short-tokens 256 --long-tokens 1024 --output-tokens 32 \
  --max-num-seqs 32 --max-model-len 4096 --token-budget 4096

bash scripts/test_v2_baselines.sh --verify --suite "$V2_BASE_SUITE"
bash scripts/test_v2_baselines.sh --regress --suite "$V2_BASE_SUITE"
bash scripts/test_v2_baselines.sh --smoke --suite "$V2_BASE_SUITE"
bash scripts/test_v2_baselines.sh --report --suite "$V2_BASE_SUITE"
bash scripts/test_v2_baselines.sh --verify --suite "$V2_BASE_SUITE"
echo "完成：$V2_BASE_SUITE"
)
```

回归包含已有CPU/Attention测试，以及6种Mixed策略×2个状态场景，共12个独立GPU进程。等待保护回归使用可控调度时钟触发阈值，输出时钟仍为真实时间；整轮仅诊断，不用于性能。

烟雾为3种负载×7种策略，共21个独立GPU进程，运行顺序按seed打乱。它生成18组与原始调度的配对，每种负载共享该负载的原始参照。

## 结果与门禁

- `protocol.json`、`traces/*.json`：冻结设置和请求。
- `regression/report.json`：状态/局部Attention与完整模型数值验收分开。
- `regression/<policy>-<case>/`：新策略GPU状态证据。
- `runs/<mode>/<policy>/`：environment/run_config/summary/requests/steps/backlog与产物哈希。
- `report.*/report.md`：单seed结果导航，不能作为正式加速结论。
- `report.*/pairs.json`：原始与候选的所有请求、首次Token分歧和共同历史。
- `report.*/summary.json`：是否允许进入探索性容量扫描。

每个运行先做代表性16Token预热，再重置冷Prefix元数据，正式运行不插入逐Step状态扫描。记录测量结束时积压以及完整Drain日志；未完成、拒绝、超时均保留。性能Trace的最后状态检查在计时之外。

默认TTFT/ITL/TPOT汇总针对测量窗口到达的cohort；窗口吞吐包含所有窗口输出Token，Goodput只含窗口内完成且满足双SLO的请求。`all_requests_tpot_*`另统计所有完成请求的全生成TPOT，包括Warmup到达；未完成请求仍在SLO分母，不能把完成请求TPOT当成全部请求的无偏估计。等待/重算/抢占统计包括Drain。KV水位为postprocess采样，显存峰值为PyTorch allocator且包括Drain，不等于NVML总显存。

若一轮失败，原始数据和错误保留；不覆盖、不自动重试。不完整回归需要保留失败目录并检查原因，不伪造通过产物。运行完整但没有满足门禁时，smoke/report返回2，并仍生成报告。

进入下一批需：状态/Attention通过，21轮完整，18组配对均有效，实际KV Block数/字节及环境一致，所有Token分歧已审阅。当前自动入口仅在没有分歧时标记`ready_for_exploratory_capacity_scan=true`；有分歧则保留并停止，需要进一步诊断。它不改变V1数值失败，也不授权正式验收Tag。

## 冻结与历史

新增代码会使旧suite的“精确源码集合”校验拒绝当前代码，即使旧文件没有修改。旧数据与原来源仍保留；新prepare检查旧`protocol.json`清单中的每一个文件未变，并记录新增文件。旧suite的原始结果不是失效或被重写，但旧运行命令不能直接在扩展源码后继续；后续实验必须使用新suite。

参数只在prepare前设置；运行期间不改源码、不改变门槛。第一批通过后再根据实际结果实现容量粗扫，不预先假定最佳Chunk或饱和RPS。
