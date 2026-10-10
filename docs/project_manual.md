# 大模型在线推理调度与量化KV Cache优化项目手册

> 仓库目录：SLO-Aware-LLM-Serving-Engine（项目简称SLOPrefill）。修订日期：2026-10-10。
>
> 实现主线：混合Prefill/Decode执行 → Decode-Slack自适应调度 → 量化Paged KV Cache与融合反量化GQA Decode。
>
> 本文按V0–V6共7个正式版本组织功能与验收，保留V0、V1已有实现和历史记录，后续推进V2–V6共5个版本。版本合并不删减核心功能、对照实验、质量门槛和原始证据；已实现不等于完整验收通过或性能数字已取得。每阶段保存代码版本、配置、原始数据与阶段报告。

## 0. 项目目标与范围

### 0.1 三条项目主线

| 主线 | 核心问题 | 应展示的证据 |
|---|---|---|
| 混合执行 | 长Prefill阻塞在线Decode | 计算进度和KV状态正确、输出对齐、Decode Stall变化 |
| SLO调度 | 固定Chunk不适应动态负载 | 时延预测、Slack预算、双SLO Goodput、吞吐代价 |
| 量化KV与Decode | 长Context、高并发下KV占显存与访存 | 真实8-bit存储、Scale管理、融合读取、质量与容量收益 |

量化会改变可用Block数、执行时延和抢占频率。最终必须重标定调度预测器，再验证三条主线的组合收益，不能把独立加速比相乘。

### 0.2 底座与技术归属

以[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)为底座。开工时固定完整commit SHA并建立：

- `BASELINE_COMMIT.txt`；
- `CHANGELOG_FROM_UPSTREAM.md`；
- `docs/experiment_protocol.md`；
- `docs/quantization_contract.md`。

保留上游LICENSE。FlashAttention接入、基础Paged KV、Prefix Cache、TP、NCCL和CUDA Graph属于复用能力；自己实际完成的Mixed Forward、调度控制器、量化存储、Scale生命周期和Triton Kernel才写“设计/实现/重构”。

原始版本常见接口为schedule返回整批is_prefill、ModelRunner分阶段prepare。上游会更新，开发依据冻结版本而非假设main永远不变。

本次修订保留Qwen3-4B统一模型基准，并将路线合并为7个正式版本。按当前项目进度，V0、V1已实现；V0测量与回归证据见仓库docs/versions/V0.md，V1实现入口及验收边界见本手册V1.4与仓库docs/versions/V1.md。具体通过矩阵与性能结果以阶段报告和原始数据为准；V2–V6及尚未落地的目录、接口、Tag仍为实现计划，不能将计划或回归通过表述为已取得性能优化收益。

### 0.3 硬件选择

主开发环境为RTX 3090单卡与双卡TP=2；RTX PRO 6000 Blackwell作为扩展环境，先确认具体型号。

| 环境 | 推荐实现 | 必须区分 |
|---|---|---|
| RTX 3090，Ampere | FP8字节存储能力探测；必要时软件编解码或INT8后备；反量化后FP16/BF16计算 | FP8存储、转换指令、Tensor Core计算是不同能力 |
| RTX PRO 6000 Blackwell | 先复现同一存储/反量化路径，再探索原生FP8计算 | 硬件支持不保证特定Kernel/API可用 |

3090没有原生FP8 Tensor Core计算支持。PyTorch能创建float8张量，不代表目标Triton Kernel支持对应转换或dot。V0记录编译、运行、精度与转换时延。

3090若直接转换不支持，可研究软件E4M3FN编码/解码；解码位运算/查表与写入编码需分别验证。INT8数值不能冒充FP8位模式。FP8不可用时先用INT8完成架构，并同步修改结果与简历名称。不能只在Blackwell通过后宣称3090 FP8 Kernel已支持。

### 0.4 完成标准与周期

必需完成：

1. 非量化Mixed Batch及状态测试；
2. 固定策略与在线到达基线；
3. Slack控制器与独立预测误差报告；
4. Aging、水位保护的消融；
5. 一个真实运行的量化格式；
6. 量化写入、Scale管理与Prefix生命周期；
7. Gather+反量化+FA参考执行；
8. 直接读取量化分页KV的融合GQA Decode；
9. 质量评估、Graph Replay和TP=2回归；
10. 调度×量化×Kernel组合实验、原始数据和失败边界。

加入量化Kernel后，原4–6周不作为全部功能的完成承诺。完整项目约8–12周仅作粗略参考，不因版本合并而同比缩短。当前V0、V1已实现，剩余工作按V2基线扫描、V3完整调度、V4量化参考与真实缓存、V5融合Decode、V6集成与最终实验重新估算；结合已有验收证据、GPU可用时间与失败修复调整，不以版本数量推算工期。V4内部检查点顺序推进，V6的质量预算与兼容性设计提前在V4、V5准备。

时间不足时先延期原生FP8计算、静态K校准优化、复杂Split-KV、Mixed Graph Bucket；保持测量、真实存储、融合Decode、数值质量和端到端消融。

### 0.5 统一模型基准：Qwen3-4B

V0–V6的快速验证、正确性回归、调度实验、量化质量与主性能结果统一使用**Qwen/Qwen3-4B、BF16权重**。取消原“先0.6B验收、8B主结果”的要求。其他模型仅作明确标注的扩展，不混入4B结果，也不把旧模型的吞吐、SLO阈值或饱和RPS直接迁移到4B。

- 配置唯一来源：仓库configs/qwen3_4b.json；
- 本服务器模型目录：/home/lq/vllm-1p1d-v029/model/Qwen3-4B；
- 冻结revision：1cfa9a7208912126459214e8b04321603b3df60c；
- 实际结构：36层、32 Q Head、8 KV Head、head_dim=128、hidden_size=2560，GQA ratio=4；TP=2每rank为16 Q Head/4 KV Head。

本模型head_dim必须读取config，不能误用hidden_size/num_heads=80。BF16 KV payload每Token跨所有层为2×36×8×128×2=147456 bytes（144KiB）；TP=2每rank为73728 bytes（72KiB），总逻辑KV字节不因TP翻倍而减半。上述数值不含Padding、Scale、Block Table、Workspace与Graph。

模型config的max_position_embeddings=40960不表示默认引擎或单卡容量允许同样长度。V0正式短/中上下文基线max-model-len=4096；后续8K/16K须显式扩大引擎长度、验证显存并重测容量，不能依赖默认4K配置。

在服务器项目根目录执行source scripts/env_qwen3_4b.sh后，可直接运行V0入口；workload/run/sweep/test_v0.sh默认读取4B配置，仍允许显式覆盖模型路径与真实revision。此脚本仅设置变量，不安装环境、不加载模型。

本机TP=2须使用NCCL_IB_DISABLE=1绕过首次AllReduce的IB/RoCE段错误，保留NCCL_DEBUG=INFO日志。已实测GPU拓扑NV4、通道P2P/CUMEM；禁用IB不等于禁用P2P/NVLink，其他机器需独立验证。

## 1. 理论与测量契约

### 1.1 混合执行语义

统一Mixed Forward：

```text
扁平Token批次
  → 共享QKV投影
  → 每层Attention按Prefill/Decode分支执行
  → 恢复Token顺序
  → 共享输出投影/MLP
  → 有效位置采样
```

同一Scheduler Step中先完整运行Decode模型、再完整运行Prefill模型，属于分相执行基线，不能等同统一前向。V1先实现统一前向、分Attention调用的正确版本。

“Decode优先”表示选择顺序，不代表在统一前向中能够提前向用户返回Token；整体前向完成后才能采样。

### 1.2 时间与双SLO

使用perf_counter_ns，记录arrival、enqueue、first_scheduled、每个token_time、finish。

```text
arrival：Trace规定的到达时间
enqueue：实际进入Engine等待队列的时间
enqueue_lag = enqueue - arrival
TTFT = first_token - arrival
ITL_j = token_time_j - token_time_(j-1)
TPOT = (last_token - first_token) / (output_tokens - 1)
```

单Token请求的ITL/TPOT不适用，单独报告，不填伪造的0。单线程Driver在Step之间提交，应把enqueue lag计入TTFT。Token时钟是CPU侧Engine可见时间，不含网络与文本解码。

双SLO：

```text
TTFT_i <= TTFT_SLO_i
请求内P99(ITL_i) <= ITL_SLO_i
```

全局池化Token的P99 ITL与请求内P99分别记录。同时报告最大ITL和超阈值Token比例，避免P99掩盖极端暂停。

### 1.3 Goodput、窗口和Drain

实验顺序：初始化 → 在线Warmup → 测量窗口 → 停止到达 → 有限Drain。

```text
Window Goodput =
    窗口内完成且满足双SLO的请求数 / 窗口秒数
Window Throughput =
    窗口内输出Token数 / 窗口秒数
Cohort Success Rate =
    窗口内到达且最终合格请求数 / 窗口内到达请求数
Cohort Violation Rate = 1 - Cohort Success Rate
```

Goodput是速率，Success Rate是比例。未完成、拒绝、超时均计入Cohort违约；另外报告完成请求内违约率。Window Goodput可包含Warmup到达、窗口内完成的请求。Drain完成数不能全部除以注入时长冒充稳态吞吐。

### 1.4 饱和负载

区分容量饱和点与满足预定违约率的SLO容量点。粗扫后细扫，每配置至少3次，检查完整Backlog曲线、完成率平台和持续积压。

策略比较使用共同绝对req/s，以及相对同一个BF16固定策略基线容量的：

```text
0.3 / 0.6 / 0.8 / 1.0 / 1.2 × baseline_saturation_load
```

不能让各策略按自身容量归一化后直接比较。策略自身容量另报。

### 1.5 Slack预算控制

候选Prefill预算P：

```text
slack_i(P) =
    ITL_SLO_i
    - (decision_time - last_token_time_i)
    - predicted_step_time(P)
    - safety_margin
```

在0/256/512/1024/2048/4096候选中选预测风险可接受的最大可行P，同时满足全局Token Budget、max_num_seqs和KV容量。

预测器先用实测P95分桶查表；这是经验风险估计，不能承诺“不会违约”。记录欠预测率与实际SLO结果。紧急减预算可跨档，增加预算用Hysteresis。P=0仍违约时记录unavoidable_risk；无Decode时允许扩大Prefill，但仍检查TTFT。

预测终点应与Token时钟一致：从预算决策时刻到下一Token在CPU侧可见，而非只有GPU Forward时间。预算决策后的metadata准备、H2D、采样与postprocess均计入预测或单独的保守开销项。比较多个候选P期间消耗的CPU时间也会减少Slack，调度提交前再次检查。

### 1.6 Paged KV、GQA与FA

按层/TP rank推荐：

```text
K/V: [layers, blocks, block_size, local_kv_heads, head_dim]
K/V Scale: [layers, blocks, block_size, local_kv_heads]
Block Table: [batch, max_logical_blocks]
physical_block = block_table[request, token // block_size]
offset = token % block_size
kv_head = q_head // (num_q_heads / num_kv_heads)
```

GQA不物理复制K/V。TP下核验本地Head映射与整除。独立Kernel可测试小Block；引擎Block配置必须符合冻结上游约束，不能直接给只支持256倍数的上游设置16。

PagedAttention描述缓存分页，FlashAttention描述分块计算与IO优化。分页KV可以使用Flash-style算法。本项目原创为量化分页读写与Decode融合，Prefill复用FA。

## 2. 版本与目录

### 2.1 路线总览

共7个正式版本，V0、V1沿用已有编号和记录；后续只新增V2–V6。表中Tag为目标名称，仅在对应阶段验收通过后创建。

| 版本 | 目标 | Tag |
|---|---|---|
| V0 | 环境、在线测量、原始基线、格式能力探测 | v0-measurement |
| V1 | 非量化Mixed Batch与状态正确性 | v1-mixed-batch |
| V2 | 固定策略、工作负载与容量基线 | v2-baselines |
| V3 | Step预测、Slack预算、Hysteresis、Aging与KV Guard | v3-slo-scheduler |
| V4 | 高精度参考、量化契约、真实量化Cache与Gather+FA接入 | v4-quant-cache |
| V5 | 融合反量化Paged GQA Decode | v5-fused-decode |
| V6 | Graph/TP集成、预测器重标定、质量与组合实验、最终交付 | v6-final |

默认按V2 → V3 → V4 → V5 → V6推进。V2保留为独立基线阶段，V5保留为独立Kernel阶段。V4内部依次通过高精度参考、量化契约、真实Cache三个检查点；检查点不是新增正式版本。V4的后端接口与参考实现可基于V1/V2执行路径准备，但正式调度对照先在BF16路径下完成。兼容性设计和初步质量检查随V4/V5开展，V6集中完成完整回归与最终实验。

以下“旧版”专指本次合并前的V0–V10路线，不指更早修订中的同名编号。

| 合并前版本 | 本手册版本 | 迁移范围 |
|---|---|---|
| 旧版V0 | V0 | 保留已有实现、报告、命令与Tag |
| 旧版V1 | V1 | 保留已有实现、报告、命令与Tag |
| 旧版V2 | V2 | 固定策略、负载与容量基线 |
| 旧版V3＋旧版V4 | V3 | Slack与预测器、Aging、KV Guard及独立消融 |
| 旧版V5＋旧版V6＋旧版V7 | V4 | BF16参考、数值契约、真实量化Cache、Prefix与Gather路径 |
| 旧版V8 | V5 | 融合Decode及三组对照 |
| 旧版V9＋旧版V10 | V6 | Graph、TP、重标定、质量、组合实验与交付 |

已有V0/V1代码、阶段报告、测试入口、原始数据及Tag保持原样，缺失证据另补，不因重编号重跑全部历史实验。如已存在旧版V2及后续报告或Tag，保留原件并标注旧路线身份，建立到新版本的映射；不得覆盖、移动已有Tag，或用改名代替新版验收。未来报告按新路线记录，引用历史产物时同时标明旧版本、commit、配置与数据路径。

### 2.2 分阶段交付清单

| 版本 | 主要程序/改动 | 必须保存的阶段产物 |
|---|---|---|
| V0 | benchmarks/v0与格式能力探测 | 环境、到达Trace、请求/Step日志、量化支持报告 |
| V1 | Scheduler、Sequence、BatchMetadata、ModelRunner | Mixed Forward设计、状态不变量与非量化输出对齐报告 |
| V2 | 固定策略、workloads、容量搜索 | 最佳固定参数、容量曲线、真实/合成负载定义、正式Decode Stall对照 |
| V3 | profile_step_time、Slack控制器、Aging与KV Guard | 预测器版本与独立误差、Budget轨迹、五组调度消融、等待段与约束冲突报告 |
| V4 | backend、BF16参考、quantization、quant_store、gather_dequant、Cache分配 | FP32真值与分页矩阵、格式/Scale契约、编码测试、真实显存、Prefix生命周期、初步质量及冻结质量预算 |
| V5 | quantized_decode与可选Merge | 相同量化参考对齐、Gather/Fused/BF16对照、Kernel微基准与失败shape |
| V6 | Graph、TP metadata、重标定、evaluation与组合实验 | Replay/TP回归、拓扑与通信、量化时延profile、独立质量、全链路消融、失败边界及简历实测数字 |

未来每个阶段新增docs/versions/Vn.md，填写输入版本、修改文件、运行命令、配置、原始数据位置和退出条件。已存在的V0/V1报告保留；V3和V4的内部检查点记录在各自阶段报告中，不要求新增正式Tag。报告与数据须标明七版路线身份；旧路线同名报告按2.1所述保留与映射，避免重名覆盖。阶段退出表示实现和证据齐全；是否可以宣称性能优化，还需实际收益达到预先约定标准。

### 2.3 待实现目录

```text
SLOPrefill/
├── nanovllm/
│   ├── config.py
│   ├── engine/
│   │   ├── sequence.py
│   │   ├── scheduler.py
│   │   ├── block_manager.py
│   │   ├── model_runner.py
│   │   ├── llm_engine.py
│   │   ├── batch_metadata.py
│   │   ├── metrics.py
│   │   └── policies/{fixed_chunk,slo_aware}.py
│   ├── layers/attention.py
│   ├── attention/
│   │   ├── backend.py
│   │   ├── reference.py
│   │   ├── paged_gqa_decode.py
│   │   └── quantized_decode.py
│   └── kv_cache/
│       ├── layout.py
│       ├── quantization.py
│       ├── quant_store.py
│       └── gather_dequant.py
├── benchmarks/
│   ├── v0/
│   ├── online_benchmark.py
│   ├── saturation_search.py
│   ├── profile_step_time.py
│   ├── bench_kv_store.py
│   ├── bench_decode.py
│   └── workloads/
├── evaluation/{teacher_forced_logits,perplexity,long_context}.py
├── tests/
├── configs/
├── results/{raw,processed,figures}/
├── profiling/{nsys,ncu,topology}/
├── docs/{quantization_contract,experiment_protocol,failure_boundaries}.md
├── BASELINE_COMMIT.txt
├── CHANGELOG_FROM_UPSTREAM.md
└── README.md
```

这些是规划目录，不表示程序已存在；按版本新增。代码重构、正确性修复、性能优化和统计口径变更分别提交。

## V0：环境、原始基线与在线测量

### V0.1 理论

预生成开放环到达Trace，Engine忙时保留原到达时间。通过观测原schedule/run/postprocess建立Token时钟，不改变Scheduler选择。

### V0.2 实现任务

沿用此前对话给出的workload.py、metrics.py、engine.py、environment.py、run.py、sweep.py结构，新增probe_quant_support.py。此前代码仅在回复提供，是否落地以实际仓库为准。

1. 冻结上游SHA、模型revision及CUDA/PyTorch/Triton/FA/NCCL。
2. 请求指标放外部Record，CPU时间戳不必传入TP worker。
3. 中间Prefill Chunk不记录Token输出；仅postprocess追加Token时产生事件。
4. 包裹preempt，统计重算和等待段；重置computed不等于输出回退。
5. Trace固定到达、单线程Step间提交、记录enqueue lag。
6. 原Sampler无Greedy分支时，参考与观测引擎都安装argmax采样器；original采样性能组独立。
7. 测量结束后写日志，避免每Step文件IO。
8. 窗口、Warmup、Drain、unfinished和拒绝全部归档。
9. 分别探测8-bit存储、FP8转换、软件编码、Triton编译、Graph和转换时延。

环境命令：

```bash
git rev-parse HEAD
python -m torch.utils.collect_env
nvidia-smi
nvidia-smi topo -m
nvidia-smi nvlink --status
```

启动TP前设置NCCL_DEBUG=INFO并保留各rank日志。P2P可访问性与NCCL实际链路分别报告；3090支持NVLink不代表机器装有Bridge。

### V0.3 验证与产物

CPU：分位数、未完成、Drain、单Token；GPU：在Qwen3-4B上验证离线全部到达与在线零间隔Greedy一致。先4B单卡Eager，再单卡Graph与TP=2。对照无观测/最小观测/完整观测，量化观测开销。4B现有正确性产物可继续使用，不需补0.6B或重写历史实验；新默认配置与NCCL环境记录变更后另存新轮次。

产物：environment.json、trace.jsonl、requests.jsonl、steps.csv、backlog.csv、summary.csv、容量扫描、quant_support.json。

退出条件：时间单调、无请求丢失、原调度/输出一致、三次可复现、量化格式路线及设备限制已确定。此阶段尚未启用量化缓存。

## V1：混合Prefill/Decode执行与状态正确性

### V1.1 理论

先高精度KV统一前向，通过每层Attention分支保留不同后端。调度与执行语义分开测试。

### V1.2 实现任务

修改Scheduler、Sequence、ModelRunner、LLMEngine与Attention Context，增加BatchMetadata。用逐请求phase替代整批is_prefill。

```python
@dataclass
class ScheduledRequest:
    seq_id: int
    phase: str             # prefill/decode/recompute
    start_pos: int
    num_scheduled_tokens: int

@dataclass
class SchedulerOutput:
    requests: list[ScheduledRequest]
    num_prefill_tokens: int
    num_decode_tokens: int
    token_budget: int
```

执行顺序：

1. Decode优先选择；未选请求仍在队列，记录max_num_seqs限制。
2. 剩余预算安排Prefill，记录每请求Chunk与全Step Budget。
3. 构造扁平input_ids、positions、query offsets、context_lens、Slot Mapping和Block Table。
4. 同层共享QKV投影，按phase调用Attention，再scatter回Token顺序。
5. sample_indices仅包含最后Prefill Chunk和Decode有效行。
6. postprocess逐请求提交，不对中间Chunk采样。
7. Mixed用eager，纯Decode先保留原Graph。
8. 新metadata同步到所有TP rank，核验序列化格式与缓冲区上限。

状态不变量：

```text
computed + scheduled <= logical_tokens
每请求每Step最多产生一个输出Token
computed只在实际KV写入完成后提交
preempt可重置computed，历史high_water另存
recompute不得重复追加逻辑输出Token
Block存在、KV有效、Prefix Hash发布是不同状态
```

### V1.3 验证与退出

Batch=1/2/8/32，Chunk=256/512/1024/2048，跨页、Prefix命中、长Prompt插入、抢占、TP=1/2、Graph开/关。

非量化Greedy在约定集对齐；同时比较Teacher-forced logits。不同shape可能有浮点差异，Token分歧需检查argmax margin和状态，不能假定所有输入bitwise一致。

产物：Mixed设计、状态测试、通过数量、Logits误差、Decode Stall原始数据。退出：无重复采样、跨请求泄漏、Block引用泄漏，统一前向语义明确。

### V1.4 实现入口（2026-10-10）

新增独立Mixed执行路径，保留V0引擎。逐文件实现、基础/GPU/在线测试命令、数值口径和验收边界见`versions/V1.md`。代码已存在不等于完整矩阵或正式性能实验已通过。语言fixture与随机Token压力测试分开，已发现的失败记录保留，不能将单轮烟雾结果作为简历性能收益。

```bash
bash scripts/test_v1.sh --cpu
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1.sh --gpu --tp 1
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_online.sh
```

## V2：固定策略与工作负载基线

### V2.1 实现任务

保留原版单独基线；其他策略共用V1执行路径：

- 固定Chunk=256/512/1024/2048；
- 纯Decode-first；
- Decode-first+最大Prefill等待。

Chunk Size是每请求上限，总Prefill Budget是Step总量，两者不同。额外保留分相执行基线时准确标注。

| Workload | 要求 |
|---|---|
| Poisson | 固定req/s、指数间隔、预生成Trace |
| Burst | 同平均到达率对照，保存峰值与恢复时间 |
| 长短混合 | 256–1K与8K–16K，遵守模型Context上限 |
| Prefix-sharing | 真正相同Token前缀，冷/热分组 |
| Output-heavy | 短输入、长输出 |
| Prefill-heavy | 长输入、短输出 |
| ShareGPT | 区分真实文本回放与仅长度回放 |

性能实验固定输出长度；任务质量使用自然停止。合成随机Token可研究调度，不能冒充自然文本质量。

### V2.2 验证与退出

扫req/s和共同基线容量，记录TTFT/ITL/Goodput/吞吐、最大等待、重算、KV水位。固定策略在调参Trace选择，在独立seed/分布评测；不能用弱固定策略对照。

过长请求明确拒绝计数，不静默截断，不允许无法容纳请求造成死循环。

退出：证明Prefill造成ITL尖峰及固定最优Chunk是否随负载改变，保存容量曲线。如果固定策略稳定最优，保留负结果并简化控制器。

## V3：完整SLO感知调度与保护机制

依赖V2可信固定策略与共同容量基线，本版只使用BF16 KV。内部按Step预测 → Slack预算 → Hysteresis → Aging → KV Guard推进，各机制保留独立开关。每个检查点先通过正确性与局部验证，再开启下一机制；合并正式版本不合并实验变量。

### V3.1 Step预测与Slack控制器

先只用高精度KV。新增profile_step_time.py与替换式控制器。

预测特征：

```text
decode_count, prefill_tokens, prefill_request_count
sum_context_tokens, max_context_len
tp_size, graph_mode, kv_dtype, attention_backend
```

先实测P95分桶查表，超范围保守回退；独立Trace验证欠预测率、P95误差、CPU预测成本。选择候选P时使用对应时延，不用固定预测值。

每Step：构造可行候选 → 估时 → 检查每请求Slack → 选最大预算 → 紧急下降/缓慢恢复 → 记录预测与实际。

### V3.2 Slack-only验证与内部检查点

Slack-only vs 最佳固定Chunk：双SLO、Goodput、P99 ITL、TTFT、吞吐代价和budget-switch count。允许吞吐代价在实验前固定，例如5%，它是选择标准而非既成结果。

Slack内部检查点：correctness通过、结果可重复、收益或失败可解释。若只改善ITL却恶化Goodput，不称为整体优化。

### V3.3 Aging与KV水位保护

将公平性和水位作为约束，不与Slack任意加权。

- Aging warning逐步预留预算；hard阈值尝试至少一个Chunk。
- 记录初始Prefill和抢占后等待段。
- KV低/高水位Hysteresis，限制新Admission，保护运行中Decode。
- 首版复用抢占选择，受害者优化另做消融。
- KV不足或Decode已违约时记录约束冲突，不能靠参数宣称保证最大等待。

### V3.4 独立消融与整版退出

| 组 | 策略 | Slack | Aging | KV Guard |
|---|---|---|---|---|
| F | 最佳固定Chunk | 关闭 | 关闭 | 关闭 |
| S | Slack-only | 开启 | 关闭 | 关闭 |
| SA | Slack＋Aging | 开启 | 开启 | 关闭 |
| SK | Slack＋KV Guard | 开启 | 关闭 | 开启 |
| SAK | 完整策略 | 开启 | 开启 | 开启 |

S/F评估自适应预算；SA/S与SK/S分别评估Aging与KV Guard；SAK/SA、SAK/SK检查组合条件下的增量与相互影响。各组使用相同执行后端、Trace、绝对到达率和共同基线容量，公平性及水位信号分别关闭时不应残留隐藏行为。

预先定义starvation，例如等待超过指定阈值仍无Prefill进展；unfinished不得排除。报告Burst恢复、最大等待、ITL峰值、TTFT违约、Admission等待、preemption/recompute。

产物：冻结配置、预测器及独立误差报告、Budget轨迹、五组消融表与BF16最终策略。整版退出：状态正确，预先固定吞吐代价上限与调度验收门槛，收益或失败可重复解释；信号没有独立收益则从默认策略删除，保留失败实验，并同步调整简历中关于独立收益的表述。

## V4：量化KV参考、真实缓存与Gather执行闭环

本版合并参考后端、量化数值契约与真实Cache接入。目标是得到能够运行模型、实际节省KV存储、可供融合Kernel对照的量化执行路径；Gather路径暂时变慢可以接受，但必须记录代价。

### V4.1 实现顺序与内部检查点

| 检查点 | 实现内容 | 通过条件 |
|---|---|---|
| V4-A | 后端接口、FP32真值、BF16 Paged GQA参考 | 分页寻址、Head映射、Mask与非量化输出正确 |
| V4-B | FP8/INT8数值参考、设备编解码、Scale契约 | 格式与舍入测试通过，误差与转换成本可测 |
| V4-C | 真实8-bit Cache、Triton写入、Gather+FA、Prefix生命周期 | 模型可运行，实际存储节省可测，初步质量与状态正确 |

V4-A → V4-B → V4-C顺序推进，失败时在相应检查点定位。BF16 Kernel承担正确性参考和对照职责，不要求先做完整性能调优或在所有shape上超过原后端。首版交付一个实际可用的8-bit格式即可；FP8为主路线，INT8为设备能力不足时的后备，两者数据、命名与结果严格区分。提前按Graph固定地址与TP本地Head约束设计KV/Scale池及metadata，并安排小规模兼容性回归；完整Graph/TP验收仍在V6。

### V4.2 V4-A 高精度分页GQA参考

#### V4.2.1 理论

先排除分页与GQA错误，再改变存储精度。FP32 Online Softmax维护m、l、acc：

```text
m_new = max(m_old, max(scores))
alpha = exp(m_old - m_new)
p = exp(scores - m_new)
l_new = alpha*l_old + sum(p)
acc_new = alpha*acc_old + p@V
output = acc/l
```

空分区和全mask不能产生exp(-inf - -inf)的NaN。

#### V4.2.2 实现任务

1. 后端接口：store_kv、prefill、decode、workspace需求。
2. PyTorch Gather+FP32 Attention真值。
3. Triton FP16/BF16分页Decode，GQA直接Head映射。
4. Mask覆盖context、页有效性、head_dim尾部。
5. ragged batch、随机物理页，先一Query Head一程序。
6. 接入V1每层分支，保持Token输出顺序。
7. 自写BF16与原后端独立计时，不要求所有shape获胜。

#### V4.2.3 验证与退出

Batch=1/4/16/64；Context=1/P-1/P/P+1/511/1024/4096/8192，P是Block Size；D=64/128和模型实际值；GQA ratio=1/2/4/8；随机页、尾页、非连续数据。

容差以FP32参考校准，报告最大绝对误差、归一化误差、cosine；近零不能只看relative error。引擎实际Block与独立小Block测试分组。

退出：分页和GQA正确性通过，形成量化可信参考与shape延迟表。

### V4.3 V4-B 量化数值参考与Scale契约

#### V4.3.1 数值定义

主路线为FP8 E4M3FN，最大有限值448；明确区别E4M3FNUZ。首版K/V各自per-token、per-KV-head动态Scale，沿head_dim取amax：

```text
amax = max(abs(x))
amax=0时scale=1
否则scale=max(amax/qmax, scale_floor)
q = quantize(clamp(x/scale, -qmax, qmax))
x_hat = dequantize(q)*scale
```

FP8 qmax=448；INT8对称qmax=127、零点0、round-to-nearest。首版FP32计算/存储Scale，scale_floor固定配置；非有限输入明确处理。

K在RoPE之后量化，V按模型实际写入值；先检查模型执行顺序。权重、Q和MLP保持原精度，避免扩大变量。

#### V4.3.2 实现任务

- PyTorch/CPU量化参考和实际GPU转换；
- 3090格式能力探测结果对应的软件或原生转换实现；
- INT8后备，数据与命名独立；
- 零值、极小Scale、outlier、极值、非有限、舍入边界；
- 真实模型各层K/V分布；
- 格式bit-level与往返误差。

Scale契约还须规定rounding、clamp、scale_floor、存储dtype、Tensor stride与Padding。uint8容器只保存FP8编码位模式，读取时必须做对应浮点解码；直接把uint8转为BF16得到的是整数数值，不是FP8解码。INT8后备使用独立dtype和解码函数。

每Token/Head Scale写入后不变。动态per-block Scale追加时可能要求重编码旧Token，首版不使用。

静态K+动态V、per-head或per-layer作为后续对照；静态Scale独立数据校准、分层/head保存，变化后旧Cache全部失效。

#### V4.3.3 容量核算

每层每Token/Head：

```text
BF16 K+V = 2*D*2 bytes
8-bit K+V + 两份FP32 Scale = 2*D + 8 bytes
D=128：512 bytes vs 264 bytes，payload+Scale约1.94倍容量
```

还需计入布局Padding、Block Table、Gather工作区、Graph和Allocator余量。KV容量不是整模型显存，也不等于吞吐/并发严格翻倍。

#### V4.3.4 验证与退出

K/V分别统计MSE、最大误差、cosine、clipping比例和Scale分布；Teacher forcing固定相同历史做初步Logits检查。

退出：参考与GPU转换正确、Scale身份明确、FP8实际可用性和成本已记录。Fake Quant不计真实存储成果。

本阶段先建立quantization_contract.md与quality_budget.yaml草案，在独立评测前冻结。任务质量上限由BF16基线、数据集和用途决定，不在手册中预设一个通用“无损”数值。

### V4.4 V4-C 真实量化Cache与Gather接入

#### V4.4.1 实现任务

修改ModelRunner分配、Attention写入、BlockManager元数据。

1. 分配8-bit K/V与独立Scale池，以实际含Scale的block_bytes计算Block数。
2. Triton按Slot完成amax、编码、量化store；slot=-1跳过所有写入。
3. 禁止同批有效Slot冲突，检查随机Slot与页边界。
4. 按Block Table Gather/反量化有效Context到BF16。
5. Prefill接FA变长接口，核验分块Q绝对位置和causal对齐。
6. Decode暂用相同量化Gather参考，为V5融合Decode提供对照。
7. Workspace按层复用、有上限，不能反量化整模型所有层。
8. 关闭量化走原BF16路径，不保留全量BF16 Shadow KV冒充节省。

#### V4.4.2 Prefix与生命周期

KV字节与Scale共同组成Cache对象：

- 计算与量化写入完成后才发布完整Prefix Block Hash。
- Cache namespace包括模型/权重、RoPE、格式、Scale方案及校准版本。
- 静态Scale身份变化时清缓存。
- 共享完整Block只读；共享尾块写入需Copy-on-write或禁止共享可写尾块。
- 拷贝/释放/重算同时处理Scale和有效长度。
- 页复用不泄漏旧Scale；GPU读写未结束不能复用物理页。
- refcount、有效长度、历史计算high_water独立维护。
- Hash不覆盖尚未计算的尾部。

#### V4.4.3 验证与退出

随机页/Slot、padding、重复Slot、部分写入、Prefix热/冷、抢占恢复、Scale复用、Chunked Prefill和长Context。

记录量化写入、Gather、反量化、FA耗时及峰值临时显存。先等逻辑容量，再等字节预算。

Prefill参考需逐Chunk核验有效KV长度和Query绝对位置。当Q长短于K时，使用所选FA版本实际的causal对齐语义，不能仅把causal=True视为正确。先对照显式绝对位置Mask的PyTorch参考，再接入性能路径。

退出：真实存储节省可测，无BF16全量副本，量化参考模型稳定。Gather路径变慢也归档，不提前声称端到端加速。

### V4.5 整版退出与质量准备

三个检查点均通过；真实量化Cache不保留全量BF16 Shadow副本，Scale与Prefix生命周期回归通过，模型Teacher-forced logits及小规模任务质量已检查。记录BF16 KV、8-bit payload、Scale、Padding与临时Workspace，分别核算等逻辑容量和等字节预算。在本版划分Scale校准、策略调参和独立评测集，固定模型revision与模板，建立quality_budget.yaml并在独立评测前冻结，V6不得事后放宽门槛。V4暂不宣称融合或端到端加速，报告写明Gather路径代价和失败边界。

## V5：融合反量化Paged GQA Decode

依赖V4已通过的真实量化Cache、Scale契约与Gather参考，先正确再优化。本版保留为独立Kernel交付阶段，不改变既有量化格式与质量预算；量化质量和Kernel实现误差分别评估。

### V5.1 基础Kernel

```text
Q/context_len
  → Block Table分页寻址
  → K/V量化字节+Scale加载
  → Tile内反量化FP16/BF16
  → QK与有效长度Mask
  → FP32 Online Softmax
  → PV累加
  → 输出
```

不物化完整高精度KV。3090称FP8存储+高精度Attention计算，不能称原生FP8 Tensor Core加速。Token级Scale应应用到对应元素，不能错当全局常量。

### V5.2 优化顺序

1. 一Query Head一程序先正确。
2. Tile、num_warps、向量化加载。
3. 多Query Head共享K/V与Scale，group=1/2/4对照。
4. 记录register、Spill、occupancy；共享不保证更快。
5. 长Context/小Batch再考虑Split-KV。
6. 分区输出m/l/acc，Merge为：
   - m=max(m_part)；
   - w=exp(m_part-m)；
   - l=sum(w*l_part)；
   - acc=sum(w*acc_part)；
   - O=acc/l。
7. 空分区特殊处理，避免NaN。
8. 按shape选择单阶段/Split，保存dispatch规则。

共享加载是程序内复用还是L2复用需要Profiler证明。原生FP8 dot作为Blackwell可选扩展，不作为首版门槛。

### V5.3 三组对照

| 对照 | 目的 |
|---|---|
| 融合Kernel vs 相同量化字节/Scale的FP32参考 | 排除Kernel Bug |
| 融合Kernel vs Gather+反量化+FA | 融合执行收益 |
| 量化融合 vs BF16原后端 | 最终精度、容量与性能代价 |

报告Event计时P50/P95、误差、字节流量、Scale加载、workspace与shape。逻辑字节/时延只能称effective bandwidth；DRAM带宽用Nsight计数器。

Kernel数值门槛先在FP32参考与高精度实现上校准，再对“同一量化字节+同一Scale”冻结容差。至少使用max_abs_error和relative_L2_error：后者为norm(output-reference)/max(norm(reference), eps)。另报cosine及失败shape；不能用较宽的量化质量容差掩盖地址、Mask或Merge错误。

退出：correctness通过，目标长Context对Gather有可测收益；比BF16慢的shape保留并解释，单Kernel加速不等于模型加速。

## V6：系统集成、最终质量与组合实验交付

依赖V3冻结的BF16调度策略、V4真实量化Cache与V5融合Kernel。本版集中完成量化路径的Graph/TP回归、预测器重标定和独立评测；质量预算及数据划分已在V4准备。V1非量化TP/Graph证据按已有报告核对，缺项须补齐；本版补齐量化写入、Scale、融合读取和页复用的回归。先完成兼容性与重标定，再进行正式端到端实验；任何格式、Scale或后端变更都须补对应回归与profile。

### V6.1 Graph任务

先量化纯Decode Graph，Mixed仍可eager。

- 固定KV/Scale池、Graph输入与Workspace地址。
- Replay更新positions、Context、Block Table与Slot；Context缩短不读旧尾部。
- Graph内写KV/Scale，再同流读取，建立正确依赖。
- padding request无效、slot=-1，Kernel不能写padding地址。
- Split/merge Workspace固定且限制有效范围。
- 禁止动态Python .item()、临时分配和过期Context。
- 改KV格式、布局或静态Scale身份后重建Graph。

记录Capture次数、Replay、hit、eager fallback、padding率、Graph显存。比较全eager、纯Decode Graph、Mixed eager，复杂Mixed Bucket可延期。

### V6.2 TP任务

- 本rank KV Heads独立量化，首版不增加全局Scale collective。
- 核验本地Head整除和映射，不偷偷复制KV。
- 各rank请求与Block顺序一致，Scale由本rank生成。
- 新metadata完整传输，检查共享内存容量。
- rank0采样/观测维持一致职责。
- 拓扑、P2P、实际NCCL链路与通信占比保存。

单卡/双卡对照同模型、精度、后端和负载，不能换条件声称TP加速。

### V6.3 重标定

量化、融合、Graph改变时延，重新profile并按kv_dtype/backend/TP/Graph区分模型。Prefill预测包含Gather成本；不能只更新Decode。无匹配profile时保守回退。

水位同时报告：

```text
used_blocks/total_blocks
active_kv_tokens/kv_token_capacity
payload_bytes+scale_bytes
workspace_peak_bytes
```

核验冻结BlockManager是在Admission时预分配完整Prompt，还是按Chunk增量分配。Chunked Prefill只保证计算分块，不自动减少已预留KV容量。首版保留原分配策略并准确报告；若新增增量分配，应独立提交、补齐抢占/Prefix/页复用测试，并在消融中单列。

### V6.4 集成验证与内部检查点

连续Replay改变Context、页映射、Prefix、抢占与页复用，eager/Graph对齐；TP=2长时压力。

集成检查点：无过期Scale/Context，无跨rank状态错误，纯Decode Graph正确，量化预测器单独归档。解释Graph/TP收益或退化。

### V6.5 三层质量评估

1. Kernel vs 相同量化参考：寻址、Mask、Softmax正确。
2. BF16 vs 量化Teacher-forced logits：固定同一Token历史，MAE/最大误差/cosine/KL/top1变化。
3. 自回归真实任务：PPL、长Context检索及至少一个实际任务；固定文本、模板、revision与自然停止。

量化不要求100% BF16 Token一致。闭环生成早期分歧后，后续Logits历史不同，不能用它独立归因量化误差。

沿用V4划分的Scale校准、策略调参和独立评测集，以及独立评测前冻结的quality_budget.yaml。预先固定可接受PPL/任务质量代价；质量门槛不能结果出来后重写。

质量不达标时调整Scale或局部高精度回退，重新核算存储；高cosine不能替代任务质量。任务分数变化写百分点，PPL写绝对和相对变化。

### V6.6 组合消融

| 组 | 调度 | Cache | Decode |
|---|---|---|---|
| U | 冻结上游 | BF16 | 原后端 |
| A | 最佳固定Mixed | BF16 | 固定高精度后端 |
| B | SLO自适应 | BF16 | 与A相同 |
| C | 与A相同 | 8-bit | Gather参考 |
| D | 与A相同 | 8-bit | 融合Decode |
| E | SLO自适应+重标定 | 8-bit | 与D相同 |

B/A证明调度；D/C证明融合；E/D证明量化后的调度；E/B观察量化整体增量。自写BF16与原后端差异另列，不能混入量化收益。

### V6.7 公平容量实验

**等逻辑容量**：相同Block/Token容量，隔离访存与计算变化。

**等总字节预算**：相同模型/权重、总显存限制，计入Scale、工作区与Graph，允许量化获得更多Block，观察稳定并发、抢占和Goodput。

最大并发按固定输入/输出长度和持续时间定义，不用OOM前瞬时请求数。两组都报告KV payload、Scale、Workspace、Graph、峰值总显存。

### V6.8 实验矩阵

- Qwen3-4B统一快速验证、质量评估与主结果，冻结同一revision及BF16权重；
- 3090 TP=1/2；Blackwell独立扩展；
- 短/中/4K/8K/16K Context和模型实际范围；
- Poisson、Burst、长短混合、Prefix-sharing、Output-heavy、真实文本/长度回放；
- 相同绝对req/s及共同0.3–1.2×基线容量；
- BF16/FP8或INT8，Gather/Fused，Graph开/关；
- 至少3个seed，均值/标准差/样本数。

先代表子集筛选，再关键对照，不穷举所有维度。策略与Scale调参数据不进入独立评测。

主指标：双SLO Goodput、Cohort违约、TTFT/ITL分位数、吞吐、unfinished、最大等待/starvation、KV容量、recompute、Graph hit、Kernel延迟和质量。

### V6.9 Profiler与失效边界

NVTX：

```text
scheduler / prepare_mixed / metadata_h2d
kv_quant_store / kv_gather_dequant
prefill_attention / paged_decode / splitkv_merge
graph_replay / sampling / postprocess / nccl_collective
```

Nsight Systems看系统、拷贝、空隙和通信；Nsight Compute看Kernel。主性能实验不带Profiler。

必须回答：低负载控制器成本；短Context反量化开销；GQA组大小收益；Gather对TTFT；等容量无收益而扩容后有收益；权重访存主导；TP通信抵消；过载SLO不可满足；量化质量随长度变化。

### V6.10 产物与PR

README、设计/契约、环境、Tag、冻结配置、原始CSV/JSONL、绘图脚本、质量/性能报告、Profiler Trace。

优先小型PR：独立测试、Bug修复、指标或Kernel。完整研究策略保留fork。提交与合并分别表述，不预写已合入。

### V6.11 整版退出与最终声明

量化纯Decode Graph与TP=2回归通过，量化/融合/Graph对应预测器单独归档，独立任务质量满足预先冻结预算，组合消融与两类容量实验完成。每个关键对照保留至少3个seed、样本数、原始数据、冻结配置与可重放命令；未取得收益的shape和负载同样交付。性能结果、数值正确性、任务质量与功能兼容性分别陈述；简历只填写实际完成且有证据支持的内容，不能保证所有指标同时改善。

## 3. 配置、数据与回归规范

### 3.1 待实现配置

以下是设计字段，不是现有nano-vLLM可直接使用的CLI；null值须校准后显式填写。

```yaml
model:
  name: Qwen/Qwen3-4B
  path: /home/lq/vllm-1p1d-v029/model/Qwen3-4B
  revision: 1cfa9a7208912126459214e8b04321603b3df60c
  weight_dtype: bfloat16
scheduler:
  policy: slo_aware
  prefill_budget_candidates: [0, 256, 512, 1024, 2048, 4096]
  predictor_path: results/profiles/profile.json
  safety_margin_ms: null
  aging_enabled: true
  max_prefill_wait_ms: null
  kv_guard_enabled: true
kv_cache:
  dtype: fp8_e4m3fn        # bf16/int8/fp8_e4m3fn
  scale_granularity: per_token_head
  scale_dtype: fp32
  scale_mode: dynamic
  quantization_id: null
attention:
  prefill_backend: gather_dequant_fa
  decode_backend: triton_quant_paged
  shared_q_heads: 1
  split_kv: false
execution:
  tp_size: 1
  graph_mode: pure_decode
benchmark:
  warmup_s: 30
  measurement_s: 120
  drain_s: 60
  seed: 42
  baseline_saturation_rps: null
```

### 3.2 记录Schema

Run：schema_version、commit/dirty、模型revision、软件版本、拓扑/NCCL、配置、Trace hash、校准/预测器身份、质量预算。

Request：到达/入队/首次调度/所有Token/完成、状态、输入输出长度、SLO、等待、抢占、重算。

Step：

```text
step_id, start_ns, duration_ms, policy, prefill_budget
num_decode, prefill_tokens, sum_context, max_context
min_slack, predicted_ms, under_prediction_ms
kv_dtype, attention_backend, scale_scheme
used_blocks, total_blocks, kv_payload_bytes, scale_bytes
workspace_bytes, graph_mode, graph_hit, padded_tokens
preemptions, recomputed_tokens
```

未测字段填null，不填0；旧CSV与新Schema不能无声混合。

### 3.3 测试与门禁

| 类别 | 必测 |
|---|---|
| 测量 | Trace、enqueue lag、Drain、unfinished、单Token |
| 状态 | 中间Chunk、抢占、引用计数、尾块、Prefix发布 |
| 量化 | 零/极值/非有限、舍入、Scale、格式身份 |
| 写入 | 随机Slot、padding、重复Slot、部分有效页 |
| Attention | ragged、GQA、长Context、空分区、Merge |
| 集成 | Mixed、Prefix冷/热、Graph、页复用、TP |
| 质量 | Teacher forcing、PPL、长Context、独立评测 |

先correctness再性能。CPU CI跑参考与状态，GPU门禁用稳定shape；阈值由噪声实测确定。非量化Token对齐与量化质量预算不同，不能用100% BF16 Token匹配作为量化门槛。

## 4. 最新简历模板

### 大模型在线推理调度与量化KV Cache优化

**项目概述：** 面向Qwen3-4B在线混合负载中长Prompt阻塞Decode、固定Chunk Size难以适应动态负载及长上下文KV显存瓶颈，从混合批处理、SLO感知调度和量化Attention执行三个层面优化nano-vLLM，在控制精度与吞吐代价的同时，提高KV容量与满足TTFT/ITL双SLO的请求吞吐。

**技术栈：** PyTorch、Triton、nano-vLLM、Chunked Prefill、Paged KV Cache、FP8、GQA、FlashAttention、CUDA Graph、Tensor Parallel、Nsight Systems

- **混合Prefill/Decode执行重构：** 重构Scheduler、Sequence与ModelRunner，实现全局Token Budget约束下的Mixed Batch，优先调度Decode，并利用剩余预算分块处理Prefill；跨Step维护计算进度、Block Table、Slot Mapping与增量KV状态。通过XX组覆盖Prefix Cache、Block边界、Preemption和TP=2的测试，在非量化路径下Greedy Token匹配率达到XX%，长Prompt插入期间的Decode Stall由XX ms降至XX ms。

- **Decode-Slack自适应调度：** 根据ITL SLO、Token等待时间和预测Step时延计算Decode Slack，动态选择Prefill Budget，通过Hysteresis抑制预算振荡，并加入Prefill Aging与KV水位保护；相比最佳固定XX-token Chunk，在Qwen3-4B、[硬件配置]、0.8×饱和负载下，将双SLO违约率由XX%降至XX%、P99 ITL降低XX%，吞吐损失控制在XX%以内，并通过消融验证公平性与水位保护的独立收益。

- **FP8 Paged KV Cache与GQA Decode优化：** 基于Triton实现FP8 KV在线量化写入及Scale管理，重构直接读取量化分页KV的GQA Decode Kernel，融合Block Table寻址、分块反量化与Online Softmax，减少KV访存及高精度中间张量；Prefill接入Gather+反量化+FlashAttention参考路径，并验证Prefix Cache与CUDA Graph兼容性。相比BF16，在[上下文长度/并发条件]下，KV总存储降低XX%、Decode Kernel延迟降低XX%，SLO Goodput提升XX%，长上下文任务得分变化控制在XX个百分点以内。

填写要求：

- XX均用实测；未实现内容删除或标记计划。
- 第一条非量化对齐，第三条量化误差与任务质量分开。
- 最终INT8时技术栈、标题、图表同步修改。
- 违约率明确相对百分比或百分点，优先写从A%到B%。
- KV节省含Scale，说明是否含Workspace及容量对照条件。
- 每条最终选3–4个最有力数字；不单列泛化压测条目。
- 3090路径写量化存储、反量化后计算，不写原生FP8加速。

### 4.1 简历能力与版本证据对应

| 简历能力 | 主要版本 | 填写前需要的证据 |
|---|---|---|
| Mixed Batch、计算进度、Block/Slot、非量化对齐 | V1 | 状态矩阵、通过数量、Greedy与logits对齐报告 |
| 长Prompt期间Decode Stall改善 | V1＋V2 | 原始引擎与Mixed执行正式对照、相同负载及原始Token时间 |
| Slack、Hysteresis、Aging与KV Guard | V2＋V3 | 最佳固定基线、预测误差、五组独立消融与吞吐代价 |
| FP8写入、Scale、真实分页KV、Gather+FA、Prefix | V4 | 实际格式支持、真实显存含Scale核算、Cache生命周期与初步质量 |
| 融合反量化、GQA与Online Softmax | V5 | 同量化参考正确性、Gather/Fused/BF16微基准 |
| CUDA Graph、量化TP、SLO Goodput与长上下文质量 | V6 | Replay/TP回归、重标定profile、组合实验、独立质量报告 |

当前V0、V1已实现，后续版本内容仍按完成情况填写。外部简历若仍写Qwen3-8B，应与本项目Qwen3-4B实测模型同步；采用INT8后备时，技术栈、项目条目和图表同步改为INT8。版本合并保证功能与证据覆盖，不预先保证延迟降低、Goodput提升或质量达标。

## 5. 开工顺序与参考资料

### 5.1 每阶段流程

```text
理论 → 数据布局/执行顺序 → 简单参考 → 边界测试
  → 单模块替换 → correctness → 微基准 → 模型回归
  → 数据归档 → 版本报告 → Tag
```

阶段报告写：改动、基线、测试、收益、代价、失败原因和后续前置。没有可重复证据时不写成最终成果。

### 5.2 V0与V1之后的推进顺序

1. 核对V0/V1已有报告与缺失验收项，保留原实现、入口和数据，补齐缺口。
2. V2完成固定策略、共同容量与正式Decode Stall基线，冻结对照条件。
3. V3先Slack-only，再独立开启Aging与KV Guard，完成五组消融，冻结BF16最终策略。
4. V4依次通过V4-A/V4-B/V4-C，建立真实量化执行闭环，冻结质量预算。
5. V5替换Decode为融合Kernel，完成相同量化参考、Gather与BF16三组对照。
6. V6完成量化Graph/TP回归，重标定预测器，再跑独立质量、组合消融与公平容量实验。

内部检查点分别保留提交与测试记录，可在同一阶段报告中整理；正式版本只在整版退出条件满足后打Tag。已有通过证据在模型、配置与接口不受本次变更影响时可复用；新增功能或受影响路径须补回归。测量契约、测试矩阵、原始数据与失败记录不因合并版本删减。

### 5.3 参考链接

文档与代码核验于2026-09-29；开发时固定对应commit/依赖版本，API可随上游变化。

- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：引擎底座。
- [Scheduler](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)：冻结原始调度。
- [BlockManager](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py)：分配/Hash/引用。
- [Sarathi-Serve](https://github.com/microsoft/sarathi-serve)：Chunked Prefill。
- [SLAI](https://github.com/agrimUT/SLAI)：SLO实验组织，区分真实执行与仿真。
- [vLLM](https://github.com/vllm-project/vllm)：生产级后端。
- [vLLM量化KV文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)：Scale与格式参考。
- [vLLM Triton Unified Attention](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py)：分页、GQA与量化读取。
- [vLLM Triton后端解读](https://vllm-project.github.io/2026/03/04/vllm-triton-backend-deep-dive.html)：算法和布局。
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：Prefill后端，检查变长causal对齐。
- [Triton Attention教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：Online Softmax参考。
- [NVIDIA支持矩阵](https://nvidia.github.io/TransformerEngine/support_matrix.html)：低精度架构支持，不能替代自定义转换测试。
- [FastServe](https://github.com/MachineLearningSystem/NSDI26-FastServe)：可选抢占参考。

最终成果应能分别说明：执行语义为什么正确，预算策略为什么有效，量化如何节省存储，融合Kernel如何减少开销，以及这些收益在什么条件下消失。
