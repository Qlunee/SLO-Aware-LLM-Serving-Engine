# V2单卡Eager：固定Decode群体与长Prompt注入实验

本入口只实现最新版七版路线V2中的**受控Decode Stall对照子集**，不是全部V2容量/工作负载阶段，更不是V1数值验收豁免。程序位于`benchmarks/v2_stall/`；不修改生产`nanovllm/`、原V0/V1入口或数值阈值。

## 1. 默认实验与统计口径

- Qwen3-4B，同一revision、BF16权重/KV，TP=1，强制Eager。
- 8个Decode请求，输入256 Token、固定输出512 Token、忽略EOS。
- 从Engine计时原点起2秒时，计划注入一个3072-token长Prompt，输出64 Token。
- 观察窗口为共同计划时间`[2s, 4s)`；有限Drain为120秒。
- 预热：同批输入运行16-token代表性Prefill/Mixed/Decode；随后重建空BlockManager元数据，清除Prefix身份和观测日志，不重新分配权重/KV张量。
- 原始调度、固定Chunk=256/512/1024/2048共用全局Token Budget=4096、上下文上限4096、max_num_seqs和显存比例。每请求Chunk上限不等于Step总Prefill Budget。
- 调参seed=141/142/143；评测seed=241/242/243。每个seed有不同Token输入，同一配对共享完全相同Trace。使用人工合成语言fixture，不冒充ShareGPT、真实流量或真实任务质量。

### Decode稳定与固定群体

在**计划注入时刻**检查每个Decode请求：已有至少9个输出Token，即8个完整ITL；最近8个ITL的总体CV<=0.5；输出足够新鲜，且请求尚未完成。窗口结束前所有原Decode请求必须仍在生成，并有覆盖窗口末端的输出证据。实际注入发生在第一个不早于计划时刻的Step边界，入队延迟单列且必须<=250ms。

上述条件不满足时仍保留请求与运行产物，标为无效轮次。不得延后计划到达、减少Decode请求数量、增加输出长度后覆盖原数据，或静默重试直至“通过”。如需改变时间/长度/阈值，另建新suite，保留旧suite。

### Stall、TTFT与吞吐

主Stall为所有原Decode请求中，与`[inject, window_end)`重叠的完整相邻Token间隔的最大值。若一个暂停从注入前开始、观察窗口后才结束，仍完整计入，不裁剪或遗漏。另报池化P99 ITL、每请求Stall、注入前ITL、长Prompt TTFT、实际入队延迟、全局窗口吞吐与双SLO。

吞吐为共同窗口内CPU可见输出Token/s，不是服务最大容量。整批完成跨度、preemption/recompute、KV容量也保存。全局TTFT/Cohort指标主要包含窗口内到达的长请求；不要把该单请求Cohort比例解读为大规模在线SLO估计。

峰值显存是测量Trace及Drain期间的**rank0 PyTorch allocated/reserved峰值**，不是NVML进程总显存。初始化、预热不在计时窗口；预热保留的Allocator reserve可能体现在reserved读数中。

## 2. 测试与运行命令

在服务器项目根目录执行；GPU命令依次运行，不要并行。可以放在tmux会话中。

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh

# CPU契约与时间/统计测试；不会加载模型或使用GPU。
bash scripts/test_v2_stall.sh --cpu

# 新suite，冻结配置、源码、模型元数据和全部Trace。
V2_SUITE="$(mktemp -d "$PWD/results/raw/v2-stall.XXXXXX")"
bash scripts/test_v2_stall.sh --prepare --suite "$V2_SUITE"
echo "$V2_SUITE"

# 一对冒烟运行，不参与参数选择或最终统计。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v2_stall.sh --smoke --suite "$V2_SUITE"
```

**先看smoke是否有效，再运行以下命令。**不是有效就反复重跑：程序会复用旧轮次，不会覆盖。若稳定性/长度等假设不成立，保留该suite，重新约定配置并建立新suite。

```bash
# 调参：3个seed × (原始 + 4种固定Chunk)，共15个新进程。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v2_stall.sh --tune --suite "$V2_SUITE"

# 所选配置的状态/数值回归 + 两种引擎的独立逐Step注入状态扫描。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v2_stall.sh --regress --suite "$V2_SUITE"

# 正式独立评测：3个新seed × (原始 + 已冻结Chunk)，共6个新进程。
# V1数值未闭环，因此仍标记为探索性结果，不签署整体V1验收。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v2_stall.sh --evaluate --suite "$V2_SUITE"

# 每个有分歧请求的首次分叉：共同历史下重新计算并保存Logits/Top-2间距。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v2_stall.sh --diagnose --suite "$V2_SUITE"

# CPU重算审计与汇总，生成新的report.XXXX目录，不覆盖历史报告。
bash scripts/test_v2_stall.sh --report --suite "$V2_SUITE"
bash scripts/test_v2_stall.sh --verify --suite "$V2_SUITE"
```

如换了终端，先`source scripts/env_qwen3_4b.sh`，再将`V2_SUITE`赋为第一次打印的**实际绝对路径**，不要再次mktemp后期望读取旧结果。

所有子进程日志写入对应`.log`，启动时打印路径。各运行独立初始化模型，运行时间还包含512-token完整生成和数值参考计算；不要按2秒观察窗口估算整套耗时。

## 3. 参数选择与数据纪律

选择规则在prepare时冻结：从每个seed都有效的候选中，要求平均窗口吞吐损失<=5%、每轮长Prompt TTFT<=1000ms，再选择平均每轮最大Stall最小的Chunk；并列取较小Chunk。无可行候选保留负结果，`selection.chunk=null`，不硬选、不事后放宽阈值。评测集不参与选择。

任何一个必要配对无效/缺失，该候选不汇总收益；不会丢掉无效轮次后对剩余数据取均值。每个最终指标报告均值、样本标准差（n-1）、最小/最大值和n，另给配对Stall降幅、吞吐变化。n=3是三个独立seed的配对，不是同一个输入的三次独立文本。

若需要不同负载/长度，prepare提供以下配置项；只作用于新suite：

```bash
# 例：只有明确需要更晚注入/更长输出时才另建一轮，不覆盖原suite。
V2_NEXT_SUITE="$(mktemp -d "$PWD/results/raw/v2-stall.XXXXXX")"
bash scripts/test_v2_stall.sh --prepare --suite "$V2_NEXT_SUITE" \
  --inject-s 3 --decode-output-tokens 768
```

还支持`--decode-count 1|2|8|32`、`--window-s`、`--long-prompt-tokens`、`--max-model-len`、`--token-budget`、`--chunks`。原始全Prefill必须能装入全局Token Budget，输入加输出必须符合模型长度；容量不足直接报错，不静默截断或依赖抢占维持实验。

当前分歧诊断复用的高层参考引擎上限4096；若新suite扩大Context，必须先扩展并验证该诊断入口，建立新源码冻结，不能声称已有4K检查覆盖8K/16K。

## 4. 状态回归与数值分歧

`regress`运行现有CPU状态、GPU Mixed Attention和独立FP32 Attention参考单元测试；对所选Chunk运行mixed/prefix/preempt候选及原始BF16数值参考，保留原门槛。另使用评测集第一条Trace在两种引擎中逐Step扫描计算进度、每Step采样数、页引用与空闲分区、输出长度；这些时间不参与性能统计。

`regression/report.json`分别记录`state_checks_passed`和`legacy_numerical_checks_passed`。后者仍失败时不会改成通过；状态通过仅允许带“探索性”标记继续V2。状态失败时禁止evaluate。

每个性能配对的全部输出与首次分叉保存在报告`pairs.json`。`diagnose`保存每个分歧请求的共同历史，并用同一权重重新计算完整Prefill、原始分块Prefill、Mixed分块Prefill的Logits，报告误差、Top-1/Top-2及间距。**这不是原在线Decode Batch逐层重放**，不能独立证明分叉原因，也不自动豁免近似并列。所有分歧保持`NEEDS_REVIEW`，需要结合已有/新增状态、KV与逐层证据人工检查。不得按输出匹配与否筛选性能样本。

## 5. 冻结关系与产物

旧`results/raw/v1-protocol.2_c5wbrp`保持原样。新增源码自然使其“完整源码集合相等”检查不再通过，这是新阶段源码扩展，不应修改旧manifest掩盖。prepare会核验旧协议档案、旧证据及旧清单内每个源码仍完全一致，再将新增V2源码和旧协议哈希记入新协议。之后使用新`--verify`入口。

```text
suite/
  protocol.json / artifact_hashes.json
  tune/seed-141.json ...       # 冻结输入
  eval/seed-241.json ...
  smoke/                      # 不进入正式统计
  runs/{tune,eval}/            # summary/requests/steps/backlog/environment/manifest
  selection/                  # 调参所有配对、聚合与不可覆盖的选择
  regression/                 # 状态与原数值参考分别记录
  divergence-review/          # 共同历史、Logits、误差和审阅状态
  report.XXXX/                # pairs/aggregate/report.md，重跑另建报告
```

成功轮次可断点复用并核验哈希；中断造成不完整manifest的轮次不得当作完成或直接覆盖。保留原suite并新建明确记录的新实验。新增代码后必须prepare新suite，不能在运行中改注释或程序。

填写简历前：所选配置状态回归通过；三组独立评测都有效；指标审计通过；分歧已记录并完成对应审阅；数字与实际配置、统计口径一致。当前入口不承诺收益，也不会将V1完整验收状态改为通过。
