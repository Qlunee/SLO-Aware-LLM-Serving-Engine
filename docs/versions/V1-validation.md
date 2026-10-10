# V1补充数值与质量检查

这些检查不替换`test_v1.sh`，不改旧BF16阈值，也不自动宣布V1通过。新代码独立位于`benchmarks/v1_validation/`，不会改变旧候选记录的`benchmarks/v1/*.py`摘要；旧结果必须保留。

## 1. 检查边界

| 入口 | 参考与覆盖 | 不能证明什么 |
|---|---|---|
| `--cpu` | 概率指标、右下对齐因果掩码、GQA、FP32残差别名保护、错误输入 | GPU/模型实现正确 |
| `--attention` | 实际BF16 Mixed FA对独立FP32稠密参考；32Q/8KV/128维、249+8边界、混合行、物理页复用、损坏缓存负对照 | 完整模型/所有布局均正确 |
| `--semantic` | FP32完整/分块前向及真实混合Step；仍使用生产调度、Metadata、KV写入；Attention换为独立稠密数学参考 | BF16 FA数值、TP=2、Graph已验收 |
| `--calibrate` | 同权重、同历史：原始完整Prefill、原始分块Prefill、V1分块Prefill三方对照 | BF16偏差无害或可直接放宽阈值 |
| `--quality` | 长输出分叉；原始串行/批量自差异；同一文本历史下的概率和目标Token损失 | MMLU/真实业务质量已达标 |

原始执行路径复用已加载权重、继承的`ModelRunner.run/prepare_prefill`和原始`Scheduler`，不是另一个独立框架。独立算术参考为显式FP32 QK→因果掩码→Softmax→V，不调用FlashAttention。

## 2. 顺序执行命令

不要并行运行GPU命令，避免共享GPU及NCCL端口。无需重新安装环境。

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh

bash scripts/test_v1_validation.sh --cpu
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --attention

# FP32语义检查，额外包含原失败记录的seq_id=7输入。
# 旧Logits不会作为新版本参考，只导入历史Token并重新计算。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --semantic \
  --artifact results/raw/v1-check.R8ETmV/repeat_1/mixed-candidate

# 三个seed、四种长度、四种分块。只测量，不自动拟合阈值。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --calibrate \
  --lengths 255 257 513 1025 --chunks 128 249 256 512 --seeds 41 42 43 \
  --artifact results/raw/v1-check.R8ETmV/repeat_1/mixed-candidate

# 内置8段多段落中英文本：每请求固定生成128 Token。
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --quality \
  --cases 8 --output-tokens 128 --teacher-samples 4
```

`semantic`仅支持单卡Eager，冻结诊断门槛为相对L2<=0.001、最大绝对差异<=0.02且Top-1一致。内含255/513-token请求的混合Step和4-token短生成对照。FP32需要较多显存，因此限制为8个KV Block；它不是吞吐测试，也不能把小容量语义检查代替正式容量矩阵。

`calibrate`也仅支持单卡Eager。默认无后端精度开关修改，不把关闭低精度归约当作修复。

单卡Eager结果审阅后，再做长输出Graph/TP对照：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --quality \
  --graph --cases 8 --output-tokens 128 --teacher-samples 4

CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \
  bash scripts/test_v1_validation.sh --quality \
  --tp 2 --cases 8 --output-tokens 128 --teacher-samples 4
```

Graph只用于Decode，固定历史Prefill参考仍走Eager。TP=2日志中的链路情况需单独归因，不能把质量检查当性能结果。

## 3. 使用真实连续文本

内置材料是编写的多段落sanity fixtures，避免单句重复，但仍不是ShareGPT、真实流量或正式任务评测。真实质量结论必须增加独立连续文本或适配正式评测集。

准备UTF-8 JSONL，每行如下；`id`唯一，`text`包含连续正文，不是只有Prompt，也不应把模型本次生成当成目标答案。

```json
{"id":"article-001","text":"完整的连续文章正文……"}
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_validation.sh --quality \
  --corpus /absolute/path/heldout_texts.jsonl \
  --cases 8 --output-tokens 128 --teacher-samples 8
```

每段文本至少32 Token，正文Token数加固定输出长度不能超过4096。程序取正文前半段作为生成Prompt；固定历史评估则在后半段选取位置，用真实下一个Token作为目标。默认4个位置时并未覆盖全文，报告为**采样位置条件PPL比值**，不能写成完整语料PPL。

## 4. 如何读报告

程序输出`V1 validation artifacts: ...`，目录由`mkdtemp`唯一生成；显式`--out`要求新目录，不覆盖历史结果。

- `report.json`：状态、分组汇总、逐样本指标。
- `generations.json`：三种执行方式的Token、解码文本、首次分叉位置、最长相同前缀。
- `inputs.json`：精确Token输入及数据来源摘要。
- `logits.pt`：CPU Logits证据，可独立重算指标；只加载自己生成、可信的产物。
- `environment.json`、`config.json`、`artifact_hashes.json`：模型/代码/配置/产物摘要；权重文件记录大小及声明revision，不声称完整权重内容哈希。
- `error.json`：运行异常。

主要比较组：

1. `legacy_chunk_vs_legacy_full`：原始执行路径自身漂移。
2. `mixed_chunk_vs_legacy_full`：V1相对完整Prefill的跨形状差异。
3. `mixed_chunk_vs_legacy_chunk`：匹配分块形状后的实现差异。

概率指标包括原始/中心化Logits相对L2、KL(reference||actual)、JS散度、总变差、Top-1、Top-10重合率。中心化指标消除对Softmax无影响的整体Logits平移。KL/JS默认温度1，无采样随机性。

长输出比较必须同时看V1对原始批量、原始串行对原始批量；后者用于揭示基线自身的批量敏感性。分叉后的同位置Token匹配率只是描述性指标，不是准确率。教师强制指标始终使用相同来源正文历史，绝不在两个不同生成分支之间计算KL。

## 5. 状态与验收纪律

- CPU/GPU单元测试：`OK`仅表示对应测试边界通过。
- `semantic`：`PASS`仅表示该轮FP32语义检查通过，不表示全部V1完成。
- `calibrate`/`quality`：`MEASURED_NEEDS_REVIEW`，退出0仅代表测量完成。Top-1或长生成分叉不会被自动豁免；需结合基线与质量数据审阅。
- 所有报告固定保留`v1_acceptance_passed: false`，不会替你签署全项目验收。
- 不自动扩大BF16容差，不删除旧失败结果，不把Attention参考或Logits拷贝的耗时用于吞吐/SLO结论。

审阅后，才能决定是否实现形状稳定的计算、修正发现的具体Bug，或制定有质量证据支撑的数值验收规则。
