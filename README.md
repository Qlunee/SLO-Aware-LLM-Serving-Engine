# SLOPrefill — SLO-Aware-LLM-Serving-Engine

大模型在线推理调度与量化KV Cache优化。底座为冻结版[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，统一实验模型为**Qwen/Qwen3-4B，BF16权重**。

## 统一模型基准

配置唯一来源：`configs/qwen3_4b.json`。本服务器默认权重目录为`/home/lq/vllm-1p1d-v029/model/Qwen3-4B`，冻结revision为`1cfa9a7208912126459214e8b04321603b3df60c`。

Qwen3-4B为36层、32个Q Head、8个KV Head、Head Dim=128，GQA ratio=4。TP=2时每rank为16个Q Head和4个KV Head。**Head Dim不能用hidden_size / num_attention_heads推算**，本模型2560/32=80与实际128不同。

V0至V10的正确性、容量、调度与量化主结果均使用4B，不再要求先0.6B、后8B。其他模型只作为明确标注的扩展实验，不混入4B主结果。模型配置支持40960个位置不代表默认引擎长度或单卡显存允许这么长；V0正式基线默认max-model-len=4096，长Context实验须显式调整并重新测容量。

## 环境与入口

在服务器项目根目录执行：

```bash
cd /home/lq/project/SLO-Aware-LLM-Serving-Engine
source scripts/env_qwen3_4b.sh
bash scripts/test_v0.sh --cpu
CUDA_VISIBLE_DEVICES=0 "$V0_PYTHON" -m benchmarks.v0.preflight --tp 1
```

`env_qwen3_4b.sh`仅设置V0_PYTHON、V0_MODEL、V0_MODEL_REVISION，不安装依赖、不启动推理、不改变NCCL传输。新开Shell/tmux窗口后重新source；恢复已有tmux窗口不必重复。配置指向本地权重，不会自动下载模型。

workload/run/sweep和GPU测试脚本默认读取4B基准，仍兼容显式`--model`/`--model-revision`或环境变量覆盖；自定义模型目录须同时给出revision。`example.py`、`bench.py`也使用统一模型，但它们只是离线示例，不代替在线SLO测试。

## GPU回归

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v0.sh --gpu --tp 1 --repeats 3
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v0.sh --gpu --tp 1 --repeats 3 --graph
CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 bash scripts/test_v0.sh --gpu --tp 2 --repeats 3
```

本机双卡NCCL默认IB/RoCE路径发生段错误，NCCL_IB_DISABLE=1可绕过；它不禁用GPU间P2P/NVLink。不要并行启动引擎，冻结上游使用固定NCCL端口。实际链路和该环境变量必须进入实验记录，不能据此推断所有服务器都需要禁用IB。

## 当前状态（2026-10-10）

- Qwen3-4B单卡Eager、单卡Graph、TP=2 Eager各3轮×3种观测模式，27次Greedy对照全部通过，Token匹配率100%。
- 38项CPU回归通过；省略模型参数的默认4B真实推理/审计/Greedy对照通过。3090的Triton原生E4M3FN转换不可用，软件/查表路径通过，不能宣称原生FP8 Tensor Core支持。
- 在线稳态容量、观测开销和V0最终报告仍需验收；V1 Mixed Batch、自适应调度、真实量化KV和融合Decode属于后续阶段，不因模型支持而视为完成。

完整命令见`docs/versions/V0.md`，统计口径见`docs/experiment_protocol.md`。既有results/raw产物不可改写；新代码/配置产生的新实验独立归档。
