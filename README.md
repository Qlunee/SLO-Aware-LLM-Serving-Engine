# SLOPrefill — SLO-Aware-LLM-Serving-Engine

大模型在线推理调度与量化KV Cache优化。底座为冻结版[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，统一实验模型为**Qwen/Qwen3-4B，BF16权重**。

## 统一模型基准

配置唯一来源：`configs/qwen3_4b.json`。本服务器默认权重目录为`/home/lq/vllm-1p1d-v029/model/Qwen3-4B`，冻结revision为`1cfa9a7208912126459214e8b04321603b3df60c`。

Qwen3-4B为36层、32个Q Head、8个KV Head、Head Dim=128，GQA ratio=4。TP=2时每rank为16个Q Head和4个KV Head。**Head Dim不能用hidden_size / num_attention_heads推算**，本模型2560/32=80与实际128不同。

V0至V6的正确性、容量、调度与量化主结果均使用4B，不再要求先0.6B、后8B。其他模型只作为明确标注的扩展实验，不混入4B主结果。模型配置支持40960个位置不代表默认引擎长度或单卡显存允许这么长；V0正式基线默认max-model-len=4096，长Context实验须显式调整并重新测容量。

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
- V0在线粗扫结果保留；观测开销与精确稳态容量仍需独立验收。V1已新增可选的BF16 Mixed Batch实现与测试入口，完整验收范围见下文。自适应调度、量化KV和融合Decode仍未实现。

完整命令见`docs/versions/V0.md`，统计口径见`docs/experiment_protocol.md`。既有results/raw产物不可改写；新代码/配置产生的新实验独立归档。

## V1：固定Chunk的混合执行

新增独立`MixedLLMEngine`，不改变原V0引擎。全局Token Budget下Decode-first选择请求，各层共享投影/MLP，分phase调用现有FlashAttention；中间Chunk不采样，跨Step维护KV与Prefix/抢占状态。Mixed走Eager，纯Decode保留已有Graph。

```bash
bash scripts/test_v1.sh --cpu
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1.sh --gpu --tp 1
CUDA_VISIBLE_DEVICES=0 bash scripts/test_v1_online.sh
```

标准语言fixture与随机ID压力测试分开；保留Greedy/Top-1与数值误差报告，不宣称任意输入bitwise一致。测试脚本显式使用初版BF16数值策略，严格逐元素模式仍可单独运行。完整矩阵、双卡/Graph命令、已知限制和产物见[docs/versions/V1.md](docs/versions/V1.md)。没有新增Slack控制、量化或自研Attention，不用烟雾数据填写性能收益。

## 仓库与实验产物

GitHub保存源代码、配置、测试、运行脚本、项目手册及阶段报告。当前路线为V0–V6共7个正式版本，详见[项目手册](docs/project_manual.md)。

完整实验数据保存在实验服务器，`results/raw/`、`results/processed/`、`logs/`及Nsight原始Trace目录不纳入后续提交。原始CSV/JSONL、logits张量、临时运行目录和模型权重不随代码发布；`.gitignore`不会删除服务器已有文件。

复现实验时按`docs/versions/`中的命令生成本地结果，保留模型revision、commit、配置、seed、Trace身份和原始数据。阶段报告中的服务器数据路径是证据位置，不表示克隆仓库会同时获得完整实验数据。需要展示的汇总表、关键图及复现说明经过筛选后单独提交；未验收结果不作为性能成果。
