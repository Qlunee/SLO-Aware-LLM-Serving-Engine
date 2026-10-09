# 相对上游的改动

冻结nano-vLLM：bb823b3e06983d71485a8e1f23715ebd87d98ef8。

初始V0仅增加benchmarks/v0、tests和scripts/test_v0.sh，不修改nanovllm调度选择、KV格式或Attention内核。观测通过ObservedEngine及Scheduler方法包装实现，请求Record保留在主进程；Greedy实验中参考与观测均替换为argmax，original sampler性能组单独报告。

FlashAttention、Paged KV Cache、Prefix Cache、CUDA Graph、Tensor Parallel及NCCL均属于上游能力，不是本项目新增实现。软件FP8探测仅验证格式、转换、Graph及转换时延，不等于实现FP8 KV Cache或量化Attention。

本轮保留并增强用户原有八个V0程序：增加Trace/参数校验、独立开放环Driver、有限Drain、异常结果保存、原始产物重算审计、依赖检查、CPU测试和分进程量化能力探测。

## 2026-10-10：统一Qwen3-4B基准

新增configs/qwen3_4b.json与benchmarks/defaults.py集中管理模型身份、目录和冻结revision；V0入口及GPU脚本读取同一默认值，保留显式覆盖和revision校验。example.py/bench.py改为共享4B默认模型并提供--model/--tp，未修改nanovllm模型或Attention实现。新增scripts/env_qwen3_4b.sh只导出变量、不安装环境。环境摘要补充NCCL路由设置及共享配置哈希，新增默认值回归测试。

README、V0文档、实验协议和本地项目手册同步取消“0.6B先验收/8B主结果”，V0–V10统一4B。历史原始实验不修改，上游SHA不变；本次配置修改不代表后续Mixed/量化功能已完成。
