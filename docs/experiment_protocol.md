# V0实验与统计协议

## 统一模型基准（2026-10-10）

V0–V10统一使用Qwen/Qwen3-4B与BF16权重，身份/本地目录/冻结revision见configs/qwen3_4b.json。0.6B与8B不再是必需验收模型；模型、精度、TP、Graph、KV后端或长度配置改变后，重新profile并扫描容量，不能跨条件套用饱和RPS、SLO或旧加速比。

实际模型参数为36层、32 Q Head、8 KV Head、head_dim=128、GQA ratio=4。TP=2每rank为16 Q Head/4 KV Head。BF16 KV payload每Token跨全部层为147456 bytes（144KiB），TP=2每rank为73728 bytes（72KiB），不含页尾Padding/Scale/Block Table/Workspace/Graph。量化KV收益必须以该实际结构测量，不能照搬其他模型的KV字节数。hidden_size/num_heads不是本模型的head_dim。

## 时间与请求生命周期

所有时间为相对本轮perf_counter_ns的纳秒，Trace提前生成，计划arrival_ns不因GPU忙碌顺延。单线程仅在Step之间提交已到达请求；enqueue_lag必须计入TTFT。Token可见时间在同步返回的run及postprocess后记录，不把中间Prefill Chunk当输出。此测量不包含网络、HTTP或Tokenizer时间。

启动模型、分配KV和Graph Capture均在计时前；到达流Warmup发生在计时后。窗口为[warmup, warmup+measurement)，Drain截止为window_end+drain。原子Step无法中断，跨截止完成计timeout并报告deadline_overrun_ms。未来未入队请求仍保留Record；拒绝、未完成、超时、失败均不得从到达队列分母删除。异常运行valid_run=false，不能用于性能结论。

## 双SLO与Goodput

TTFT=首Token可见时间-计划到达时间。ITL为相邻输出可见时间差；请求级P99 ITL<=ITL SLO并且TTFT<=TTFT SLO才通过。TPOT单列，不混用替换ITL判据。max_output_tokens=1单列，不进入双SLO队列。ignore_eos=true使输出长度可控。

cohort_violation_rate以测量窗口内到达的双SLO请求为分母，Drain内完成可通过；window_goodput_rps以窗口内完成且满足双SLO的请求数除窗口秒数，不把Drain完成混进分子。window_throughput_tok_s以窗口内可见Token计数。两者必须一起报告，不能把arrival cohort和completion window混算。

分位数采用(n-1)q线性插值，输出token池化ITL分位数及请求级ITL P99分布。未完成请求的部分Token只用于已观测延迟分布，不能视作SLO通过。空Trace仅是合法到达生成结果，不是容量测量证据。

## 对照、复现与设备能力

同一个Trace、相同模型和源码、相同种子、相同Graph/TP/Token Budget/显存设置，分别运行native、minimal、full至少三次。native仍包含外部开放环提交及完成记录，不是完全零开销；其逐Token指标为null。对照原generate只用非空全arrival=0 Trace，双方argmax、ignore_eos一致。

冻结上游SHA+实际源码哈希、benchmark源码哈希、模型revision声明+模型元数据哈希和权重文件大小；这不等于完整权重内容校验，发布实验前还需确认权重来源与revision。生成的synthetic_tokens=true负载用于因果与正确性测试，不代表ShareGPT真实语义。

TP=2必须单独运行，保留NCCL_DEBUG=INFO日志、topology、P2P能力及实际NCCL链路；P2P可访问不等于NVLink启用。不得并行启动冻结上游引擎（固定NCCL端口）。

本机已实测GPU拓扑NV4，默认NCCL IB/RoCE路径在首次AllReduce段错误；本机TP=2实验使用NCCL_IB_DISABLE=1，实测NCCL通道P2P/CUMEM。该设置不禁用GPU间P2P，其他机器不能无验证照搬。environment.json新增记录NCCL_IB_DISABLE/NCCL_P2P_DISABLE/NCCL_NET/NCCL_SOCKET_IFNAME；旧产物缺失项不得事后补填假装当时已记录，以原命令日志补充说明。

量化探测区分INT8/PyTorch FP8/软件E4M3FN/Triton原生转换/Triton软件查表解码。每条路线隔离进程，编译abort或超时不能丢失其他报告。Graph测试更新同地址输入后回放。转换成功不证明原生FP8 Tensor Core、量化Attention正确或有加速；SM86/3090不具备原生FP8 Tensor Core路线。V0不启用量化KV，不提前承诺某个kernel。

## 饱和负载扫描

使用sweep逐RPS独立进程、三个seed，保留原始结果。先粗扫再细扫，检查窗口完成率/吞吐、到达率、backlog趋势、unfinished/timeout与enqueue lag。短烟雾窗口不估计饱和点；测量时长需覆盖稳态。后续0.3/0.6/0.8/1.0/1.2×扫描必须采用同一个已定义、同模型同配置的基线饱和点，不能各策略各选分母。sweep_summary.csv提供单轮结果，不自动宣称某RPS为饱和点。
