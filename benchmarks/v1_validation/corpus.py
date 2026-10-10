"""Non-repeating bilingual sanity passages; NOT ShareGPT or a quality benchmark."""
import hashlib
import json
from pathlib import Path

PASSAGES = [
    "A small library kept a record of the books borrowed each afternoon. On Monday, a student returned a history book and asked for a guide to local birds. The librarian explained that the guide had been moved to the nature shelf beside the window. Before leaving, the student wrote down three questions for a weekend walk: which birds stayed through winter, where they built their nests, and how their songs differed. The next visit would be a chance to compare those notes with observations rather than rely on memory alone.",
    "A reliable experiment separates the variable being changed from the conditions that should remain fixed. Suppose two serving policies are compared on different request traces. Their measured latency may differ because of the policy, but it may also differ because one trace contains longer prompts. Replaying identical arrival times and token lengths makes the comparison easier to interpret. Repeating the trial with several seeds then shows how sensitive the conclusion is to workload variation. A short successful smoke test establishes that the pipeline runs; it does not establish a general performance advantage.",
    "To prepare vegetable soup, first wash and chop the carrots, onions, and potatoes into pieces of similar size. Heat a little oil in a pot and soften the onions without burning them. Add the carrots and potatoes, cover them with stock, and bring the mixture to a gentle simmer. Stir occasionally and taste before adding salt. When the vegetables are tender, remove the pot from the heat. Leftovers should be cooled promptly and stored safely. The useful lesson in this recipe is that the order of the steps matters as much as the list of ingredients.",
    "An online inference server handles requests that arrive at different times. Prompt processing can involve many tokens, while decoding usually advances each active request by one token. A scheduler must share limited compute and memory between these phases. Dividing a long prompt into smaller chunks can reduce the time that ongoing decodes wait, but excessively small chunks add overhead. The best setting depends on the workload and hardware. Any optimization should preserve token positions, request isolation, and the causal relationship between a query and the keys it is allowed to attend to.",
    "一家社区书店准备举办周末读书会。店主先询问读者喜欢的主题，再根据报名人数安排座位。有人希望讨论旅行经历，也有人更关心科学史。最后大家决定选择一本篇幅适中的纪实作品，并提前公布三个讨论问题。活动当天，每个人先分享一段印象深刻的内容，再说明自己为什么赞同或反对作者的观点。主持人提醒大家区分书中的事实、作者的判断和读者自己的推测。这样的安排让交流更有条理，也给第一次参加活动的人留下了发言机会。",
    "实验记录不仅需要保存最终的图表，也应保存模型版本、硬件配置、输入数据和运行参数。假如一次测试显示尾延迟降低，研究者仍需检查吞吐量是否下降、长请求是否等待更久，以及失败请求是否被遗漏。接近饱和与超过饱和的负载可能得到不同结论。只有把这些条件写清楚，其他人才能复现实验，并判断结果适用于哪些场景。一次没有报错的运行只是流程检查，不能代替正确性验证，也不能证明系统在所有负载下都能稳定工作。",
    "小林计划在周末完成一段徒步路线。他先查看地图，把总路程分成三个部分，并给每一段预留休息时间。出发前，他检查饮水、雨具和通信设备，还把预计返回时间告诉了家人。走到第二段时天气发生变化，他没有为了原定目标继续冒险，而是选择了较短的返回路线。回家后，他记录了实际用时和途中遇到的问题。这份记录帮助他下一次制定更合理的计划，也说明计划的价值在于支持判断，而不是要求人忽略变化。",
    "Consider a cache made of fixed-size physical blocks. A logical sequence may occupy blocks that are not adjacent in memory, so a block table maps logical positions to physical storage. A shared prefix can be reused only when the corresponding key and value entries are already valid. Newly computed tokens must not overwrite storage still referenced by another request. If a request is preempted and later recomputed, accounting should distinguish useful new work from repeated work. These rules are correctness requirements; memory savings and throughput improvements are separate measurements that need their own evidence.",
]


def load_corpus(path=None):
    if path is None:
        rows = [{"id": f"curated-{i}", "text": "\n\n".join(PASSAGES[(i+j) % len(PASSAGES)] for j in range(4))}
                for i in range(len(PASSAGES))]
        return (rows, {"kind": "curated_multi_paragraph_sanity_not_real_user_data",
                       "sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()})
    source = Path(path)
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("corpus row requires a nonempty string id")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("corpus row requires nonempty text")
        ids.append(row["id"])
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("empty corpus or duplicate ids")
    return rows, {"kind": "user_supplied_text", "path": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
