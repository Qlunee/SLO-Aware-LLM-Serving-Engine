import argparse

from benchmarks.defaults import model_path_default


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B离线示例，不是在线SLO基准")
    parser.add_argument("--model", default=model_path_default())
    parser.add_argument("--tp", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=args.tp)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
