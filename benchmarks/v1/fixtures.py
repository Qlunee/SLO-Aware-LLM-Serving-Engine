"""Deterministic language-like fixtures; synthetic, NOT a real-user dataset."""
def language_prompt(tokenizer, length, index):
    prefix = tokenizer.encode(f"Request {index}. ")
    passage = tokenizer.encode(
        "Explain how an online language model server manages concurrent requests. "
        "Paged key and value caches store the attention history in physical blocks. "
        "A scheduler assigns a limited token budget to decoding and prompt processing. "
        "Correctness requires preserving token positions, causal masking, and request isolation. ")
    if not passage or length < 1:
        raise ValueError("invalid fixture length or tokenizer")
    return (prefix + passage * (length // len(passage) + 1))[:length]
