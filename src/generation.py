"""Stage 3: report synthesis with a frozen LLM served by vLLM."""
from .prompts import build_conversation


def load_llm(cfg):
    from vllm import LLM

    return LLM(
        model=cfg["model"],
        dtype=cfg["dtype"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        quantization=cfg.get("quantization"),
        enforce_eager=cfg.get("enforce_eager", False),
        trust_remote_code=False,
    )


def generate(llm, user_prompts, cfg):
    """Generate one report per user prompt (outputs are returned in input order)."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
                            top_p=cfg["top_p"], seed=cfg.get("seed"))
    conversations = [build_conversation(p) for p in user_prompts]
    outputs = llm.chat(messages=conversations, sampling_params=params)
    return [out.outputs[0].text.strip() for out in outputs]
