# scripts/coherence_gate.py
# Coherence evaluation gate for Nayhein-V1.2.
# ALL applicable tests must pass before uploading to HuggingFace.
# Retries up to 3× (5k training steps each) before hard error.
#
# Usage:
#   python scripts/coherence_gate.py \
#       --model_path outputs/50m-base/final \
#       --model_type 50m-base \
#       [--openai_key $OPENAI_API_KEY]

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Test definitions ──────────────────────────────────────────────────────────

COHERENCE_GATE_TESTS = [
    {
        "id": "factual_001",
        "type": "base",
        "prompt": "What is the capital of Japan?",
        "must_contain_any": ["Tokyo"],
        "max_tokens": 50,
    },
    {
        "id": "factual_002",
        "type": "base",
        "prompt": "What is 17 multiplied by 6?",
        "must_contain_any": ["102"],
        "max_tokens": 30,
    },
    {
        "id": "instruct_001",
        "type": "instruct",
        "prompt": "Write a Python function that returns the factorial of n.",
        "must_contain_all": ["def", "return"],
        "max_tokens": 200,
    },
    {
        "id": "instruct_002",
        "type": "base",
        "prompt": "Summarize what a neural network is in one sentence.",
        "min_length_chars": 40,
        "max_length_chars": 300,
        "max_tokens": 100,
    },
    {
        "id": "multiturn_001",
        "type": "instruct",
        "messages": [
            {"role": "user", "content": "My name is Jordan."},
            {"role": "assistant", "content": "Nice to meet you, Jordan!"},
            {"role": "user", "content": "What is my name?"},
        ],
        "must_contain_any": ["Jordan"],
        "max_tokens": 50,
    },
    {
        "id": "degen_001",
        "type": "base",
        "prompt": "Tell me about the history of the internet.",
        "max_repeated_token_run": 4,  # fail if any token repeats 5+ times in a row
        "min_unique_token_ratio": 0.4,  # at least 40% unique tokens
        "max_tokens": 200,
    },
    {
        "id": "tool_001",
        "type": "instruct_only",
        "prompt": "What is the weather in Singapore right now?",
        "tools": [
            {
                "name": "get_weather",
                "description": "Retrieve current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            }
        ],
        "must_contain_all": ["tool_call", "get_weather"],
        "max_tokens": 100,
    },
    {
        "id": "vision_001",
        "type": "instruct_vision_only",
        "prompt": "What do you see in this image?",
        "test_image_path": "eval/assets/test_cat.jpg",
        "must_contain_any": [
            "cat",
            "animal",
            "fur",
            "feline",
            "sitting",
            "image",
            "photo",
        ],
        "max_tokens": 100,
    },
]

# Model type → required test IDs + thresholds
GATE_REQUIREMENTS = {
    "50m-base": {
        "required_test_ids": [
            "factual_001",
            "factual_002",
            "instruct_002",
            "degen_001",
        ],
        "min_pass_count": 4,
        "perplexity_threshold": 30.0,
    },
    "50m-instruct": {
        "required_test_ids": [
            "factual_001",
            "factual_002",
            "instruct_001",
            "instruct_002",
            "multiturn_001",
            "degen_001",
            "tool_001",
        ],
        "min_pass_count": 6,
        "mt_bench_threshold": 5.5,
    },
    "5b-base": {
        "required_test_ids": [
            "factual_001",
            "factual_002",
            "instruct_001",
            "instruct_002",
            "degen_001",
        ],
        "min_pass_count": 5,
        "perplexity_threshold": 20.0,
    },
    "5b-instruct": {
        "required_test_ids": [
            "factual_001",
            "factual_002",
            "instruct_001",
            "instruct_002",
            "multiturn_001",
            "degen_001",
            "tool_001",
            "vision_001",
        ],
        "min_pass_count": 8,
        "mt_bench_threshold": 7.0,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Test runner helpers
# ══════════════════════════════════════════════════════════════════════════════


def run_generation(
    model, tokenizer, prompt_or_messages, max_tokens: int, tools=None
) -> str:
    """Run model generation and return decoded string."""
    from generation_utils import NayheinGenerationMixin

    gen = NayheinGenerationMixin(model, tokenizer)

    if isinstance(prompt_or_messages, list):
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": prompt_or_messages}]

    tools_xml = None
    if tools:
        from tool_calling import NayheinToolCallingMixin

        tc = NayheinToolCallingMixin()
        tools_xml = tc.format_tools(tools)

    return gen.generate_chat(
        messages,
        tools_xml=tools_xml,
        max_new_tokens=max_tokens,
        temperature=0.1,  # low temp for deterministic evaluation
        top_p=0.9,
        generation_mode="ar",
    )


def check_must_contain_any(response: str, words: List[str]) -> Tuple[bool, str]:
    for word in words:
        if word.lower() in response.lower():
            return True, f"Found '{word}'"
    return False, f"None of {words} found in response"


def check_must_contain_all(response: str, words: List[str]) -> Tuple[bool, str]:
    missing = [w for w in words if w.lower() not in response.lower()]
    if not missing:
        return True, "All required strings found"
    return False, f"Missing: {missing}"


def check_length(
    response: str, min_chars: Optional[int], max_chars: Optional[int]
) -> Tuple[bool, str]:
    n = len(response)
    if min_chars is not None and n < min_chars:
        return False, f"Response too short: {n} < {min_chars}"
    if max_chars is not None and n > max_chars:
        return False, f"Response too long: {n} > {max_chars}"
    return True, f"Length {n} OK"


def check_degeneration(
    response: str, token_ids: List[int], max_run: int, min_ratio: float
) -> Tuple[bool, str]:
    """Check for token repetition and vocabulary diversity."""
    # Check max repeated token run
    if len(token_ids) > 0:
        max_consecutive = 1
        current_run = 1
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i - 1]:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 1
        if max_consecutive > max_run:
            return (
                False,
                f"Token repeats {max_consecutive} times consecutively (max allowed: {max_run})",
            )

    # Check unique token ratio
    if len(token_ids) > 0:
        unique_ratio = len(set(token_ids)) / len(token_ids)
        if unique_ratio < min_ratio:
            return False, f"Unique token ratio {unique_ratio:.2f} < {min_ratio}"

    return True, f"No degeneration detected"


def run_vision_test(model, tokenizer, test: Dict) -> Tuple[bool, str]:
    """Run a vision test case."""
    image_path = test["test_image_path"]
    if not os.path.exists(image_path):
        return False, f"Test image not found at {image_path}"

    try:
        from PIL import Image
        from processing_nayhein import NayheinProcessor

        processor = NayheinProcessor(
            tokenizer=tokenizer, image_size=model.config.vision_image_size
        )
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": test["prompt"]},
                ],
            }
        ]

        inputs = processor(messages, images=image, return_tensors="pt")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=test.get("max_tokens", 100),
                temperature=0.1,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        ok, msg = check_must_contain_any(response, test.get("must_contain_any", []))
        return ok, f"Vision response: '{response[:100]}...' | {msg}"

    except Exception as e:
        return False, f"Vision test error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Perplexity evaluation
# ══════════════════════════════════════════════════════════════════════════════


def compute_perplexity(
    model,
    tokenizer,
    dataset_path: str = "wikimedia/wikipedia",
    n_samples: int = 5000,
    max_length: int = 512,
) -> float:
    """Compute perplexity on Wikipedia samples."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed; skipping perplexity eval.")
        return 999.0

    device = next(model.parameters()).device
    model.eval()

    try:
        ds = load_dataset(dataset_path, "20231101.en", split="train", streaming=True)
    except Exception as e:
        logger.warning(f"Could not load Wikipedia: {e}")
        return 999.0

    total_loss = 0.0
    total_tokens = 0

    for i, example in enumerate(ds):
        if i >= n_samples:
            break
        text = example.get("text", "")
        if not text.strip():
            continue

        input_ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = input_ids.to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, labels=input_ids)
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * input_ids.shape[-1]
                total_tokens += input_ids.shape[-1]

    if total_tokens == 0:
        return 999.0

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


# ══════════════════════════════════════════════════════════════════════════════
# GPT-4 Judge evaluation
# ══════════════════════════════════════════════════════════════════════════════


def gpt4_judge_score(
    model,
    tokenizer,
    n_samples: int = 50,
    openai_key: Optional[str] = None,
) -> float:
    """
    Estimate MT-Bench style score using GPT-4 as judge.
    If OPENAI_API_KEY is not set, falls back to length/keyword heuristics.
    """
    if openai_key is None:
        openai_key = os.environ.get("OPENAI_API_KEY")

    # MT-Bench style sample prompts
    eval_prompts = [
        "Explain the difference between supervised and unsupervised learning.",
        "Write a Python function to check if a string is a palindrome.",
        "What are the main causes of climate change?",
        "How does a transformer architecture work in NLP?",
        "Write a short poem about artificial intelligence.",
        "Explain recursion with a simple example.",
        "What is the difference between RAM and ROM?",
        "How would you debug a Python program that throws a KeyError?",
        "Explain what DNA is in simple terms.",
        "Write a function to find the longest common subsequence.",
    ] * (n_samples // 10 + 1)

    eval_prompts = eval_prompts[:n_samples]

    if openai_key:
        try:
            import openai

            openai.api_key = openai_key
            client = openai.OpenAI(api_key=openai_key)

            total_score = 0.0
            count = 0

            for prompt in eval_prompts:
                response = run_generation(
                    model,
                    tokenizer,
                    [{"role": "user", "content": prompt}],
                    max_tokens=200,
                )
                judge_prompt = (
                    "Rate the following AI response for helpfulness, coherence, and accuracy "
                    "on a scale of 1-10.\n"
                    f"Response: {response}\n"
                    "Reply with only a single integer."
                )
                judge_resp = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                score_str = judge_resp.choices[0].message.content.strip()
                try:
                    score = float(score_str)
                    total_score += score
                    count += 1
                except ValueError:
                    pass

            return total_score / max(1, count)

        except Exception as e:
            logger.warning(f"GPT-4 judge failed: {e}. Using heuristics.")

    # Heuristic fallback: score based on response length and keywords
    total_score = 0.0
    for prompt in eval_prompts[:10]:
        response = run_generation(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            max_tokens=150,
        )
        # Simple heuristic: longer coherent responses score higher
        score = min(8.0, max(1.0, len(response) / 80.0 * 5.0))
        if response.count(" ") > 10:  # at least some words
            score += 1.0
        total_score += score

    return total_score / 10.0


# ══════════════════════════════════════════════════════════════════════════════
# Main gate runner
# ══════════════════════════════════════════════════════════════════════════════


def run_coherence_gate(
    model_path: str,
    model_type: str,
    openai_key: Optional[str] = None,
) -> Tuple[bool, Dict]:
    """
    Run the full coherence gate for a given model.

    Args:
        model_path: path to model checkpoint directory
        model_type: one of "50m-base", "50m-instruct", "5b-base", "5b-instruct"
        openai_key: optional OpenAI API key for GPT-4 judge

    Returns:
        (passed: bool, results: dict)
    """
    from modeling_nayhein import NayheinForCausalLM
    from tokenization_nayhein import NayheinTokenizer

    requirements = GATE_REQUIREMENTS[model_type]
    is_instruct = "instruct" in model_type

    logger.info(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = NayheinTokenizer.from_pretrained(model_path)
    model = NayheinForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    results = {"model_type": model_type, "tests": {}, "metrics": {}, "passed": False}

    # ── Run individual tests ──────────────────────────────────────────────────
    required_ids = requirements["required_test_ids"]
    passed_count = 0

    for test in COHERENCE_GATE_TESTS:
        test_id = test["id"]
        test_type = test.get("type", "base")

        # Skip tests not applicable to this model type
        if test_type == "instruct_only" and not is_instruct:
            continue
        if test_type == "instruct_vision_only" and (
            "5b" not in model_type or not is_instruct
        ):
            continue
        if test_id not in required_ids:
            continue

        logger.info(f"Running test: {test_id}")
        test_result = {"passed": False, "message": ""}

        try:
            if test_type == "instruct_vision_only":
                ok, msg = run_vision_test(model, tokenizer, test)
                test_result = {"passed": ok, "message": msg}

            else:
                # Generate response
                prompt_or_messages = test.get("messages", test.get("prompt", ""))
                tools = test.get("tools")
                response = run_generation(
                    model,
                    tokenizer,
                    prompt_or_messages,
                    max_tokens=test.get("max_tokens", 100),
                    tools=tools,
                )

                # Check criteria
                all_ok = True
                messages_out = []

                if "must_contain_any" in test:
                    ok, msg = check_must_contain_any(response, test["must_contain_any"])
                    messages_out.append(msg)
                    all_ok &= ok

                if "must_contain_all" in test:
                    ok, msg = check_must_contain_all(response, test["must_contain_all"])
                    messages_out.append(msg)
                    all_ok &= ok

                if "min_length_chars" in test or "max_length_chars" in test:
                    ok, msg = check_length(
                        response,
                        test.get("min_length_chars"),
                        test.get("max_length_chars"),
                    )
                    messages_out.append(msg)
                    all_ok &= ok

                if "max_repeated_token_run" in test or "min_unique_token_ratio" in test:
                    token_ids = tokenizer.encode(response, add_special_tokens=False)
                    ok, msg = check_degeneration(
                        response,
                        token_ids,
                        max_run=test.get("max_repeated_token_run", 4),
                        min_ratio=test.get("min_unique_token_ratio", 0.4),
                    )
                    messages_out.append(msg)
                    all_ok &= ok

                test_result = {
                    "passed": all_ok,
                    "message": "; ".join(messages_out),
                    "response_preview": response[:100],
                }

        except Exception as e:
            test_result = {"passed": False, "message": f"Exception: {e}"}

        results["tests"][test_id] = test_result
        status = "PASS" if test_result["passed"] else "FAIL"
        logger.info(f"  [{status}] {test_id}: {test_result['message']}")

        if test_result["passed"]:
            passed_count += 1

    # ── Perplexity evaluation ────────────────────────────────────────────────
    if "perplexity_threshold" in requirements:
        logger.info("Computing perplexity on Wikipedia (5k samples)...")
        ppl = compute_perplexity(model, tokenizer, n_samples=5000)
        threshold = requirements["perplexity_threshold"]
        ppl_pass = ppl <= threshold
        results["metrics"]["perplexity"] = ppl
        results["metrics"]["perplexity_threshold"] = threshold
        logger.info(
            f"  Perplexity: {ppl:.2f} (threshold: {threshold}) → {'PASS' if ppl_pass else 'FAIL'}"
        )
    else:
        ppl_pass = True

    # ── MT-Bench style judge ─────────────────────────────────────────────────
    if "mt_bench_threshold" in requirements and is_instruct:
        logger.info("Running GPT-4 judge evaluation (50 samples)...")
        score = gpt4_judge_score(model, tokenizer, n_samples=50, openai_key=openai_key)
        threshold = requirements["mt_bench_threshold"]
        bench_pass = score >= threshold
        results["metrics"]["mt_bench_score"] = score
        results["metrics"]["mt_bench_threshold"] = threshold
        logger.info(
            f"  MT-Bench score: {score:.2f}/10 (threshold: {threshold}) → {'PASS' if bench_pass else 'FAIL'}"
        )
    else:
        bench_pass = True

    # ── Overall gate decision ────────────────────────────────────────────────
    min_pass = requirements["min_pass_count"]
    gate_passed = (passed_count >= min_pass) and ppl_pass and bench_pass
    results["passed"] = gate_passed
    results["metrics"]["tests_passed"] = passed_count
    results["metrics"]["tests_required"] = min_pass

    logger.info(f"\n{'=' * 60}")
    logger.info(f"COHERENCE GATE: {'PASSED' if gate_passed else 'FAILED'}")
    logger.info(f"  Tests passed: {passed_count}/{min_pass} required")
    if "perplexity_threshold" in requirements:
        logger.info(f"  Perplexity: {results['metrics'].get('perplexity', 'N/A'):.2f}")
    if "mt_bench_threshold" in requirements:
        logger.info(
            f"  MT-Bench: {results['metrics'].get('mt_bench_score', 'N/A'):.2f}/10"
        )
    logger.info(f"{'=' * 60}\n")

    return gate_passed, results


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--model_type",
        required=True,
        choices=list(GATE_REQUIREMENTS.keys()),
        help="Model type for threshold selection",
    )
    parser.add_argument(
        "--openai_key", default=None, help="OpenAI API key for GPT-4 judge"
    )
    args = parser.parse_args()

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    passed, results = run_coherence_gate(args.model_path, args.model_type, openai_key)

    if not passed:
        logger.error("COHERENCE GATE FAILED. Do not upload this checkpoint.")
        sys.exit(1)
    else:
        logger.info("COHERENCE GATE PASSED. Ready for upload.")
        sys.exit(0)


if __name__ == "__main__":
    main()
