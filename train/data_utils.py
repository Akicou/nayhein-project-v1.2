# train/data_utils.py
# Dataset loading, streaming interleaving, WildChat preprocessing,
# and sequence packing utilities for Nayhein-V1.2 training.

import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

# ── Dataset source definitions ─────────────────────────────────────────────────

PRETRAIN_DATASETS = {
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_column": "text",
        "weight": 0.65,
        "streaming": True,
    },
    "wikipedia": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.en",
        "split": "train",
        "text_column": "text",
        "weight": 0.15,
        "streaming": True,
    },
    "starcoderdata": {
        "path": "bigcode/starcoderdata",
        "split": "train",
        "text_column": "content",
        "weight": 0.10,
        "streaming": True,
        "languages": ["python", "javascript", "go"],
    },
    "proof-pile-2": {
        "path": "EleutherAI/proof-pile-2",
        "split": "train",
        "text_column": "text",
        "weight": 0.10,
        "streaming": True,
    },
}

SFT_DATASETS = {
    "wildchat": {
        "path": "allenai/WildChat-4.8M",
        "split": "train",
        "conversation_column": "conversations",
        "weight": 0.55,
        "max_samples": 500_000,
    },
    "llava-instruct": {
        "path": "liuhaotian/LLaVA-Instruct-150K",
        "split": "train",
        "conversation_column": "conversations",
        "weight": 0.15,
    },
    "ultrachat": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "conversation_column": "messages",
        "weight": 0.20,
    },
    "codefeedback": {
        "path": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "conversation_column": "conversations",
        "weight": 0.10,
    },
}

DPO_DATASETS = {
    "ultrafeedback": {
        "path": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "weight": 0.50,
    },
    "hh-rlhf": {
        "path": "Anthropic/hh-rlhf",
        "split": "train",
        "weight": 0.25,
    },
    "ultrainteract": {
        "path": "openbmb/UltraInteract_pair",
        "split": "train",
        "weight": 0.25,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# WildChat preprocessing
# ══════════════════════════════════════════════════════════════════════════════


def preprocess_wildchat_example(example: Dict) -> Optional[Dict]:
    """
    Filter and convert a single WildChat example.
    Removes toxic or non-English examples.
    Converts conversations to ChatML format.

    Args:
        example: raw WildChat row

    Returns:
        dict with 'messages' key or None if filtered out
    """
    # Filter: language
    if example.get("language") != "English":
        return None

    # Filter: toxicity
    conversations = example.get("conversations", [])
    for turn in conversations:
        if turn.get("toxic_flag", False):
            return None

    # Convert to {role, content} format
    messages = []
    for turn in conversations:
        role = turn.get("role", "").lower()
        content = turn.get("content", "")
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if content.strip():
            messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None

    return {"messages": messages}


def format_conversation_to_chatml(
    messages: List[Dict[str, str]],
    system_prompt: str = "You are Nayhein, a helpful, harmless, and honest AI assistant created by the Nayhein team (https://nayhein.com).",
) -> str:
    """
    Convert a list of message dicts to a full ChatML string.
    """
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Sequence Packing
# ══════════════════════════════════════════════════════════════════════════════


class PackedSequenceDataset(IterableDataset):
    """
    Packs tokenized sequences end-to-end into fixed-length chunks.
    No padding waste — fill each chunk to exactly `max_seq_len` tokens.
    Sequences are separated by EOS token.

    Usage:
        dataset = PackedSequenceDataset(token_stream, max_seq_len=4096, eos_token_id=1)
        loader = DataLoader(dataset, batch_size=16, num_workers=4)
    """

    def __init__(
        self,
        token_stream: Iterable[List[int]],
        max_seq_len: int = 4096,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
    ):
        self.token_stream = token_stream
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []

        for token_seq in self.token_stream:
            # Append EOS between documents
            buffer.extend(token_seq)
            if buffer and buffer[-1] != self.eos_token_id:
                buffer.append(self.eos_token_id)

            # Emit packed chunks
            while len(buffer) >= self.max_seq_len:
                chunk = buffer[: self.max_seq_len]
                buffer = buffer[self.max_seq_len :]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

        # Emit final partial chunk (padded)
        if buffer:
            pad_len = self.max_seq_len - len(buffer)
            chunk = buffer + [self.pad_token_id] * pad_len
            input_ids = torch.tensor(chunk, dtype=torch.long)
            labels = input_ids.clone()
            labels[len(buffer) :] = -100  # ignore padding in loss
            attention_mask = torch.tensor(
                [1] * len(buffer) + [0] * pad_len, dtype=torch.long
            )
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }


# ══════════════════════════════════════════════════════════════════════════════
# Streaming dataset builders
# ══════════════════════════════════════════════════════════════════════════════


def build_pretrain_token_stream(
    tokenizer,
    max_seq_len: int = 4096,
    seed: int = 42,
) -> Generator[List[int], None, None]:
    """
    Stream and interleave pretraining datasets, yielding tokenized sequences.
    Uses streaming mode for memory efficiency.
    """
    from datasets import load_dataset, interleave_datasets

    datasets_list = []
    probabilities = []

    for name, cfg in PRETRAIN_DATASETS.items():
        kwargs = {
            "path": cfg["path"],
            "split": cfg["split"],
            "streaming": cfg.get("streaming", True),
            "trust_remote_code": True,
        }
        if "name" in cfg:
            kwargs["name"] = cfg["name"]

        ds = load_dataset(**kwargs)

        # For StarCoder, filter by language
        if cfg.get("languages"):
            ds = ds.filter(lambda x: x.get("lang", "") in cfg["languages"])

        datasets_list.append((ds, cfg["text_column"]))
        probabilities.append(cfg["weight"])

    # Normalize probabilities
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    # Interleave
    raw_datasets = [d[0] for d in datasets_list]
    text_columns = [d[1] for d in datasets_list]

    interleaved = interleave_datasets(
        raw_datasets,
        probabilities=probabilities,
        stopping_strategy="all_exhausted",
        seed=seed,
    )

    # Tokenize and yield
    for i, example in enumerate(interleaved):
        # Find correct text column
        text = None
        for col in text_columns:
            if col in example and example[col]:
                text = example[col]
                break
        if text is None:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > 0:
            yield token_ids


def build_sft_dataset(
    tokenizer,
    max_seq_len: int = 4096,
    loss_mask_assistant_only: bool = True,
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    """
    Build the SFT dataset by loading, filtering, and tokenizing all SFT sources.
    Returns a list of {input_ids, labels, attention_mask} dicts.

    For SFT: labels are masked (-100) for all non-assistant tokens.
    """
    from datasets import load_dataset

    all_examples = []

    for ds_name, cfg in SFT_DATASETS.items():
        kwargs = {
            "path": cfg["path"],
            "split": cfg["split"],
            "trust_remote_code": True,
        }
        ds = load_dataset(**kwargs)

        conv_col = cfg["conversation_column"]
        max_samples = cfg.get("max_samples", None)

        count = 0
        for example in ds:
            if max_samples and count >= max_samples:
                break

            if ds_name == "wildchat":
                processed = preprocess_wildchat_example(example)
                if processed is None:
                    continue
                messages = processed["messages"]
            elif conv_col == "messages":
                messages = example.get("messages", [])
            else:
                # LLaVA/CodeFeedback format: list of {"from", "value"}
                raw_convs = example.get(conv_col, [])
                messages = []
                for turn in raw_convs:
                    role = (
                        "user"
                        if turn.get("from", "").lower() in ("human", "user")
                        else "assistant"
                    )
                    content = turn.get("value", turn.get("content", ""))
                    messages.append({"role": role, "content": content})

            if not messages:
                continue

            chatml_text = format_conversation_to_chatml(messages)
            token_ids = tokenizer.encode(chatml_text, add_special_tokens=False)

            if len(token_ids) > max_seq_len:
                token_ids = token_ids[:max_seq_len]

            input_ids = torch.tensor(token_ids, dtype=torch.long)
            labels = input_ids.clone()

            # Mask non-assistant tokens in loss
            if loss_mask_assistant_only:
                labels = _mask_non_assistant_labels(input_ids, labels, tokenizer)

            pad_len = max_seq_len - len(token_ids)
            if pad_len > 0:
                pad_tensor = torch.full(
                    (pad_len,), tokenizer.pad_token_id, dtype=torch.long
                )
                input_ids = torch.cat([input_ids, pad_tensor])
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=torch.long)]
                )
                attention_mask = torch.tensor(
                    [1] * len(token_ids) + [0] * pad_len, dtype=torch.long
                )
            else:
                attention_mask = torch.ones(max_seq_len, dtype=torch.long)

            all_examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            )
            count += 1

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(all_examples)
    return all_examples


def _mask_non_assistant_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Set labels to -100 for all tokens that are NOT part of an assistant response.
    Detects assistant turns by looking for <|im_start|>assistant\n patterns.
    """
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    ids = input_ids.tolist()
    masked = labels.clone()
    masked[:] = -100

    in_assistant = False
    i = 0
    while i < len(ids):
        if ids[i] == im_start_id:
            # Look ahead to check if role is "assistant"
            # The token sequence is: <|im_start|> assistant \n content <|im_end|>
            # Simplified heuristic: find next im_end, check role text
            window = tokenizer.decode(
                ids[i : min(i + 6, len(ids))], skip_special_tokens=False
            )
            if "assistant" in window:
                in_assistant = True
                # Skip past the role line to content
                j = i + 1
                while j < len(ids) and ids[j - 1] != tokenizer.convert_tokens_to_ids(
                    "\n"
                ):
                    j += 1
                i = j
                continue
            else:
                in_assistant = False
        elif ids[i] == im_end_id:
            in_assistant = False

        if in_assistant:
            masked[i] = labels[i]
        i += 1

    return masked


def build_dpo_dataset(seed: int = 42) -> List[Dict]:
    """
    Load and merge DPO preference datasets.
    Returns list of {prompt, chosen, rejected} dicts.
    """
    from datasets import load_dataset, concatenate_datasets

    all_pairs = []

    for ds_name, cfg in DPO_DATASETS.items():
        try:
            ds = load_dataset(cfg["path"], split=cfg["split"], trust_remote_code=True)
        except Exception as e:
            print(f"Warning: could not load DPO dataset {ds_name}: {e}")
            continue

        for example in ds:
            if ds_name == "ultrafeedback":
                prompt = example.get("prompt", "")
                chosen = example.get("chosen", [])
                rejected = example.get("rejected", [])
                # chosen/rejected are lists of {role, content}
                chosen_text = chosen[-1]["content"] if chosen else ""
                rejected_text = rejected[-1]["content"] if rejected else ""
            elif ds_name == "hh-rlhf":
                prompt = ""
                chosen_text = example.get("chosen", "")
                rejected_text = example.get("rejected", "")
            elif ds_name == "ultrainteract":
                prompt = example.get("instruction", "")
                chosen_text = example.get("chosen_response", "")
                rejected_text = example.get("rejected_response", "")
            else:
                continue

            if chosen_text and rejected_text:
                all_pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                    }
                )

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs
