# tokenization_nayhein.py
# NayheinTokenizer — custom BPE tokenizer wrapper for the Nayhein-V1.2 model family.
# Trained with HuggingFace `tokenizers` library on FineWeb-Edu + Wikipedia + StarCoder.
# Registered as a PreTrainedTokenizerFast for full Transformers integration.

import os
import json
from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizerFast
from transformers.utils import logging

logger = logging.get_logger(__name__)

# ── Special token definitions ─────────────────────────────────────────────────
SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|eos|>": 1,
    "<|bos|>": 2,
    "<|unk|>": 3,
    "<|mask|>": 4,  # MDLM absorbing mask token
    "<|im_start|>": 5,  # ChatML
    "<|im_end|>": 6,  # ChatML
    "<|vision_start|>": 7,
    "<|vision_end|>": 8,
    "<tool_call>": 9,
    "</tool_call>": 10,
    "<tool_result>": 11,
    "</tool_result>": 12,
    "<|diffusion|>": 13,  # mode switch token for hybrid generation
}

# Default ChatML chat template
CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


class NayheinTokenizer(PreTrainedTokenizerFast):
    """
    NayheinTokenizer — a BPE tokenizer with 65,536 vocab size trained on
    multilingual + code corpora. Supports ChatML chat template, vision tokens,
    tool call tokens, and the MDLM mask token.

    Special tokens:
        <|pad|>          (id=0)  — padding
        <|eos|>          (id=1)  — end of sequence
        <|bos|>          (id=2)  — beginning of sequence
        <|unk|>          (id=3)  — unknown
        <|mask|>         (id=4)  — MDLM absorbing mask
        <|im_start|>     (id=5)  — ChatML turn start
        <|im_end|>       (id=6)  — ChatML turn end
        <|vision_start|> (id=7)  — start of visual token block
        <|vision_end|>   (id=8)  — end of visual token block
        <tool_call>      (id=9)  — tool call open tag
        </tool_call>     (id=10) — tool call close tag
        <tool_result>    (id=11) — tool result open tag
        </tool_result>   (id=12) — tool result close tag
        <|diffusion|>    (id=13) — hybrid mode switch
    """

    vocab_files_names = {
        "tokenizer_file": "tokenizer.json",
    }

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<|bos|>",
        eos_token: str = "<|eos|>",
        unk_token: str = "<|unk|>",
        pad_token: str = "<|pad|>",
        mask_token: str = "<|mask|>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            chat_template=chat_template or CHATML_CHAT_TEMPLATE,
            **kwargs,
        )
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    @property
    def vocab_size(self) -> int:
        return 65536

    # ── Special token helpers ─────────────────────────────────────────────────

    @property
    def im_start_token_id(self) -> int:
        return SPECIAL_TOKENS["<|im_start|>"]

    @property
    def im_end_token_id(self) -> int:
        return SPECIAL_TOKENS["<|im_end|>"]

    @property
    def vision_start_token_id(self) -> int:
        return SPECIAL_TOKENS["<|vision_start|>"]

    @property
    def vision_end_token_id(self) -> int:
        return SPECIAL_TOKENS["<|vision_end|>"]

    @property
    def tool_call_start_token_id(self) -> int:
        return SPECIAL_TOKENS["<tool_call>"]

    @property
    def tool_call_end_token_id(self) -> int:
        return SPECIAL_TOKENS["</tool_call>"]

    @property
    def diffusion_token_id(self) -> int:
        return SPECIAL_TOKENS["<|diffusion|>"]

    # ── Build inputs with special tokens ─────────────────────────────────────

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + token_ids_1 + eos

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + ([0] * len(token_ids_0)) + eos
        return bos + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + eos

    # ── Chat formatting helpers ───────────────────────────────────────────────

    def apply_chatml(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "You are Nayhein, a helpful, harmless, and honest AI assistant created by the Nayhein team (https://nayhein.com).",
        add_generation_prompt: bool = True,
        tools_xml: Optional[str] = None,
    ) -> str:
        """
        Format a message list into ChatML string.

        Args:
            messages: list of {role, content} dicts
            system_prompt: default system prompt if not provided in messages
            add_generation_prompt: append assistant turn start
            tools_xml: optional XML tool descriptions to inject before system turn
        """
        parts = []

        # Prepend default system if not already present
        has_system = any(m["role"] == "system" for m in messages)
        if not has_system:
            sys_content = system_prompt
            if tools_xml:
                sys_content = f"{tools_xml}\n\n{sys_content}"
            parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system" and tools_xml:
                content = f"{tools_xml}\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def encode_chatml(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> List[int]:
        """Encode messages as ChatML token ids."""
        text = self.apply_chatml(messages, **kwargs)
        return self.encode(text, add_special_tokens=False)

    # ── Serialization ─────────────────────────────────────────────────────────

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()
        tokenizer_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json",
        )
        self.backend_tokenizer.save(tokenizer_file)
        return (tokenizer_file,)
