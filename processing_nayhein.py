# processing_nayhein.py
# NayheinProcessor — unified text + image processor for NayheinForCausalLM.
# Handles image preprocessing (resize, normalize) and text tokenization.
# Implements the AutoProcessor interface for HuggingFace integration.

from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import ProcessorMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

# Image normalization constants (SigLIP-style: mean/std ≈ 0.5)
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def _resize_and_normalize(
    image: Image.Image,
    image_size: int,
) -> torch.Tensor:
    """
    Resize PIL image to (image_size, image_size) and normalize to [-1, 1].
    Returns float32 tensor of shape (3, H, W).
    """
    import torchvision.transforms.functional as TF

    image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.BICUBIC)
    tensor = TF.to_tensor(image)  # (3, H, W) in [0, 1]
    # Normalize: (x - mean) / std
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


class NayheinProcessor(ProcessorMixin):
    """
    NayheinProcessor wraps NayheinTokenizer + image preprocessing.

    Usage:
        processor = NayheinProcessor.from_pretrained(model_id, trust_remote_code=True)
        inputs = processor(messages, images=image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=256)

    Messages format follows OpenAI multi-modal API:
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is in this image?"}
        ]}]
    """

    attributes = ["tokenizer"]
    tokenizer_class = "NayheinTokenizer"

    def __init__(self, tokenizer=None, image_size: int = 256, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.image_size = image_size

    def __call__(
        self,
        messages: Optional[Union[List[Dict], str]] = None,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        return_tensors: str = "pt",
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Process messages (and optional images) into model inputs.

        Args:
            messages: list of message dicts OR plain text string
            images: PIL Image(s) to encode. Must align with <image> placements.
            text: plain text override (bypasses ChatML template)
            return_tensors: "pt" (default)
            padding: whether to pad to max length in batch
            truncation: truncate to max_length
            max_length: max token sequence length
            add_generation_prompt: append <|im_start|>assistant\\n

        Returns:
            BatchEncoding with input_ids, attention_mask, and optionally pixel_values
        """
        # ── Build text string ─────────────────────────────────────────────────
        if text is not None:
            text_input = text
        elif messages is not None:
            text_input = self._process_messages(messages, add_generation_prompt)
        else:
            raise ValueError("Either `messages` or `text` must be provided.")

        # ── Tokenize ──────────────────────────────────────────────────────────
        encoding = self.tokenizer(
            text_input,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
        )

        # ── Process images ────────────────────────────────────────────────────
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]
            pixel_values = torch.stack(
                [_resize_and_normalize(img, self.image_size) for img in images]
            )  # (N_images, 3, H, W)
            encoding["pixel_values"] = pixel_values

        return encoding

    def _process_messages(
        self,
        messages: List[Dict],
        add_generation_prompt: bool,
    ) -> str:
        """
        Convert multi-modal message list to a flat string with ChatML formatting.
        Image placeholders become <|vision_start|><|vision_end|> tokens.
        """
        parts = []

        # Check for system message
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system:
            parts.append(
                "<|im_start|>system\n"
                "You are Nayhein, a helpful, harmless, and honest AI assistant "
                "created by the Nayhein team (https://nayhein.com)."
                "<|im_end|>"
            )

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Multi-modal content (list of dicts)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "image":
                        text_parts.append("<|vision_start|><|vision_end|>")
                    elif item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content_str = "".join(text_parts)
            else:
                content_str = str(content)

            parts.append(f"<|im_start|>{role}\n{content_str}<|im_end|>")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + ["pixel_values"]))

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)


# ── Auto-register ─────────────────────────────────────────────────────────────
from transformers import AutoProcessor

AutoProcessor.register("nayhein", NayheinProcessor)
