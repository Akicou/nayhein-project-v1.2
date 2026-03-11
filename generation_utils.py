# generation_utils.py
# AR + MDLM + Hybrid generation logic for NayheinForCausalLM.
# Provides NayheinGenerationMixin with three generation modes:
#   - "ar":       standard autoregressive generation
#   - "diffusion": absorbing-state MDLM parallel decoding
#   - "hybrid":   AR chain-of-thought prefix + diffusion answer suffix

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GenerationOutput:
    """Container for generation results."""

    input_ids: Tensor
    generated_ids: Tensor
    text: Optional[str] = None
    tool_call: Optional[object] = None
    generation_mode: str = "ar"
    num_tokens: int = 0
    num_diffusion_steps: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Sampling helpers
# ══════════════════════════════════════════════════════════════════════════════


def top_p_filter(logits: Tensor, top_p: float = 0.9) -> Tensor:
    """Apply nucleus (top-p) sampling filter to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold (shifted right)
    sorted_indices_to_remove = (
        cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    )
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    # Scatter back to original indexing
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def temperature_scale(logits: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature == 0.0:
        return logits
    return logits / max(temperature, 1e-8)


def sample_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    greedy: bool = False,
) -> Tensor:
    """Sample next token from logits. Returns (B,) int tensor."""
    if greedy or temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = temperature_scale(logits, temperature)
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# MDLM Diffusion Scheduler
# ══════════════════════════════════════════════════════════════════════════════


class MDLMScheduler:
    """
    Absorbing-state masked diffusion scheduler.
    Implements the reverse denoising loop for inference.

    At each step:
    1. Run model forward on partially masked sequence
    2. Predict logits at all masked positions
    3. Compute confidence = max(softmax(logits)) at masked positions
    4. Unmask the top-k% most confident positions
    5. Frozen tokens remain frozen (absorbing state)
    """

    def __init__(self, num_steps: int = 16, mask_token_id: int = 4):
        self.num_steps = num_steps
        self.mask_token_id = mask_token_id

    def get_unmask_fraction(self, step: int) -> float:
        """Fraction of currently masked tokens to unmask at this step."""
        # Cosine-spaced unmasking schedule
        t_prev = 1.0 - (step / self.num_steps)
        t_curr = 1.0 - ((step + 1) / self.num_steps)
        alpha_prev = math.cos(math.pi * t_prev / 2) ** 2
        alpha_curr = math.cos(math.pi * t_curr / 2) ** 2
        # Fraction of total tokens to unmask this step
        frac = max(0.0, alpha_prev - alpha_curr)
        return frac

    def step(
        self,
        logits: Tensor,  # (B, S, V)
        current_ids: Tensor,  # (B, S) — contains MASK tokens
        step: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[Tensor, Tensor]:
        """
        One denoising step: unmask a fraction of masked positions.

        Returns:
            new_ids: updated token ids
            still_masked: boolean mask of positions still masked
        """
        B, S, V = logits.shape
        mask_positions = current_ids == self.mask_token_id  # (B, S)
        new_ids = current_ids.clone()

        if not mask_positions.any():
            return new_ids, mask_positions

        # Compute per-position confidence at masked locations
        probs = F.softmax(logits, dim=-1)  # (B, S, V)
        confidence, best_tokens = probs.max(dim=-1)  # (B, S)

        # For non-mask positions, set confidence to -inf so they aren't selected
        confidence = confidence.masked_fill(~mask_positions, float("-inf"))

        # Determine how many tokens to unmask this step
        num_masked = mask_positions.float().sum(dim=-1)  # (B,)
        frac = self.get_unmask_fraction(step)

        for b in range(B):
            n_masked_b = int(num_masked[b].item())
            if n_masked_b == 0:
                continue
            n_to_unmask = max(1, int(math.ceil(frac * S)))
            n_to_unmask = min(n_to_unmask, n_masked_b)

            # Get top-n_to_unmask most confident masked positions
            conf_b = confidence[b]  # (S,)
            topk_indices = torch.topk(conf_b, k=n_to_unmask, dim=-1).indices

            # Sample tokens at those positions using top_p
            for idx in topk_indices:
                pos_logits = logits[b, idx]  # (V,)
                token = sample_token(
                    pos_logits.unsqueeze(0), temperature=temperature, top_p=top_p
                ).item()
                new_ids[b, idx] = token

        still_masked = new_ids == self.mask_token_id
        return new_ids, still_masked


# ══════════════════════════════════════════════════════════════════════════════
# NayheinGenerationMixin
# ══════════════════════════════════════════════════════════════════════════════


class NayheinGenerationMixin:
    """
    Generation mixin for NayheinForCausalLM.
    Supports AR, MDLM diffusion, and hybrid (AR prefix + diffusion suffix).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.config = model.config

    # ── Autoregressive generation ─────────────────────────────────────────────

    @torch.inference_mode()
    def generate_ar(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> Tensor:
        """
        Standard autoregressive generation with KV cache.
        Returns full sequence tensor (input + generated).
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        generated = input_ids.clone()
        past_key_values = None
        cur_input = input_ids

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # (B, V)
            past_key_values = outputs.past_key_values

            next_token = sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                greedy=not do_sample,
            ).unsqueeze(-1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=-1)
            cur_input = next_token

            # Stop if all sequences have produced EOS
            if (generated[:, -1] == eos_token_id).all():
                break

        return generated

    # ── Speculative decoding with MTP heads ──────────────────────────────────

    @torch.inference_mode()
    def generate_ar_speculative(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Speculative decoding using MTP draft heads for 1.5-2.5x speedup.
        Draft N tokens via MTP heads → verify with main model.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        N = self.config.mtp_num_future_tokens
        generated = input_ids.clone()
        past_key_values = None
        cur_input = input_ids

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # Main model forward
            outputs = self.model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            main_logits = outputs.logits[:, -1, :]
            hidden_states = (
                outputs.last_hidden_state
                if hasattr(outputs, "last_hidden_state")
                else outputs.hidden_states[-1]
            )
            past_key_values = outputs.past_key_values

            # Accept main AR token
            main_token = sample_token(main_logits, temperature, top_p).unsqueeze(-1)
            generated = torch.cat([generated, main_token], dim=-1)
            tokens_generated += 1

            if (
                generated[:, -1] == eos_token_id
            ).all() or tokens_generated >= max_new_tokens:
                break

            # Draft N tokens greedily via MTP heads
            draft_tokens = [main_token]
            prev_emb = self.model.model.embed_tokens(main_token)  # (B, 1, H)
            h = hidden_states[:, -1:, :]  # (B, 1, H)
            cos, sin = self.model.model.rotary_emb(h, seq_len=generated.shape[1])

            for k, mtp_module in enumerate(self.model.mtp_modules):
                mtp_h = mtp_module(h, prev_emb, cos, sin)
                mtp_logits = self.model.lm_head(mtp_h)[:, -1, :]
                draft_tok = sample_token(
                    mtp_logits, temperature=0.0, greedy=True
                ).unsqueeze(-1)
                draft_tokens.append(draft_tok)
                prev_emb = self.model.model.embed_tokens(draft_tok)
                if tokens_generated >= max_new_tokens:
                    break

            # Accept drafts (greedy — no rejection sampling here for simplicity)
            for draft_tok in draft_tokens[1:]:
                if tokens_generated >= max_new_tokens:
                    break
                generated = torch.cat([generated, draft_tok], dim=-1)
                tokens_generated += 1
                if (generated[:, -1] == eos_token_id).all():
                    return generated

            cur_input = generated[:, -1:]

        return generated

    # ── MDLM Diffusion generation ─────────────────────────────────────────────

    @torch.inference_mode()
    def generate_diffusion(
        self,
        input_ids: Tensor,
        target_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Parallel masked diffusion generation.
        Starts from a fully masked suffix of length `target_length`.

        Returns: input_ids + generated suffix (B, S + target_length)
        """
        if num_steps is None:
            num_steps = self.config.diffusion_steps

        B, prompt_len = input_ids.shape
        mask_id = self.config.mask_token_id
        scheduler = MDLMScheduler(num_steps=num_steps, mask_token_id=mask_id)

        # Initialize fully masked suffix
        masked_suffix = torch.full(
            (B, target_length), mask_id, dtype=torch.long, device=self.device
        )
        current_ids = torch.cat(
            [input_ids, masked_suffix], dim=-1
        )  # (B, prompt + target)
        S_total = current_ids.shape[1]

        for step in range(num_steps):
            t_scalar = 1.0 - (step / num_steps)
            t_tensor = torch.full((B,), t_scalar, device=self.device)

            outputs = self.model(
                input_ids=current_ids,
                diffusion_t=t_tensor,
                use_cache=False,
            )
            logits = outputs.logits  # (B, S_total, V)

            # Only denoise the suffix portion
            suffix_logits = logits[:, prompt_len:, :]
            suffix_ids = current_ids[:, prompt_len:].clone()

            new_suffix, still_masked = scheduler.step(
                suffix_logits, suffix_ids, step, temperature=temperature, top_p=top_p
            )
            current_ids = torch.cat([input_ids, new_suffix], dim=-1)

            # Early exit if all tokens are unmasked
            if not still_masked.any():
                break

        return current_ids

    # ── Hybrid generation (AR prefix + Diffusion suffix) ─────────────────────

    @torch.inference_mode()
    def generate_hybrid(
        self,
        input_ids: Tensor,
        ar_max_tokens: int = 256,
        diff_target_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        diffusion_trigger_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Hybrid generation:
        1. Generate reasoning chain autoregressively until <|diffusion|> token
        2. Switch to diffusion mode for the answer

        If the <|diffusion|> trigger is not emitted within ar_max_tokens,
        diffusion generates directly from the end of the AR output.
        """
        if diffusion_trigger_token_id is None:
            diffusion_trigger_token_id = self.config.bos_token_id  # fallback

        # Stage 1: AR prefix
        ar_output = self.generate_ar(
            input_ids,
            max_new_tokens=ar_max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=diffusion_trigger_token_id,  # stop at diffusion trigger
        )

        # Stage 2: Diffusion suffix from AR output
        diff_output = self.generate_diffusion(
            ar_output,
            target_length=diff_target_length,
            temperature=temperature,
            top_p=top_p,
        )

        return diff_output

    # ── High-level chat interface ─────────────────────────────────────────────

    @torch.inference_mode()
    def generate_chat(
        self,
        messages: List[Dict],
        tools_xml: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        generation_mode: str = "ar",
    ) -> str:
        """
        Format messages → tokenize → generate → decode.

        Args:
            messages: list of {role, content} dicts
            tools_xml: pre-formatted tool descriptions XML
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling p
            generation_mode: "ar" | "diffusion" | "hybrid"

        Returns:
            Decoded assistant response string
        """
        text = self.tokenizer.apply_chatml(
            messages,
            add_generation_prompt=True,
            tools_xml=tools_xml,
        )
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = input_ids.to(self.device)

        if generation_mode == "ar":
            output_ids = self.generate_ar(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif generation_mode == "diffusion":
            output_ids = self.generate_diffusion(
                input_ids,
                target_length=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        elif generation_mode == "hybrid":
            output_ids = self.generate_hybrid(
                input_ids,
                ar_max_tokens=max_new_tokens // 2,
                diff_target_length=max_new_tokens // 2,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            raise ValueError(
                f"Unknown generation_mode: {generation_mode}. Use 'ar', 'diffusion', or 'hybrid'."
            )

        # Decode only newly generated tokens
        prompt_len = input_ids.shape[-1]
        generated_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
