# tool_calling.py
# NayheinToolCallingMixin — OpenAI-compatible tool call injection + parsing.
# Tools are injected as XML into the system prompt.
# Model emits: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ToolCallOutput:
    """Parsed result of a model tool call."""

    tool_name: str
    arguments: Dict[str, Any]
    raw_response: str
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class ToolSchema:
    """OpenAI-compatible tool schema container."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict) -> "ToolSchema":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            parameters=d.get("parameters", {}),
        )

    def to_xml(self) -> str:
        """Render tool schema as XML for system prompt injection."""
        params_json = json.dumps(self.parameters, indent=2)
        return (
            f"<tool>\n"
            f"  <name>{self.name}</name>\n"
            f"  <description>{self.description}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# NayheinToolCallingMixin
# ══════════════════════════════════════════════════════════════════════════════


class NayheinToolCallingMixin:
    """
    Mixin that provides tool calling capabilities to NayheinForCausalLM.

    Tool schemas are accepted in OpenAI-compatible JSON Schema format.
    Tools are injected into the system prompt as structured XML.
    The model emits tool calls in the format:
        <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """

    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>(.*?)</tool_call>",
        re.DOTALL,
    )

    def format_tools(self, tools: List[Dict]) -> str:
        """
        Convert a list of OpenAI-compatible tool dicts to XML system prompt block.

        Args:
            tools: list of tool schema dicts with keys: name, description, parameters

        Returns:
            XML string to prepend to the system prompt
        """
        if not tools:
            return ""

        schemas = [ToolSchema.from_dict(t) for t in tools]
        tool_xml_blocks = [s.to_xml() for s in schemas]

        return (
            "<tools>\n" + "\n".join(tool_xml_blocks) + "\n</tools>\n\n"
            "You have access to the tools listed above. "
            "When a tool is needed, emit EXACTLY one tool call in the format:\n"
            '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>\n'
            "Do not include any additional explanation inside the tool_call tags. "
            "Wait for the tool result before continuing."
        )

    def parse_tool_call(self, response: str) -> Optional[ToolCallOutput]:
        """
        Parse a model response and extract a tool call if present.

        Args:
            response: raw decoded model output

        Returns:
            ToolCallOutput if a tool call is found, else None
        """
        match = self.TOOL_CALL_PATTERN.search(response)
        if not match:
            return None

        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            if not name:
                return ToolCallOutput(
                    tool_name="",
                    arguments={},
                    raw_response=response,
                    is_valid=False,
                    error="Missing 'name' in tool_call JSON.",
                )
            return ToolCallOutput(
                tool_name=name,
                arguments=arguments,
                raw_response=response,
                is_valid=True,
            )
        except json.JSONDecodeError as e:
            return ToolCallOutput(
                tool_name="",
                arguments={},
                raw_response=response,
                is_valid=False,
                error=f"JSON decode error: {e}",
            )

    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """
        Format a tool execution result for injection back into the conversation.

        Args:
            tool_name: name of the tool that was called
            result: any JSON-serializable result

        Returns:
            Formatted string to add as a user/tool-result turn
        """
        result_json = json.dumps(result) if not isinstance(result, str) else result
        return f"<tool_result>\n{result_json}\n</tool_result>"

    def build_tool_result_message(self, tool_name: str, result: Any) -> Dict[str, str]:
        """Return a message dict for the tool result turn."""
        return {
            "role": "tool",
            "content": self.format_tool_result(tool_name, result),
        }

    def constrained_generate(
        self,
        model,
        tokenizer,
        messages: List[Dict],
        tools: List[Dict],
        max_new_tokens: int = 256,
        enforce_json: bool = False,
        **generate_kwargs,
    ) -> ToolCallOutput:
        """
        Generate a tool call response. Optionally uses `outlines` for
        grammar-constrained JSON decoding when enforce_json=True.

        Args:
            model: NayheinForCausalLM instance
            tokenizer: NayheinTokenizer instance
            messages: conversation history
            tools: list of tool schema dicts
            max_new_tokens: max tokens to generate
            enforce_json: use outlines grammar-constrained decoding

        Returns:
            ToolCallOutput with parsed tool call
        """
        tools_xml = self.format_tools(tools)

        if enforce_json:
            try:
                import outlines
                import outlines.models
                import outlines.generate

                # Build JSON schema for structured decoding
                # We expect: {"name": str, "arguments": dict}
                tool_names = [t["name"] for t in tools]
                schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": tool_names},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                }

                text = tokenizer.apply_chatml(
                    messages, tools_xml=tools_xml, add_generation_prompt=True
                )
                input_ids = tokenizer.encode(
                    text, return_tensors="pt", add_special_tokens=False
                )
                input_ids = input_ids.to(next(model.parameters()).device)

                # Outlines grammar-constrained generation
                outlines_model = outlines.models.Transformers(model, tokenizer)
                generator = outlines.generate.json(outlines_model, schema)
                result = generator(text)
                parsed = json.loads(result)

                return ToolCallOutput(
                    tool_name=parsed.get("name", ""),
                    arguments=parsed.get("arguments", {}),
                    raw_response=result,
                    is_valid=True,
                )
            except ImportError:
                pass  # Fall through to standard generation

        # Standard generation path
        from generation_utils import NayheinGenerationMixin

        gen = NayheinGenerationMixin(model, tokenizer)
        response = gen.generate_chat(
            messages,
            tools_xml=tools_xml,
            max_new_tokens=max_new_tokens,
            generation_mode="ar",
            **generate_kwargs,
        )

        parsed = self.parse_tool_call(response)
        if parsed is None:
            return ToolCallOutput(
                tool_name="",
                arguments={},
                raw_response=response,
                is_valid=False,
                error="No tool_call tag found in model response.",
            )
        return parsed
