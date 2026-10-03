"""
Shared utilities for graph nodes.
"""


def extract_text(content) -> str:
    """
    Extract plain text from a LangChain response content.

    Thinking models (gemma-4-31b-it) return content as a list of blocks:
        [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]

    Non-thinking models return content as a plain string.

    This function handles both cases and returns just the text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block and block.get("type") != "thinking":
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    return str(content)
