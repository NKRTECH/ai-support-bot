"""
ReAct agent loop for multi-step tool calling.

Implements the Think → Act → Observe cycle:
  1. Send the conversation (messages + tool declarations) to the LLM
  2. If the LLM returns a function_call → execute it (with guardrails)
  3. Append the tool result to the conversation
  4. Loop back to step 1
  5. When the LLM returns a text response → done

This replaces the single-step _handle_action approach with a proper
agent loop that can chain multiple tool calls in sequence.
"""

from google.genai import types
from google.genai.errors import ServerError, ClientError
from logger import get_logger
from agent.guardrails import needs_approval, request_approval

log = get_logger(__name__)

MAX_STEPS = 10


def run_agent(
    client,
    model: str,
    system_prompt: str,
    user_message: str,
    tool_declarations: types.Tool,
    tool_registry: dict,
) -> str:
    """
    Run a multi-step ReAct agent loop.

    The agent can call tools multiple times in sequence, building up
    context until it has enough information to answer the customer.

    Args:
        client: The Gemini API client.
        model: Model name (e.g., "gemma-4-31b-it").
        system_prompt: System instructions for the agent's persona.
        user_message: The customer's message.
        tool_declarations: Gemini Tool object with function declarations.
        tool_registry: Dict mapping function names to callable Python functions.

    Returns:
        The agent's final text response to the customer.
    """
    # Build the initial conversation
    messages = [
        types.Content(
            role="user",
            parts=[types.Part(text=system_prompt + "\n\nCustomer message: " + user_message)],
        )
    ]

    for step in range(MAX_STEPS):
        log.info("Agent step %d/%d", step + 1, MAX_STEPS)

        try:
            response = client.models.generate_content(
                model=model,
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[tool_declarations],
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            )
        except ServerError:
            log.error("Agent loop hit server error at step %d", step + 1, exc_info=True)
            return ("I'm having trouble connecting to our systems right now "
                    "(high demand). Please try again in a moment.")
        except ClientError as e:
            log.error("Agent loop hit client error at step %d: %s", step + 1, e, exc_info=True)
            return ("Something went wrong while looking that up. "
                    "Please try again or contact us at 1800-123-4567.")

        # Find a function_call in the response parts
        fc_part = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'function_call', None) and part.function_call.name:
                    fc_part = part
                    break

        if not fc_part:
            # No tool call — the LLM is done, extract the text response
            return _extract_text(response)

        # We have a tool call — process it
        fc = fc_part.function_call
        fn_name = fc.name
        fn_args = dict(fc.args) if fc.args else {}

        log.info("Agent tool call [step %d]: %s(%s)", step + 1, fn_name, fn_args)
        print(f"\033[90m[calling: {fn_name}({fn_args})]\033[0m")

        # Check guardrails before executing
        if needs_approval(fn_name, fn_args):
            approved = request_approval(fn_name, fn_args)
            if not approved:
                # Denied — tell the LLM the action was rejected
                result = (f"Action DENIED by supervisor. The {fn_name} request was "
                          f"not approved. Inform the customer politely and offer "
                          f"alternatives (e.g., contacting support directly).")
                log.info("Tool call denied by operator: %s", fn_name)
                print("\033[90m[action denied by operator]\033[0m")
            else:
                result = _execute_tool(fn_name, fn_args, tool_registry)
                print("\033[90m[tool result received]\033[0m")
        else:
            result = _execute_tool(fn_name, fn_args, tool_registry)
            print("\033[90m[tool result received]\033[0m")

        # Append the model's function call + our result to the conversation
        messages.append(
            types.Content(
                role="model",
                parts=[types.Part(function_call=fc)],
            )
        )
        messages.append(
            types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(
                    name=fn_name,
                    response={"result": result},
                ))],
            )
        )

    # Exhausted max steps without a final text response
    log.warning("Agent exhausted %d steps without completing", MAX_STEPS)
    return ("I went through several steps but wasn't able to fully resolve this. "
            "Please contact our support team at 1800-123-4567 for direct help.")


def _execute_tool(fn_name: str, fn_args: dict, tool_registry: dict) -> str:
    """Execute a tool function from the registry and return its result."""
    if fn_name not in tool_registry:
        log.warning("Unknown tool requested: %s", fn_name)
        return f"Error: Unknown tool '{fn_name}'. Available tools: {list(tool_registry.keys())}"

    try:
        result = tool_registry[fn_name](**fn_args)
        log.info("Tool result for %s: %s", fn_name, result[:200])
        return result
    except Exception as e:
        log.error("Tool execution failed: %s(%s) → %s", fn_name, fn_args, e, exc_info=True)
        return f"Error executing {fn_name}: {str(e)}"


def _extract_text(response) -> str:
    """Extract non-thought text from a Gemini response."""
    text = ""
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if not getattr(part, 'thought', False) and getattr(part, 'text', None):
                text += part.text

    if not text:
        log.warning("Empty text response from agent. Parts: %s",
                     [type(p).__name__ for p in
                      (response.candidates[0].content.parts
                       if response.candidates and response.candidates[0].content.parts
                       else [])])
        return ("I looked into that but couldn't put together a response. "
                "Please try rephrasing your question.")

    log.info("Agent final response: %d chars", len(text))
    return text
