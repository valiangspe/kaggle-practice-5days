# pip install "langchain>=1.0.0" langchain-openai langchain-community

import io
import json
import textwrap
import contextlib
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ================== LLM FACTORY (Ollama via OpenAI-compatible API) ==================


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",  # dummy; required by client
        temperature=0.1,
    )


# ================== FUNCTION TOOLS: FEES & EXCHANGE RATES ==================


@tool("get_fee_for_payment_method")
def get_fee_for_payment_method_tool(payment_method: str) -> Dict[str, Any]:
    """
    Return the transaction fee percentage for a given payment method.
    Response schema:
      { "status": "ok" | "error",
        "payment_method": str,
        "fee_percent": float | None,
        "message": str }
    """

    pm = payment_method.strip().lower()

    if "bank" in pm:
        fee_percent = 0.01  # 1%
    elif "credit" in pm:
        fee_percent = 0.025  # 2.5%
    elif "wallet" in pm or "ewallet" in pm or "e-wallet" in pm:
        fee_percent = 0.015  # 1.5%
    else:
        return {
            "status": "error",
            "payment_method": payment_method,
            "fee_percent": None,
            "message": f"Unsupported payment method: {payment_method}",
        }

    return {
        "status": "ok",
        "payment_method": payment_method,
        "fee_percent": fee_percent,
        "message": f"Fee for {payment_method} is {fee_percent * 100:.2f}%.",
    }


@tool("get_exchange_rate")
def get_exchange_rate_tool(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """
    Return the FX rate for a given currency pair.
    Response schema:
      { "status": "ok" | "error",
        "from_currency": str,
        "to_currency": str,
        "rate": float | None,
        "message": str }

    NOTE: In production, replace this with a real FX API call.
    """

    fc = from_currency.strip().upper()
    tc = to_currency.strip().upper()

    # Demo-only mock values
    if fc == "USD" and tc == "IDR":
        rate = 15500.0
    elif fc == "IDR" and tc == "USD":
        rate = 1.0 / 15500.0
    else:
        return {
            "status": "error",
            "from_currency": fc,
            "to_currency": tc,
            "rate": None,
            "message": f"No mock rate available for {fc}/{tc}.",
        }

    return {
        "status": "ok",
        "from_currency": fc,
        "to_currency": tc,
        "rate": rate,
        "message": f"Mock FX rate for {fc}/{tc} is {rate}.",
    }


# ================== CALCULATION SUB-AGENT (CODE GENERATOR) ==================

CALC_SYSTEM_PROMPT = """
You are a specialized calculator that ONLY responds with Python code.
You are forbidden from providing any text, explanations, or conversational responses.

Your task: Take a description of a calculation and translate it into Python code.

RULES:
1. Your output MUST be ONLY a Python code block (using ```python ... ```).
2. Do NOT write any text before or after the code block.
3. The Python code MUST calculate the result using the given variables.
4. The Python code MUST print the final result as a JSON string to stdout.
5. You are PROHIBITED from performing the calculation yourself; only generate code.
"""

calculation_agent = create_agent(
    model=make_llm(),
    tools=[],  # this agent only generates code; execution is done by the tool wrapper below
    system_prompt=textwrap.dedent(CALC_SYSTEM_PROMPT),
)

print("✅ calculation_agent created.")


def _extract_python_code(block: str) -> str:
    """Extract Python code from a markdown code block if present."""
    text = block.strip()
    if "```" not in text:
        return text  # assume it's just code

    # Find first fenced block
    parts = text.split("```")
    # parts[0] = before first fence, parts[1] = maybe "python\ncode..."
    for part in parts[1:]:
        # remove leading language label if present
        if part.lstrip().startswith("python"):
            code = part.lstrip()[len("python") :]
        else:
            code = part
        return code.strip()
    return text


@tool("calculation_agent")
def calculation_agent_tool(
    amount: float,
    fee_percent: float,
    exchange_rate: float,
) -> Dict[str, Any]:
    """
    Delegates all arithmetic to a code-generation sub-agent.

    Inputs:
      - amount: original amount in source currency
      - fee_percent: fee as a decimal (e.g., 0.01 for 1%)
      - exchange_rate: FX rate from source to target

    Returns:
      {
        "status": "ok" | "error",
        "amount": float,
        "fee_percent": float,
        "fee_amount": float,
        "net_amount": float,
        "exchange_rate": float,
        "final_amount": float,
        "raw_stdout": str
      }
    """

    user_instruction = f"""
Given:
- amount = {amount}
- fee_percent = {fee_percent}
- exchange_rate = {exchange_rate}

Write Python code that:
1. Computes:
   fee_amount = amount * fee_percent
   net_amount = amount - fee_amount
   final_amount = net_amount * exchange_rate
2. Prints a JSON string to stdout using print(), with keys:
   "amount", "fee_percent", "fee_amount", "net_amount",
   "exchange_rate", "final_amount".
"""

    state = calculation_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": textwrap.dedent(user_instruction).strip(),
                }
            ]
        }
    )

    content = state["messages"][-1].content
    code = _extract_python_code(content)

    # Execute the generated Python code and capture stdout
    buf = io.StringIO()
    env: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, env, env)  # noqa: S102 - trusted only for demo
    except Exception as e:
        return {
            "status": "error",
            "amount": amount,
            "fee_percent": fee_percent,
            "exchange_rate": exchange_rate,
            "fee_amount": None,
            "net_amount": None,
            "final_amount": None,
            "raw_stdout": buf.getvalue(),
            "message": f"Error executing generated code: {e}",
        }

    stdout_val = buf.getvalue().strip()
    try:
        data = json.loads(stdout_val)
    except Exception as e:
        return {
            "status": "error",
            "amount": amount,
            "fee_percent": fee_percent,
            "exchange_rate": exchange_rate,
            "fee_amount": None,
            "net_amount": None,
            "final_amount": None,
            "raw_stdout": stdout_val,
            "message": f"Could not parse JSON from code output: {e}",
        }

    data["status"] = "ok"
    data["raw_stdout"] = stdout_val
    return data


print("✅ calculation_agent_tool created (delegates to code-gen sub-agent).")


# ================== ENHANCED CURRENCY AGENT (ORCHESTRATOR) ==================

ENHANCED_SYSTEM_PROMPT = """
You are a smart currency conversion assistant. You must strictly follow these steps and use the available tools.

Available tools:
- get_fee_for_payment_method(payment_method: str) -> {status, fee_percent, ...}
- get_exchange_rate(from_currency: str, to_currency: str) -> {status, rate, ...}
- calculation_agent(amount: float, fee_percent: float, exchange_rate: float)
    -> {status, amount, fee_percent, fee_amount, net_amount, exchange_rate, final_amount, ...}

For ANY currency conversion request:

1. Get Transaction Fee:
   - Call get_fee_for_payment_method() to determine the transaction fee.
2. Get Exchange Rate:
   - Call get_exchange_rate() to get the currency conversion rate.
3. Error Check:
   - After each tool call, you MUST check the "status" field in the response.
   - If status == "error", STOP and clearly explain the issue to the user.
4. Calculate Final Amount (CRITICAL):
   - You are STRICTLY PROHIBITED from performing any arithmetic calculations yourself.
   - You MUST call the calculation_agent tool to compute the final converted amount.
   - Pass it:
       * the original amount,
       * the fee_percent from step 1,
       * the FX rate from step 2.
5. Provide Detailed Breakdown:
   In your final answer to the user, you MUST:
   - State the final converted amount (with currency).
   - Explain how the result was calculated, including:
       * The fee percentage and the fee amount in the original currency.
       * The amount remaining after deducting the fee.
       * The exchange rate applied.
   - Base your explanation ONLY on the data returned from the tools; do NOT recompute.

If any tool returns status="error", explain the problem and do NOT proceed to the next steps.
"""

enhanced_currency_agent = create_agent(
    model=make_llm(),
    tools=[
        get_fee_for_payment_method_tool,
        get_exchange_rate_tool,
        calculation_agent_tool,
    ],
    system_prompt=textwrap.dedent(ENHANCED_SYSTEM_PROMPT),
)

print("✅ enhanced_currency_agent created")
print("🎯 New capability: Delegates calculations to specialist code-gen agent")
print("🔧 Tool types used:")
print("  • Function tools (fees, rates)")
print("  • Sub-agent tool (calculation specialist + Python execution)")


# ================== HELPER TO QUERY THE ENHANCED AGENT ==================


def ask_enhanced_currency_agent(query: str) -> str:
    state = enhanced_currency_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={
            "verbose": True,  # prints tool calls and reasoning
            "callbacks": [StreamingStdOutCallbackHandler()],
        },
    )
    return state["messages"][-1].content


if __name__ == "__main__":
    q = "Convert 1,250 USD to IDR using a Bank Transfer. Show me the precise calculation."
    print("\n=== Question: ===")
    print(q)
    print()
    answer = ask_enhanced_currency_agent(q)
    print("\n=== FINAL ANSWER TO USER ===")
    print(answer)
