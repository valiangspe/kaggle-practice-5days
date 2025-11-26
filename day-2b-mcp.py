# shipping_client.py
import asyncio
import json
from typing import List, Dict, Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


SHIPPING_SYSTEM_PROMPT = """
You are a shipping coordinator assistant.

You have access to these tools from the Shipping MCP server:

1) check_order_needs_approval(num_containers: int)
   - Tells you whether an order needs human approval.
   - Returns: {"needs_approval": bool, "threshold": int, ...}

2) place_shipping_order(
       num_containers: int,
       destination: str,
       tool_confirmation: Optional[bool]
   )
   - If num_containers <= threshold → may auto-approve.
   - For large orders:
       * First call with tool_confirmation=None → status 'pending' and needs_confirmation=True.
       * Second call with tool_confirmation=True → status 'approved'.
       * Second call with tool_confirmation=False → status 'rejected'.

When users request to ship containers:

1. Parse the number of containers and destination from the user message.
2. First, call check_order_needs_approval to understand the policy.
3. If needs_approval is False:
     - Call place_shipping_order once (tool_confirmation=None is fine).
     - Return the final status and a concise summary.
4. If needs_approval is True:
     - Call place_shipping_order with tool_confirmation=None.
     - It will return status 'pending' and a hint.
     - Tell the user approval is required and wait for them to answer 'y' or 'n' in a follow-up message.
5. On the follow-up message:
     - If the user approves, call place_shipping_order again with tool_confirmation=True.
     - If the user rejects, call place_shipping_order again with tool_confirmation=False.
6. In the final answer, summarize:
     - Order status (approved/rejected)
     - Order ID (if available)
     - Number of containers and destination
7. Keep responses concise but informative.
"""


def make_llm() -> ChatOpenAI:
    # Ollama, OpenAI-compatible
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",  # dummy string
        temperature=0.2,
    )


# ─────────────────────────────────────────────
# Helpers to extract JSON + needs_approval from agent state
# ─────────────────────────────────────────────
import json
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────
# Helpers to extract JSON + needs_approval from agent state
# ─────────────────────────────────────────────

def _extract_json_objects_from_content(content: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of JSON dicts from a message's content."""
    results: List[Dict[str, Any]] = []

    # Case 1: plain string, maybe JSON
    if isinstance(content, str):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                results.append(obj)
        except Exception:
            pass
        return results

    # Case 2: list of parts (common for tool responses in LC 1.x MCP)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                # MCP JSON part pattern: {"type": "json", "json": {...}}
                if part.get("type") == "json" and isinstance(part.get("json"), dict):
                    results.append(part["json"])
                # Sometimes JSON is inside "text"
                elif "text" in part and isinstance(part["text"], str):
                    try:
                        obj = json.loads(part["text"])
                        if isinstance(obj, dict):
                            results.append(obj)
                    except Exception:
                        pass
    return results


def extract_needs_approval_from_state(state: Dict[str, Any]) -> Optional[bool]:
    """
    Look through the agent's messages and try to find the most recent
    tool JSON that contains a 'needs_approval' key.

    `state["messages"]` is a list of LangChain message objects
    (HumanMessage, AIMessage, ToolMessage, etc.), not dicts.
    """
    msgs = state.get("messages", [])
    for msg in reversed(msgs):
        # LangChain messages expose `.content`, not dict-style .get()
        content = getattr(msg, "content", None)
        for obj in _extract_json_objects_from_content(content):
            if "needs_approval" in obj:
                return bool(obj["needs_approval"])
    return None


# ─────────────────────────────────────────────
# Build MCP client & shipping agent
# ─────────────────────────────────────────────

async def build_shipping_agent():
    client = MultiServerMCPClient(
        {
            "shipping": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()  # includes check_order_needs_approval + place_shipping_order

    llm = make_llm()
    shipping_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SHIPPING_SYSTEM_PROMPT,
    )

    return client, shipping_agent


# ─────────────────────────────────────────────
# Workflow: now only asks y/n when needs_approval == True
# ─────────────────────────────────────────────

async def run_shipping_workflow(agent, user_message: str) -> None:
    """
    One full shipping workflow:

    - First turn: user asks to ship; agent calls tools and may auto-approve.
    - We inspect the agent state to see if the latest tool JSON says
      "needs_approval": true.
    - Only if needs_approval == True we ask for y/n on the keyboard.
    """

    print(f"\n🧵 Workflow start: {user_message}")
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]

    # Step 1 – initial request
    first_state = await agent.ainvoke(
        {"messages": messages},
        config={
            "verbose": True,
            "callbacks": [StreamingStdOutCallbackHandler()],
        },
    )
    first_reply = first_state["messages"][-1].content
    print("\n🤖 Agent (step 1):")
    print(first_reply)

    
    # Check if the tool output says this actually needs approval
    needs_approval = extract_needs_approval_from_state(first_state)

    if needs_approval is False:
        # No approval needed → stop here
        print("\nℹ️ Tool reports needs_approval = False → no human decision needed.")
        print("✅ Workflow complete.\n")
        return
    elif needs_approval is None:
        # We couldn't find the flag – safest is to NOT ask automatically
        print("\n⚠️ Could not find 'needs_approval' in tool output.")
        print("Skipping approval prompt; assuming workflow is complete.\n")
        return

    # If we reach here, needs_approval == True
    decision = input("\nApprove this order? (y/n): ").strip().lower()
    if decision not in ("y", "n"):
        print("Invalid or empty decision; ending workflow without final approval.\n")
        return

    approve = decision == "y"
    decision_text = (
        "Yes, approve this order."
        if approve
        else "No, reject this order."
    )

    print(f"\n👤 You: {decision_text}")

    # Step 2 – send your approval decision back to the agent
    messages.append({"role": "assistant", "content": first_reply})
    messages.append({"role": "user", "content": decision_text})

    second_state = await agent.ainvoke(
        {"messages": messages},
        config={
            "verbose": True,
            "callbacks": [StreamingStdOutCallbackHandler()],
        },
    )
    second_reply = second_state["messages"][-1].content
    print("\n🤖 Agent (step 2 - final):")
    print(second_reply)
    print("\n✅ Workflow complete.\n")


# ─────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────

async def main():
    client, shipping_agent = await build_shipping_agent()

    # Demo 1: Small order – should auto-approve, no y/n prompt
    await run_shipping_workflow(shipping_agent, "Ship 3 containers to Singapore")

    # Demo 2: Large order – you decide y/n
    await run_shipping_workflow(shipping_agent, "Ship 10 containers to Rotterdam")

    # Demo 3: Another large order – you decide y/n
    await run_shipping_workflow(shipping_agent, "Ship 8 containers to Los Angeles")

    # await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
