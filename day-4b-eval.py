# day-4b-eval.py
# pip install "langchain>=1.0.0" langchain-openai langchain-core

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# === Tool implementation ======================================================

@tool
def set_device_status(location: str, device_id: str, status: str) -> Dict[str, Any]:
    """
    Sets the status of a smart home device.

    Args:
        location: The room where the device is located.
        device_id: The unique identifier for the device.
        status: The desired status, either 'ON' or 'OFF'.

    Returns:
        A dictionary confirming the action.
    """
    status_upper = status.upper()
    if status_upper not in ("ON", "OFF"):
        return {
            "success": False,
            "message": f"Invalid status '{status}'. Use 'ON' or 'OFF'.",
        }

    print(f"[TOOL] set_device_status: location={location}, device_id={device_id}, status={status_upper}")
    return {
        "success": True,
        "location": location,
        "device_id": device_id,
        "status": status_upper,
        "message": f"Successfully set the {device_id} in {location} to {status_upper.lower()}.",
    }


tools = [set_device_status]


# === LLM: Ollama via OpenAI-compatible endpoint ===============================

llm = ChatOpenAI(
    model="gpt-oss:20b",
    base_url="https://ollama.iotech.my.id/v1",  # your Ollama server
    api_key="ollama",  # dummy, just required by client
    temperature=0.1,
)


# === Home automation agent (new LangChain style) ==============================

home_automation_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a home automation assistant.\n\n"
        "You can control smart devices ONLY via the 'set_device_status' tool.\n"
        "- When the user asks to turn something ON or OFF, you MUST call the tool.\n"
        "- The status must be either 'ON' or 'OFF'.\n"
        "- Do not claim to control devices you have not actually set via the tool.\n"
        "- After the tool runs, summarize the action clearly for the user."
    ),
)

print("✅ Home automation agent defined (LangChain new-style agent).")


# === Helper: ask the agent ====================================================

def ask_home_agent(query: str) -> str:
    """
    Simple wrapper so you can call ask_home_agent('...') instead of
    dealing with the full agent state dict.
    """
    state = home_automation_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [StreamingStdOutCallbackHandler()], "verbose": True},
    )

    # Agent returns a state dict; last message is the assistant answer
    return state["messages"][-1].content


# === Tiny "evaluation-like" demo =============================================

TEST_CASES = [
    {
        "id": "living_room_light_on",
        "prompt": "Please turn on the floor lamp in the living room.",
        "expected_substring": "floor lamp in the living room",
    },
    {
        "id": "kitchen_main_light_on",
        "prompt": "Switch on the main light in the kitchen.",
        "expected_substring": "main light in the kitchen",
    },
]


def run_simple_eval() -> None:
    print("\n=== Running simple eval over test cases ===\n")
    for case in TEST_CASES:
        print(f"--- Test: {case['id']} ---")
        print(f"User > {case['prompt']}")
        answer = ask_home_agent(case["prompt"])
        print("\nAssistant >")
        print(answer)
        ok = case["expected_substring"].lower() in answer.lower()
        print(
            f"\nRESULT: {'PASS' if ok else 'FAIL'} "
            f"(expected to mention: '{case['expected_substring']}')\n"
        )
        print("=" * 80)


if __name__ == "__main__":
    # Quick manual run, plus the tiny eval loop
    print("\n>>> Manual single query\n")
    resp = ask_home_agent("Turn on the desk lamp in the office.")
    print("\nFinal answer:\n", resp)

    run_simple_eval()
