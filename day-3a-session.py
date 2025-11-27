"""
Single-file LangChain demo:
- Session-style tools that save & retrieve user info
- Uses a JSON file on disk as a simple session store
- No MCP, just regular tools + create_agent
"""

import json
import os
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# -------------------------------------------------------------------
# "Scope levels" concept, similar to your ADK example (just for docs)
# -------------------------------------------------------------------
USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")

# -------------------------------------------------------------------
# Very simple JSON-backed session store
# -------------------------------------------------------------------
SESSION_FILE = "session_state.json"


def _load_session_state() -> Dict[str, Any]:
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_session_state(state: Dict[str, Any]) -> None:
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# Global in-memory cache, loaded once at startup
SESSION_STATE: Dict[str, Any] = _load_session_state()

# Keys we’ll use (mirrors your "user:name", "user:country")
USER_NAME_KEY = "user:name"
USER_COUNTRY_KEY = "user:country"


# -------------------------------------------------------------------
# Tools: save & retrieve user info (session state through JSON)
# -------------------------------------------------------------------

@tool
def save_userinfo(user_name: str, country: str) -> Dict[str, Any]:
    """
    Record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    global SESSION_STATE

    SESSION_STATE[USER_NAME_KEY] = user_name
    SESSION_STATE[USER_COUNTRY_KEY] = country

    # Persist to disk so it survives process restarts
    _save_session_state(SESSION_STATE)

    return {
        "status": "success",
        "stored": {
            "user_name": user_name,
            "country": country,
            "scope_levels": USER_NAME_SCOPE_LEVELS,
        },
    }


@tool
def retrieve_userinfo() -> Dict[str, Any]:
    """
    Retrieve user name and country from session state.
    """
    state = SESSION_STATE or _load_session_state()

    user_name = state.get(USER_NAME_KEY, "Username not found")
    country = state.get(USER_COUNTRY_KEY, "Country not found")

    return {
        "status": "success",
        "user_name": user_name,
        "country": country,
    }


print("✅ Tools created.")


# -------------------------------------------------------------------
# LLM + Agent with tools
# -------------------------------------------------------------------

def make_llm() -> ChatOpenAI:
    # Ollama via OpenAI-compatible endpoint
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",   # dummy string; required by client
        temperature=0.2,
    )


ROOT_SYSTEM_PROMPT = """
You are a text chatbot.

You have two tools to manage user context:
- `save_userinfo(user_name, country)`: Call this when the user tells you their name
  and/or country so you can store it in session state.
- `retrieve_userinfo()`: Call this when you need to recall the user's name or country
  from session state.

Behavior:
1. If a user asks for their name or country and you are not sure, call `retrieve_userinfo`.
2. If they introduce themselves (e.g., "My name is X" or "I'm from Y"), call `save_userinfo`.
3. Use the retrieved info in your answer, but do not expose the raw JSON unless asked.
4. Keep responses concise and conversational.
"""

llm = make_llm()

root_agent = create_agent(
    model=llm,
    tools=[save_userinfo, retrieve_userinfo],
    system_prompt=ROOT_SYSTEM_PROMPT,
)

print("✅ Agent with session state tools initialized!")


# -------------------------------------------------------------------
# Simple multi-turn session runner (no async, single file)
# -------------------------------------------------------------------

def run_session(agent, user_messages: List[str], session_name: str) -> None:
    """
    Very small helper that:
    - Maintains message history across turns
    - Calls the agent for each user message
    """
    print(f"\n============================")
    print(f" Session: {session_name}")
    print(f"============================\n")

    state: Dict[str, Any] = {"messages": []}

    for user_text in user_messages:
        print(f"👤 User: {user_text}")
        state["messages"].append({"role": "user", "content": user_text})

        # Invoke agent; it may call tools (save/retrieve_userinfo)
        state = agent.invoke(
            state,
            config={
                "verbose": True,
                "callbacks": [StreamingStdOutCallbackHandler()],
            },
        )

        ai_msg = state["messages"][-1]
        print(f"🤖 Assistant: {ai_msg.content}\n")


if __name__ == "__main__":
    # Test conversation demonstrating session state
    convo = [
        "Hi there, how are you doing today? What is my name?",
        "My name is Valian. I'm from Indonesia.",
        "What is my name? Which country am I from?",
    ]
    run_session(root_agent, convo, session_name="state-demo-session")
