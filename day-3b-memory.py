"""
Memory Management Demo (LangChain version of Kaggle ADK Day 3 notebook)
- Session = short-term conversation (per session_id)
- Memory  = long-term store across sessions

Features:
✅ InMemorySessionService – stores conversations
✅ InMemoryMemoryService  – stores session summaries & does keyword search
✅ load_memory tool       – reactive memory lookup
✅ preload_memory tool    – proactive memory injection
✅ Manual & automatic memory saving flows
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json

# ============================================================
# Section 1: Basic config
# ============================================================

APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"

print("✅ Basic config set.")


# ============================================================
# Section 2: Session Service (short-term memory)
# ============================================================


@dataclass
class ConversationSession:
    app_name: str
    user_id: str
    session_id: str
    events: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]


class InMemorySessionService:
    def __init__(self):
        # key: (app_name, user_id, session_id)
        self._sessions: Dict[Tuple[str, str, str], ConversationSession] = {}

    def get_or_create_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> ConversationSession:
        key = (app_name, user_id, session_id)
        if key not in self._sessions:
            self._sessions[key] = ConversationSession(app_name, user_id, session_id)
        return self._sessions[key]

    def get_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> ConversationSession:
        key = (app_name, user_id, session_id)
        return self._sessions[key]


session_service = InMemorySessionService()
print("✅ InMemorySessionService initialized.")


# ============================================================
# Section 3: Memory Service (long-term store)
# ============================================================


@dataclass
class MemoryRecord:
    session_id: str
    text: str  # concatenation of conversation events


class InMemoryMemoryService:
    def __init__(self):
        # key: (app_name, user_id) -> List[MemoryRecord]
        self._store: Dict[Tuple[str, str], List[MemoryRecord]] = {}

    def add_session_to_memory(self, session: ConversationSession) -> None:
        """Ingests a full session into long-term memory."""
        key = (session.app_name, session.user_id)
        buf_lines = []
        for e in session.events:
            buf_lines.append(f"{e['role']}: {e['content']}")
        text = "\n".join(buf_lines)

        rec = MemoryRecord(session_id=session.session_id, text=text)
        self._store.setdefault(key, []).append(rec)

    def search_memory(
        self, app_name: str, user_id: str, query: str, limit: int = 5
    ) -> List[MemoryRecord]:
        """Very simple keyword search over stored texts."""
        key = (app_name, user_id)
        records = self._store.get(key, [])
        q = query.lower().strip()
        if not q:
            return records[:limit]
        results: List[MemoryRecord] = []
        for r in records:
            if q in r.text.lower():
                results.append(r)
                if len(results) >= limit:
                    break
        return results


memory_service = InMemoryMemoryService()
print("✅ InMemoryMemoryService initialized.")


# ============================================================
# Section 4: Tools: load_memory / preload_memory
# ============================================================


@tool
def load_memory(query: str) -> str:
    """
    Reactive memory search.

    The agent calls this when it *thinks* it needs past information.
    `query` should describe what it is looking for, e.g. "favorite color" / "birthday".
    """
    records = memory_service.search_memory(APP_NAME, USER_ID, query, limit=5)
    if not records:
        return "No relevant memories found."

    snippets = []
    for r in records:
        # Take first ~200 chars for brevity
        short = (r.text[:200] + "…") if len(r.text) > 200 else r.text
        snippets.append(f"[session={r.session_id}]\n{short}")
    return "\n\n---\n\n".join(snippets)


@tool
def preload_memory() -> str:
    """
    Proactive memory search.

    Returns a summary of all stored memories (up to a limit).
    The agent can call this at the start of a turn to preload context.
    """
    records = memory_service.search_memory(APP_NAME, USER_ID, query="", limit=5)
    if not records:
        return "No stored memories yet."

    summaries = []
    for r in records:
        short = (r.text[:200] + "…") if len(r.text) > 200 else r.text
        summaries.append(f"[session={r.session_id}] {short}")
    return "\n\n".join(summaries)


print("✅ Memory tools created (load_memory, preload_memory).")


# ============================================================
# Section 5: LLM factory (Ollama, OpenAI-compatible)
# ============================================================


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",  # dummy string required by client
        temperature=0.2,
    )


# ============================================================
# Section 6: Helper – run a session
# ============================================================


def run_session(
    agent,
    user_queries: List[str] | str,
    session_id: str = "default",
    auto_save_to_memory: bool = False,
) -> None:
    """
    Helper to run queries with a given agent and session_id.

    - Maintains short-term history per session (SessionService)
    - Optionally auto-saves session to Memory after each turn
    """
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    session = session_service.get_or_create_session(APP_NAME, USER_ID, session_id)
    history: List[BaseMessage] = []

    print(f"\n### Session: {session_id}\n")

    for query in user_queries:
        print(f"User > {query}")
        session.events.append({"role": "user", "content": query})
        history.append(HumanMessage(content=query))

        state = agent.invoke(
            {"messages": history},
            config={
                "verbose": True,
                "callbacks": [StreamingStdOutCallbackHandler()],
            },
        )
        # Agent returns a state dict with messages, last message is assistant
        ai_msg: AIMessage = state["messages"][-1]  # type: ignore
        history.append(ai_msg)
        session.events.append({"role": "assistant", "content": ai_msg.content})

        print(f"Model > {ai_msg.content}\n")

        if auto_save_to_memory:
            memory_service.add_session_to_memory(session)
            print("💾 (auto) Session snapshot saved to memory.\n")


print("✅ run_session helper defined.")


# ============================================================
# Section 7: Basic agent (no memory) to populate sessions
# ============================================================

base_agent_system_prompt = """
You are a friendly assistant. Answer in simple, clear language.
"""

base_agent = create_agent(
    model=make_llm(),
    tools=[],  # no memory tools yet
    system_prompt=textwrap.dedent(base_agent_system_prompt).strip(),
)

print("✅ Base agent (no memory) created.")


# ============================================================
# Section 8: Ingest Session into Memory
# ============================================================

# 8.1: Have a conversation about favorite color
run_session(
    base_agent,
    "My favorite color is blue-green. Can you write a small poem about it?",
    session_id="conversation-01",
)

# 8.2: Inspect session events (short-term memory)
session_01 = session_service.get_session(APP_NAME, USER_ID, "conversation-01")
print("📝 Session 'conversation-01' events:")
for e in session_01.events:
    print(f"  {e['role']}: {e['content'][:60]}...")

# 8.3: Add this session to long-term memory
memory_service.add_session_to_memory(session_01)
print("\n✅ Session 'conversation-01' added to memory.\n")


# ============================================================
# Section 9: Agent with load_memory (reactive retrieval)
# ============================================================

reactive_system_prompt = """
You are a helpful assistant.

You have access to a tool called `load_memory(query)`.
Use it when you need to recall information from past conversations, such as:
- the user's favorite color
- the user's birthday
- other personal preferences

If a user asks about past preferences and you are not sure, call `load_memory`
with a short query like "favorite color" or "birthday".
"""

reactive_agent = create_agent(
    model=make_llm(),
    tools=[load_memory],
    system_prompt=textwrap.dedent(reactive_system_prompt).strip(),
)

print("✅ Reactive memory agent (load_memory) created.")

# 9.1: New session – ask about favorite color
run_session(
    reactive_agent,
    "What is my favorite color?",
    session_id="color-test",
)


# ============================================================
# Section 10: Another example – birthday, manual memory ingest
# ============================================================

# 10.1: Tell the base agent about birthday
run_session(
    base_agent,
    "My birthday is on March 15th.",
    session_id="birthday-session-01",
)

# 10.2: Save that session to memory
birthday_session = session_service.get_session(APP_NAME, USER_ID, "birthday-session-01")
memory_service.add_session_to_memory(birthday_session)
print("✅ Birthday session saved to memory.\n")

# 10.3: Ask in a NEW session – must use load_memory to recall
run_session(
    reactive_agent,
    "When is my birthday?",
    session_id="birthday-session-02",
)


# ============================================================
# Section 11: Manual memory search (outside the agent)
# ============================================================

print("\n🔎 Manual memory search: 'favorite color'\n")
matches = memory_service.search_memory(APP_NAME, USER_ID, "favorite color")
print(f"  Found {len(matches)} memories.\n")
for m in matches:
    print(f"[session={m.session_id}]\n{m.text[:120]}...\n")


# ============================================================
# Section 12: Auto memory – agent with preload_memory + auto save
# ============================================================

auto_system_prompt = """
You are an assistant with automatic memory support.

Tools:
- `preload_memory()` – call this at the start of each turn to load any
  relevant past information about the user (favorite color, birthday, etc).

Instructions:
1. On EVERY new user message, first call `preload_memory()` to see what
   you already know about the user.
2. Then answer the user, using that information if it is helpful.
"""

auto_agent = create_agent(
    model=make_llm(),
    tools=[preload_memory],
    system_prompt=textwrap.dedent(auto_system_prompt).strip(),
)

print("✅ Auto-memory agent (preload_memory) created.")


# 12.1: First session – say something that should be remembered.
run_session(
    auto_agent,
    "I gifted a new toy to my nephew on his 1st birthday!",
    session_id="auto-save-test",
    auto_save_to_memory=True,  # simulate after_agent_callback
)

# 12.2: New session – ask about the gift, relying on memory
run_session(
    auto_agent,
    "What did I gift my nephew?",
    session_id="auto-save-test-2",
    auto_save_to_memory=True,
)

print("\n🎉 Memory demo complete.")

from dataclasses import asdict

print("\n📚 Dumping all sessions in session_service:\n")

for (app_name, user_id, session_id), sess in session_service._sessions.items():
    print(f"=== Session key: app={app_name}, user={user_id}, session={session_id} ===")
    # Option 1: simple dataclass print
    # print(sess)

    # Option 2: pretty JSON
    all_sessions_dict = {
        f"{app}:{user}:{sess_id}": asdict(sess)
        for (app, user, sess_id), sess in session_service._sessions.items()
    }

    print(json.dumps(all_sessions_dict, indent=2, ensure_ascii=False))
