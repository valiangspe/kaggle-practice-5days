"""
Agent Observability Demo (LangChain version of ADK Day 4 logging notebook)

- Configures DEBUG logging to logger.log
- Implements a "LoggingPlugin"-style callback using LangChain's BaseCallbackHandler
- Builds a simple "research paper finder" agent:
    * fake_google_search(query) -> list of paper titles
    * count_papers(papers: List[str]) -> number of papers
- Runs a debug call and prints logs.

Requirements:
    pip install "langchain>=1.0.0" langchain-openai

Make sure your Ollama endpoint is reachable at:
    https://ollama.iotech.my.id/v1
with model: gpt-oss:20b
"""

import logging
import os
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ============================================================
# Section 1: Logging setup
# ============================================================

# Clean up any previous logs
for log_file in ["logger.log"]:
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"🧹 Cleaned up {log_file}")

# Configure logging with DEBUG level
logging.basicConfig(
    filename="logger.log",
    level=logging.DEBUG,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s:%(message)s",
)
logger = logging.getLogger(__name__)

print("✅ Logging configured (logger.log, DEBUG level)")


# ============================================================
# Section 2: "Plugin"-style callback for observability
# ============================================================

class LoggingCallbackHandler(BaseCallbackHandler):
    """
    Rough equivalent of ADK's LoggingPlugin, but for LangChain.

    Logs:
      - user messages and agent responses
      - LLM prompts and outputs
      - tool calls and results
      - simple counters for agents, tools, and LLM requests
    """

    def __init__(self):
        self.agent_runs = 0
        self.tool_calls = 0
        self.llm_requests = 0

    # ---- High level (chains / agents) ----
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        self.agent_runs += 1
        logger.info(
            "[logging_plugin] 🤖 AGENT/CHAIN STARTING | runs=%d | serialized=%s | inputs_keys=%s",
            self.agent_runs,
            serialized.get("id"),
            list(inputs.keys()),
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        logger.info(
            "[logging_plugin] 🤖 AGENT/CHAIN COMPLETED | outputs_keys=%s",
            list(outputs.keys()),
        )

    # ---- LLM-level tracing ----
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.llm_requests += 1
        logger.info(
            "[logging_plugin] 🧠 LLM REQUEST | count=%d | model=%s",
            self.llm_requests,
            serialized.get("id"),
        )
        for i, p in enumerate(prompts):
            logger.debug(
                "[logging_plugin] 🧠 LLM PROMPT %d:\n%s\n---END PROMPT---",
                i,
                p,
            )

    def on_llm_end(self, response, **kwargs):
        # response.generations is a list of lists of ChatGeneration
        logger.info("[logging_plugin] 🧠 LLM RESPONSE RECEIVED")
        try:
            for gen_idx, gen_list in enumerate(response.generations):
                for choice_idx, choice in enumerate(gen_list):
                    text = getattr(choice, "text", None) or getattr(choice.message, "content", "")
                    logger.debug(
                        "[logging_plugin] 🧠 LLM RESPONSE (%d,%d): %s",
                        gen_idx,
                        choice_idx,
                        text,
                    )
        except Exception as e:
            logger.exception("[logging_plugin] Error while logging LLM response: %s", e)

    # ---- Tool calls ----
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        self.tool_calls += 1
        logger.info(
            "[logging_plugin] 🔧 TOOL STARTING | count=%d | name=%s | input=%s",
            self.tool_calls,
            serialized.get("name"),
            input_str,
        )

    def on_tool_end(self, output: str, **kwargs):
        logger.info(
            "[logging_plugin] 🔧 TOOL COMPLETED | output=%s",
            output[:200] + "..." if isinstance(output, str) and len(output) > 200 else output,
        )

    def on_tool_error(self, error: Exception, **kwargs):
        logger.error("[logging_plugin] 🔧 TOOL ERROR | %s", error)


# ============================================================
# Section 3: Tools – fake google search + count_papers
# ============================================================

@tool
def fake_google_search(request: str) -> List[str]:
    """
    Fake 'google_search' tool returning a dummy list of 'papers'.

    In a real system, you'd call SerpAPI, Tavily, or your own search backend.
    Here we just pretend to return research paper titles.
    """
    logger.debug("[fake_google_search] called with request=%r", request)

    # Pretend these came from a search engine:
    papers = [
        "Recent Advances in Quantum Computing Architectures",
        "Error Correction Techniques for Logical Qubits in Quantum Processors",
        "Hybrid Quantum-Classical Algorithms for Optimization Problems",
        "Scalable Quantum Hardware: Superconducting and Trapped-Ion Approaches",
        "Applications of Quantum Computing in Cryptography and Finance",
    ]
    return papers


@tool
def count_papers(papers: List[str]) -> int:
    """
    Count the number of papers in a list of strings.

    Args:
        papers: A list of strings, each string is a research paper title.
    Returns:
        The number of papers in the list.
    """
    logger.debug("[count_papers] called with %d papers", len(papers))
    return len(papers)


print("✅ Tools created (fake_google_search, count_papers)")


# ============================================================
# Section 4: LLM + Agent definition
# ============================================================

def make_llm() -> ChatOpenAI:
    """Create an Ollama-backed ChatOpenAI client."""
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",  # dummy required by client
        temperature=0.2,
    )


research_agent_system_prompt = """
You are a research paper finder agent.

Your task is to find research papers and count them.

You MUST follow these steps:

1) Use the `fake_google_search` tool to search for research papers on the topic given by the user.
2) Pass the list of papers to the `count_papers` tool to count how many were found.
3) Return BOTH:
   - The list of research papers (titles)
   - The total number of papers found.

Be concise but clear in your explanation.
"""

research_agent = create_agent(
    model=make_llm(),
    tools=[fake_google_search, count_papers],
    system_prompt=research_agent_system_prompt.strip(),
)

print("✅ Research agent created.")


# ============================================================
# Section 5: Debug runner (like InMemoryRunner + run_debug)
# ============================================================

def run_debug(prompt: str) -> str:
    """
    Equivalent to ADK's runner.run_debug(...) but in LangChain.

    - Wraps a single user turn
    - Attaches LoggingCallbackHandler and StreamingStdOutCallbackHandler
    - Returns the final text response from the agent
    """
    print("🚀 Running agent with LoggingCallbackHandler...")
    print("📊 Watch console output, then inspect logger.log for detailed traces.\n")

    # messages history: single-turn debug
    messages: List[BaseMessage] = [HumanMessage(content=prompt)]

    callbacks = [LoggingCallbackHandler(), StreamingStdOutCallbackHandler()]

    # agent.invoke expects a dict with "messages"
    state = research_agent.invoke(
        {"messages": messages},
        config={
            "verbose": True,
            "callbacks": callbacks,
        },
    )

    final_msg = state["messages"][-1]
    if isinstance(final_msg, AIMessage):
        print("\nresearch_agent >", final_msg.content)
        return final_msg.content
    else:
        print("\n[WARN] Last message is not AIMessage:", final_msg)
        return str(final_msg)


# ============================================================
# Section 6: Main – run and dump logs
# ============================================================

if __name__ == "__main__":
    user_query = "Find recent research papers on quantum computing."
    result = run_debug(user_query)

    print("\n✅ Agent run completed.")
    print("📝 Agent response:\n", result)

    # Show the contents of logger.log like ADK's `!cat logger.log`
    print("\n🔍 Examining logger.log for debugging clues...\n")
    try:
        with open("logger.log", "r", encoding="utf-8") as f:
            log_contents = f.read()
        print(log_contents)
    except FileNotFoundError:
        print("logger.log not found (did logging configuration run correctly?).")
