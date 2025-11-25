# pip install "langchain>=1.0.0" langchain-openai langchain-community duckduckgo-search

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# === LLM: Ollama via OpenAI-compatible endpoint ===
llm = ChatOpenAI(
    model="gpt-oss:20b",
    base_url="https://ollama.iotech.my.id/v1",  # your Ollama server
    api_key="ollama",  # dummy, just required by client
    temperature=0.2,
)

# === Tool: DuckDuckGo web search (as your Google Search replacement) ===
google_search = DuckDuckGoSearchRun(
    name="google_search",
    description=(
        "Use this to search the web for current / up-to-date information, "
        "or when you are unsure."
    ),
)

tools = [google_search]

# === Root agent (new LangChain style) ===
root_agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Use the google_search tool for current info or if you are unsure."
    ),
)

print("✅ Root Agent defined (LangChain new-style agents).")


# === Example usage helper ===
def ask_root_agent(query: str) -> str:
    """
    Simple wrapper so you can call ask_root_agent('...') instead of
    dealing with the full agent state dict.
    """
    state = root_agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"callbacks": [StreamingStdOutCallbackHandler()], "verbose": True},
    )

    # Agent returns a state dict; last message is the assistant answer
    return state["messages"][-1].content


if __name__ == "__main__":
    query = (
        "What is the capital city of Indonesia, and 1–2 recent news headlines about it?"
    )
    answer = ask_root_agent(query)
    print("\n=== Query ===")
    print(query)
    print()
    print(answer)
