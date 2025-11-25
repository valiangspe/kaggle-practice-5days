# pip install "langchain>=1.0.0" langchain-openai langchain-community duckduckgo-search

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# ============= LLM (Ollama via OpenAI-compatible API) =============
def make_llm():
    return ChatOpenAI(
        model="gpt-oss:20b",
        base_url="https://ollama.iotech.my.id/v1",
        api_key="ollama",  # dummy; required by client
        temperature=0.2,
    )


# ============= Shared web-search tool (acts like google_search) =============
google_search = DuckDuckGoSearchRun(
    name="google_search",
    description=(
        "Use this to search the web for current / up-to-date information. "
        "Always use it for facts, stats, or when you are unsure."
    ),
)


# ============= Research Agent =============
RESEARCH_SYSTEM_PROMPT = """
You are a specialized research agent.
Your only job is to use the `google_search` tool to find 2–3 pieces of
relevant information on the given topic and present the findings with citations.

Rules:
- ALWAYS call the google_search tool at least once.
- Include 2–3 short bullet points.
- Add inline source hints, e.g. (Source: site.com).
"""

research_agent = create_agent(
    model=make_llm(),
    tools=[google_search],
    system_prompt=RESEARCH_SYSTEM_PROMPT,
)

print("✅ research_agent created.")


# Wrap research_agent as a tool so the root agent can call it
@tool("ResearchAgent")
def research_agent_tool(topic: str) -> str:
    """Use this to research a topic on the web and return 2–3 bullet-point findings with citations."""
    state = research_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": f"Research this topic: {topic}"}
            ]
        },
        config={"verbose": True},  # prints thoughts / tool calls for this sub-agent
    )
    content = state["messages"][-1].content
    
    print("\n=== Research Agent Content ===")
    print(content)
    print()
    
    return content


# ============= Summarizer Agent =============
SUMMARIZER_SYSTEM_PROMPT = """
You are a summarization specialist.

You will be given research findings text.
Your job:

- Read the provided research findings carefully.
- Create a concise summary as a bulleted list with 3–5 key points.
- Be clear and non-redundant.
"""

summarizer_agent = create_agent(
    model=make_llm(),
    tools=[],  # no external tools; pure LLM summarizer
    system_prompt=SUMMARIZER_SYSTEM_PROMPT,
)

print("✅ summarizer_agent created.")


# Wrap summarizer_agent as a tool so the root agent can call it
@tool("SummarizerAgent")
def summarizer_agent_tool(research_findings: str) -> str:
    """Use this to summarize research findings into a concise bulleted list (3–5 key points)."""
    prompt = (
        "Read the provided research findings below and create a concise summary "
        "as a bulleted list with 3–5 key points.\n\n"
        f"=== RESEARCH FINDINGS ===\n{research_findings}\n"
    )
    state = summarizer_agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"verbose": True},  # print reasoning for this sub-agent too
    )

    content = state["messages"][-1].content
    
    print("\n=== Summarizer Agent Content ===")
    print(content)
    print()
    

    return content


print("✅ research_agent_tool and summarizer_agent_tool created.")


# ============= Root Coordinator Agent =============
ROOT_SYSTEM_PROMPT = """
You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.

You have access to two tools:
- ResearchAgent(topic: str) -> str
- SummarizerAgent(research_findings: str) -> str

Follow this workflow STRICTLY:
1. First, you MUST call the `ResearchAgent` tool to find relevant information
   on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the
   `SummarizerAgent` tool, passing the findings as `research_findings`.
3. Finally, present ONLY the final summary from the SummarizerAgent clearly
   to the user as your response.

Do NOT expose raw tool JSON or internal traces in the final answer.
Only return the clean human-readable summary.
"""

root_agent = create_agent(
    model=make_llm(),
    tools=[research_agent_tool, summarizer_agent_tool],
    system_prompt=ROOT_SYSTEM_PROMPT,
)

print("✅ root_agent created.")


# ============= Helper to call the root agent =============
def run_research_pipeline(user_query: str) -> str:
    """
    Calls the root coordinator agent on a user's query.

    This will:
    - Call ResearchAgent (web search)
    - Then SummarizerAgent (summary over findings)
    - Then return the final summary text
    """
    state = root_agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        # verbose + callback to print the "thoughts" / tool calls to console
        config={
            "verbose": True,
            "callbacks": [StreamingStdOutCallbackHandler()],
        },
    )
    return state["messages"][-1].content


# ============= Example usage =============
if __name__ == "__main__":
    query = "Explain the main benefits and challenges of containerized data centers."
    print("\n=== Query ===")
    print(query)
    print()
    final_answer = run_research_pipeline(query)
    print("\n=== FINAL ANSWER TO USER ===")
    print(final_answer)
