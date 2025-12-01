# customer_support_mcp.py
import os
from typing import Any, Dict

from fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

# Configure OpenAI-compatible endpoint for LangChain (Ollama)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://ollama.iotech.my.id/v1")

# LangChain "provider:model" string; provider=openai, model=gpt-oss:20b
MODEL_NAME = "openai:gpt-oss:20b"

mcp = FastMCP("CustomerSupportAgent")


def _extract_text_from_agent_result(result: Any) -> str:
    """
    Best-effort extraction of plain text from LangChain agent result.
    """
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                pieces = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        pieces.append(part["text"])
                if pieces:
                    return "\n".join(pieces)
    return str(result)


@mcp.tool
async def customer_support_query(question: str) -> str:
    """
    Customer-facing tool.

    Uses LangChain + MultiServerMCPClient to call the Product Catalog MCP agent
    (port 8001) and answer questions using the Ollama gpt-oss:20b model.
    """
    # 1) Create MCP client (NO context manager on the client itself)
    client = MultiServerMCPClient(
        {
            "product_catalog": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # 2) Open a session for the 'product_catalog' server
    async with client.session("product_catalog") as session:
        # 3) Load tools from that session
        tools = await load_mcp_tools(session)

        # 4) Create a LangChain agent that can use those tools
        agent = create_agent(MODEL_NAME, tools)

        system_prompt = (
            "You are a friendly customer support agent for an electronics store.\n"
            "Always use the available tools from the product_catalog MCP server\n"
            "to look up product information before answering. Be concise but clear."
        )

        result: Dict[str, Any] = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ]
            }
        )

    return _extract_text_from_agent_result(result)


if __name__ == "__main__":
    # Expose this agent as MCP HTTP server on localhost:8000/mcp
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
    )
