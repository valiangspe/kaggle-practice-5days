# main.py
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


async def main() -> None:
    question = input("💬 Customer question: ")

    # 1) Create MCP client (no context manager on the client)
    client = MultiServerMCPClient(
        {
            "customer_support": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # 2) Open a session to the customer_support server
    async with client.session("customer_support") as session:
        # 3) Load tools from that session
        tools = await load_mcp_tools(session)

        # Find our customer_support_query tool
        support_tool = None
        for t in tools:
            if t.name.endswith("customer_support_query"):
                support_tool = t
                break

        if support_tool is None:
            raise RuntimeError(
                "Could not find customer_support_query tool on MCP server"
            )

        # Invoke the tool – args must match the MCP tool signature
        result = await support_tool.ainvoke({"question": question})

    print("\n🎧 Customer Support Agent:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
