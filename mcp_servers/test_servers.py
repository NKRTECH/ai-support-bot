"""
Quick test to verify all three MCP servers work.

Connects to each server via stdio, discovers tools/resources,
and makes one test call per server.
"""

import asyncio
import sys
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def test_server(name: str, module: str, test_fn):
    """Connect to a server, list capabilities, and run a test function."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print(f"\nTools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"  - {tool.name}")

            # List resources
            resources = await session.list_resources()
            print(f"\nResources ({len(resources.resources)}):")
            for res in resources.resources:
                print(f"  - {res.uri}")

            # Run server-specific test
            await test_fn(session)


async def test_orders(session):
    result = await session.call_tool(
        "check_order_status_tool", {"order_id": "ORD-1001"}
    )
    print(f"\ncheck_order_status_tool('ORD-1001'):")
    print(f"  {result.content[0].text[:120]}...")

    resource = await session.read_resource("orders://recent")
    print(f"\norders://recent -> {len(resource.contents)} content block(s)")


async def test_accounts(session):
    result = await session.call_tool(
        "get_customer_info_tool", {"email": "rahul.sharma@email.com"}
    )
    print(f"\nget_customer_info_tool('rahul.sharma@email.com'):")
    print(f"  {result.content[0].text[:120]}...")

    resource = await session.read_resource("accounts://tiers")
    print(f"\naccounts://tiers -> {len(resource.contents)} content block(s)")


async def test_knowledge(session):
    # List prompts
    prompts = await session.list_prompts()
    print(f"\nPrompts ({len(prompts.prompts)}):")
    for p in prompts.prompts:
        print(f"  - {p.name}")

    resource = await session.read_resource("docs://list")
    print(f"\ndocs://list -> {len(resource.contents)} content block(s)")

    result = await session.call_tool(
        "search_docs", {"query": "return policy"}
    )
    text = result.content[0].text
    print(f"\nsearch_docs('return policy'):")
    print(f"  {text[:150]}...")


async def main():
    await test_server(
        "smarttech-orders",
        "mcp_servers.orders_server",
        test_orders,
    )
    await test_server(
        "smarttech-accounts",
        "mcp_servers.accounts_server",
        test_accounts,
    )
    await test_server(
        "smarttech-knowledge",
        "mcp_servers.knowledge_server",
        test_knowledge,
    )

    print(f"\n{'=' * 50}")
    print("  All 3 servers passed!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
