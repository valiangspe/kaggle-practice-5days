# product_catalog_mcp.py
from fastmcp import FastMCP

mcp = FastMCP("ProductCatalogAgent")


@mcp.tool
def get_product_info(product_name: str) -> str:
    """
    Get product information for a given product.

    This simulates the external vendor product catalog.
    In a real system this would query a DB or API.
    """
    product_catalog = {
        "iphone 15 pro": "iPhone 15 Pro, $999, Low Stock (8 units), 128GB, Titanium finish",
        "samsung galaxy s24": "Samsung Galaxy S24, $799, In Stock (31 units), 256GB, Phantom Black",
        "dell xps 15": 'Dell XPS 15, $1,299, In Stock (45 units), 15.6" display, 16GB RAM, 512GB SSD',
        "macbook pro 14": 'MacBook Pro 14", $1,999, In Stock (22 units), M3 Pro chip, 18GB RAM, 512GB SSD',
        "sony wh-1000xm5": "Sony WH-1000XM5 Headphones, $399, In Stock (67 units), Noise-canceling, 30hr battery",
        "ipad air": 'iPad Air, $599, In Stock (28 units), 10.9" display, 64GB',
        "lg ultrawide 34": 'LG UltraWide 34" Monitor, $499, Out of Stock, Expected: Next week',
    }

    key = product_name.lower().strip()
    if key in product_catalog:
        return f"Product: {product_catalog[key]}"
    else:
        available = ", ".join(p.title() for p in product_catalog.keys())
        return (
            f"Sorry, I don't have information for '{product_name}'. "
            f"Available products: {available}"
        )


if __name__ == "__main__":
    # Streamable HTTP MCP endpoint on localhost:8001/mcp
    # This matches the langchain MultiServerMCPClient config with
    # transport="streamable_http", url="http://localhost:8001/mcp"
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
        path="/mcp",
    )
