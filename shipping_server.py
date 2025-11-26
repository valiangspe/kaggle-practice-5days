# shipping_server.py
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Shipping")

LARGE_ORDER_THRESHOLD = 5


@mcp.tool()
async def check_order_needs_approval(num_containers: int) -> Dict[str, Any]:
    """
    Check if an order for `num_containers` requires human approval.

    Returns:
      {
        "status": "ok",
        "num_containers": int,
        "threshold": int,
        "needs_approval": bool,
        "message": str
      }
    """
    needs = num_containers > LARGE_ORDER_THRESHOLD
    return {
        "status": "ok",
        "num_containers": num_containers,
        "threshold": LARGE_ORDER_THRESHOLD,
        "needs_approval": needs,
        "message": (
            "Order exceeds auto-approval threshold; human approval required."
            if needs
            else "Order is within auto-approval limit; no human approval required."
        ),
    }


@mcp.tool()
async def place_shipping_order(
    num_containers: int,
    destination: str,
    tool_confirmation: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Places a shipping order.

    Long-running / confirmation logic:

    1. If num_containers <= LARGE_ORDER_THRESHOLD:
         - Auto-approve the order (no confirmation needed).
    2. If num_containers > LARGE_ORDER_THRESHOLD and tool_confirmation is None:
         - Return status 'pending' and a hint asking for confirmation.
    3. If num_containers > LARGE_ORDER_THRESHOLD and tool_confirmation is True:
         - Approve the order.
    4. If num_containers > LARGE_ORDER_THRESHOLD and tool_confirmation is False:
         - Reject the order.
    """

    # SCENARIO 1: small orders auto-approve
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            "destination": destination,
            "message": (
                f"Order auto-approved: {num_containers} containers to {destination}."
            ),
        }

    # SCENARIO 2: large order, first call → need confirmation
    if tool_confirmation is None:
        return {
            "status": "pending",
            "needs_confirmation": True,
            "num_containers": num_containers,
            "destination": destination,
            "hint": (
                f"⚠️ Large order: {num_containers} containers to {destination}. "
                "Ask the user whether to approve or reject this order."
            ),
            "message": f"Order for {num_containers} containers requires approval.",
        }

    # SCENARIO 3: second call with decision
    if tool_confirmation:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "num_containers": num_containers,
            "destination": destination,
            "message": (
                f"Order approved by user: {num_containers} containers to {destination}."
            ),
        }
    else:
        return {
            "status": "rejected",
            "num_containers": num_containers,
            "destination": destination,
            "message": (
                f"Order rejected by user: {num_containers} containers to {destination}."
            ),
        }


if __name__ == "__main__":
    # e.g. http://localhost:8000/mcp
    mcp.run(transport="streamable-http")
