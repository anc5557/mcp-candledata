"""MCP server entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from modelcontextprotocol.server import Server
from modelcontextprotocol.server.http import HTTPServerTransport

from .datasource import CandleDataError, fetch_and_prepare
from .schemas import CandleDataRequest

LOGGER = logging.getLogger("mcp_candledata")

server = Server("mcp-candledata")


@server.tool(
    name="get_candle_data",
    description="Fetch OHLCV candles and technical indicator values for a symbol.",
)
async def get_candle_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tool handler invoked by the runtime."""
    request = CandleDataRequest.model_validate(payload)
    try:
        result = await asyncio.to_thread(fetch_and_prepare, request)
    except CandleDataError as exc:
        LOGGER.exception("Failed to compute candle data")
        raise ValueError(str(exc)) from exc

    return result.model_dump(mode="json")


def run(
    host: Optional[str] = None,
    port: Optional[int] = None,
    enable_streaming: Optional[bool] = None,
) -> None:
    """Start the MCP HTTP server."""
    logging.basicConfig(level=logging.INFO)

    resolved_host = host or os.getenv("MCP_CANDLEDATA_HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("MCP_CANDLEDATA_PORT", "8765"))
    resolved_streaming = (
        enable_streaming
        if enable_streaming is not None
        else os.getenv("MCP_CANDLEDATA_STREAMING", "1") != "0"
    )

    transport = HTTPServerTransport(
        host=resolved_host, port=resolved_port, enable_streaming=resolved_streaming
    )

    if hasattr(server, "run"):
        server.run(transport)
    elif hasattr(server, "serve"):
        asyncio.run(server.serve(transport))
    else:
        raise RuntimeError("modelcontextprotocol Server does not expose run/serve helpers")


if __name__ == "__main__":
    run()
