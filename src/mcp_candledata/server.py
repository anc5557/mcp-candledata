"""MCP server entrypoint."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .datasource import CandleDataError, fetch_and_prepare
from .schemas import CandleDataRequest, CandleDataResponse

LOGGER = logging.getLogger("mcp_candledata")

mcp = FastMCP("mcp-candledata")


@mcp.tool()
async def get_candle_data(request: CandleDataRequest) -> CandleDataResponse:
    """
    Get stock/crypto price data and technical analysis indicators.
    
    Use this when users ask for:
    - Stock prices (AAPL, TSLA, etc.) or crypto prices (BTC, ETH)
    - Chart data, candlestick patterns, OHLCV data
    - Technical indicators like RSI, MACD, moving averages
    - Market analysis, price trends, trading signals
    
    Supports Yahoo Finance (stocks) and Upbit (Korean crypto exchange).
    """
    try:
        result = await asyncio.to_thread(fetch_and_prepare, request)
    except CandleDataError as exc:
        LOGGER.exception("Failed to compute candle data")
        raise ValueError(str(exc)) from exc

    return result


def run(
    host: Optional[str] = None,
    port: Optional[int] = None,
    enable_streaming: Optional[bool] = None,
) -> None:
    """Start the MCP HTTP server (FastMCP streamable-http)."""
    logging.basicConfig(level=logging.INFO)

    resolved_host = host or os.getenv("MCP_CANDLEDATA_HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("MCP_CANDLEDATA_PORT", "8765"))
    resolved_streaming = (
        enable_streaming
        if enable_streaming is not None
        else os.getenv("MCP_CANDLEDATA_STREAMING", "1") != "0"
    )

    # FastMCP uses json_response to disable SSE streaming
    json_response = not resolved_streaming

    # Configure runtime settings
    mcp.settings.host = resolved_host
    mcp.settings.port = resolved_port
    mcp.settings.json_response = json_response

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run()
