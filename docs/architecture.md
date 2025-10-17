# mcp-candledata Architecture

The project exposes a Model Context Protocol (MCP) server that streams candle and technical-indicator data over HTTP. The implementation is organised into three modules:

## Server entrypoint
- `src/mcp_candledata/server.py` defines the MCP server instance using the official `modelcontextprotocol` Python SDK.
- The server registers a single tool, `get_candle_data`, which validates incoming requests with Pydantic schemas and streams the result payload back to the MCP runtime.
- A small `main` runner starts an HTTP transport so the server can be tunnelled through solutions such as `ngrok`.

## Data access layer
- `src/mcp_candledata/datasource.py` downloads OHLCV candles via `yfinance`.
- Minute-resolution intervals (e.g. `1`, `5`, `15`) are mapped to Yahoo Finance interval strings.
- The module also exposes helpers for computing and formatting technical indicators.

## Schemas and indicator engine
- `src/mcp_candledata/schemas.py` defines request/response models and enumerations.
- `src/mcp_candledata/indicators.py` maps indicator identifiers to pandas-based implementations (SMA, EMA, RSI, MACD, Bollinger Bands).
- Indicator results are serialised as records so they can be streamed back line-by-line when the MCP transport requests incremental updates.

## Dependency management
- Dependencies are declared in `pyproject.toml` and resolved with `uv`.
- Core runtime dependencies: `modelcontextprotocol[http]`, `pydantic`, `pandas`, `yfinance`.
- Development tooling (formatting, linting) can be added later via `uv add --dev`.

## Execution flow
1. MCP runtime calls the `get_candle_data` tool with a payload containing ticker, minute interval, candle limit, optional indicators, and optional explicit start/end timestamps.
2. The server resolves the request into a canonical time range, downloads the required candles, and computes the requested indicators.
3. The server yields a structured response containing recent candles and a map of indicator name → series data.

This layout keeps business logic focused and allows future extension (e.g. caching, alternative data vendors) by adding new modules under `src/mcp_candledata/`.
