"""Request and response schemas for the mcp-candledata server."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class IndicatorName(str, Enum):
    """Supported indicator identifiers."""

    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"


class DataProvider(str, Enum):
    """Available market data vendors."""

    YAHOO = "yahoo"
    UPBIT = "upbit"


class IndicatorSpec(BaseModel):
    """Indicator configuration payload."""

    name: IndicatorName
    window: Optional[int] = Field(
        default=None,
        description="Primary lookback window (if applicable). "
        "Falls back to sensible default per indicator.",
        gt=1,
    )


class CandleDataRequest(BaseModel):
    """Input model for the get_candle_data tool."""

    symbol: str = Field(
        ...,
        description="Ticker symbol supported by Yahoo Finance.",
        validation_alias=AliasChoices("symbol", "ticker"),
    )
    provider: Optional[DataProvider] = Field(
        default=None,
        description="Market data provider ('yahoo' for equities or 'upbit' for crypto).",
        validation_alias=AliasChoices("provider", "source", "vendor", "exchange"),
    )
    interval_minutes: int = Field(
        ...,
        gt=0,
        description="Candle interval in minutes (e.g. 1, 5, 15, 60).",
        validation_alias=AliasChoices("interval_minutes", "interval"),
    )
    limit: int = Field(
        default=120,
        ge=5,
        le=500,
        description="Number of recent candles to return.",
    )
    indicators: List[IndicatorSpec] = Field(
        default_factory=list,
        description="List of technical indicators to evaluate.",
    )
    start: Optional[datetime] = Field(
        default=None,
        description="Optional start timestamp (UTC). Overrides limit when provided.",
    )
    end: Optional[datetime] = Field(
        default=None,
        description="Optional end timestamp (UTC). Defaults to now when omitted.",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "CandleDataRequest":
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be earlier than end")
        if self.provider is None:
            guessed = (
                DataProvider.UPBIT
                if _looks_like_upbit_symbol(self.symbol)
                else DataProvider.YAHOO
            )
            self.provider = guessed
        return self

    @field_validator("interval_minutes", mode="before")
    @classmethod
    def _coerce_interval(cls, value: int | str) -> int | str:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned.endswith("m"):
                return int(cleaned[:-1])
            if cleaned.endswith("h"):
                return int(cleaned[:-1]) * 60
            if cleaned.endswith("d"):
                return int(cleaned[:-1]) * 1440
            if cleaned.isdigit():
                return int(cleaned)
            raise ValueError(
                "interval must be provided as minutes (e.g. 5) or duration string "
                "like '5m', '2h', or '1d'"
            )
        return value


def _looks_like_upbit_symbol(symbol: str) -> bool:
    """Heuristically determine whether the ticker resembles an Upbit market code."""
    market = symbol.strip().upper()
    if "-" not in market:
        return False
    base, _quote = market.split("-", 1)
    return base in {"KRW", "BTC", "USDT"}


class CandleRecord(BaseModel):
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorPoint(BaseModel):
    """Single indicator datapoint."""

    timestamp: datetime
    values: Dict[str, Optional[float]]


class CandleDataResponse(BaseModel):
    """Response payload streamed back to the MCP runtime."""

    symbol: str
    interval_minutes: int
    candles: List[CandleRecord]
    indicators: Dict[str, List[IndicatorPoint]]
