"""Request and response schemas for the mcp-candledata server."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class IndicatorName(str, Enum):
    """Supported indicator identifiers."""

    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"


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

    symbol: str = Field(..., description="Ticker symbol supported by Yahoo Finance.")
    interval_minutes: int = Field(
        ..., gt=0, description="Candle interval in minutes (e.g. 1, 5, 15, 60)."
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
        return self


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
