"""Request and response schemas for the mcp-candledata server."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


class IndicatorName(str, Enum):
    """Supported indicator identifiers."""

    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"
    
    @classmethod
    def from_alias(cls, value: str) -> "IndicatorName":
        """Convert common aliases to standard indicator names."""
        aliases = {
            "bollinger": cls.BBANDS,
            "bb": cls.BBANDS,
            "bands": cls.BBANDS,
            "simple_moving_average": cls.SMA,
            "exponential_moving_average": cls.EMA,
            "relative_strength_index": cls.RSI,
            "rsi14": cls.RSI,
            "macd_histogram": cls.MACD,
        }
        normalized = value.lower().strip()
        return aliases.get(normalized, cls(normalized))


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
        le=200,
    )
    
    @field_validator("name", mode="before")
    @classmethod
    def _validate_indicator_name(cls, value: Union[str, IndicatorName]) -> IndicatorName:
        if isinstance(value, str):
            return IndicatorName.from_alias(value)
        return value


class CandleDataRequest(BaseModel):
    """Input model for the get_candle_data tool."""

    symbol: str = Field(
        ...,
        description="Ticker symbol (e.g., 'AAPL' for Yahoo Finance, 'KRW-BTC' for Upbit).",
        validation_alias=AliasChoices("symbol", "ticker"),
        min_length=1,
        max_length=20,
    )
    provider: Optional[DataProvider] = Field(
        default=None,
        description="Market data provider ('yahoo' for equities or 'upbit' for crypto).",
        validation_alias=AliasChoices("provider", "source", "vendor", "exchange"),
    )
    interval_minutes: int = Field(
        ...,
        gt=0,
        description="Time interval for each candle: 1=1min, 5=5min, 15=15min, 60=1hour, 1440=1day",
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
        description="Technical indicators to calculate. Use simple names like 'rsi', 'macd', 'sma', 'ema', 'bollinger'",
    )

    @field_validator("indicators", mode="before")
    @classmethod
    def _coerce_indicators(cls, value: Union[List[str], List[dict], List[IndicatorSpec]]) -> List[dict]:
        if not value:
            return []
        
        result = []
        for item in value:
            if isinstance(item, str):
                try:
                    indicator_name = IndicatorName.from_alias(item)
                    result.append({"name": indicator_name.value})
                except ValueError:
                    raise ValueError(
                        f"Unknown indicator '{item}'. Supported indicators: {', '.join([i.value for i in IndicatorName])}"
                    )
            elif isinstance(item, dict):
                result.append(item)
            elif hasattr(item, 'model_dump'):
                result.append(item.model_dump())
            else:
                result.append(item)
        return result
    start: Optional[datetime] = Field(
        default=None,
        description="Optional start timestamp (UTC). Overrides limit when provided.",
    )
    end: Optional[datetime] = Field(
        default=None,
        description="Optional end timestamp (UTC). Defaults to now when omitted.",
    )
    
    @field_validator("symbol", mode="before")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Symbol cannot be empty")
        cleaned = value.strip().upper()
        if len(cleaned) > 20:
            raise ValueError("Symbol too long (max 20 characters)")
        return cleaned
    
    @field_validator("start", "end", mode="before")
    @classmethod
    def _validate_timestamps(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return value
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        
        # Convert to UTC if timezone-naive
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        
        # Check reasonable date range (not too far in past/future)
        if value.year < 2000:
            raise ValueError("Start/end date cannot be before year 2000")
        if value > now:
            raise ValueError("Start/end date cannot be in the future")
            
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "CandleDataRequest":
        if self.start and self.end and self.start >= self.end:
            raise ValueError("Start timestamp must be earlier than end timestamp")
        
        # Auto-detect provider if not specified
        if self.provider is None:
            guessed = (
                DataProvider.UPBIT
                if _looks_like_upbit_symbol(self.symbol)
                else DataProvider.YAHOO
            )
            self.provider = guessed
        
        # Validate symbol format for specific providers
        if self.provider == DataProvider.UPBIT and not _looks_like_upbit_symbol(self.symbol):
            raise ValueError(
                f"Symbol '{self.symbol}' does not match Upbit format (expected: BASE-QUOTE like KRW-BTC)"
            )
        
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
                f"Invalid interval format '{value}'. Use minutes (e.g. 5) or duration string like '5m', '2h', '1d'"
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
    open: float = Field(ge=0, description="Opening price")
    high: float = Field(ge=0, description="Highest price")
    low: float = Field(ge=0, description="Lowest price")
    close: float = Field(ge=0, description="Closing price")
    volume: float = Field(ge=0, description="Trading volume")
    
    @model_validator(mode="after")
    def validate_ohlc(self) -> "CandleRecord":
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("OHLC values must satisfy: low ≤ open,close ≤ high")
        return self


class IndicatorPoint(BaseModel):
    """Single indicator datapoint."""

    timestamp: datetime
    values: Dict[str, Optional[float]]


class CandleDataResponse(BaseModel):
    """Response payload streamed back to the MCP runtime."""

    symbol: str = Field(description="The requested ticker symbol")
    interval_minutes: int = Field(gt=0, description="Candle interval in minutes")
    provider: DataProvider = Field(description="Data provider used")
    candles: List[CandleRecord] = Field(description="OHLCV candle data")
    indicators: Dict[str, List[IndicatorPoint]] = Field(
        default_factory=dict, 
        description="Technical indicator calculations"
    )
    
    @field_validator("candles")
    @classmethod
    def _validate_candles_sorted(cls, value: List[CandleRecord]) -> List[CandleRecord]:
        if len(value) > 1:
            timestamps = [candle.timestamp for candle in value]
            if timestamps != sorted(timestamps):
                raise ValueError("Candles must be sorted by timestamp in ascending order")
        return value
