"""Data access utilities for fetching candles and computing indicators."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .schemas import (
    CandleDataRequest,
    CandleRecord,
    IndicatorName,
    IndicatorPoint,
    IndicatorSpec,
    CandleDataResponse,
)


class CandleDataError(RuntimeError):
    """Raised when candle data cannot be retrieved or processed."""


_MINUTE_TO_YF_INTERVAL: Dict[int, str] = {
    1: "1m",
    2: "2m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "60m",
    90: "90m",
    120: "2h",
    240: "4h",
    360: "6h",
    480: "8h",
    720: "12h",
    1440: "1d",
}


def fetch_and_prepare(request: CandleDataRequest) -> CandleDataResponse:
    """Fetch raw candles, compute indicators, and produce a response payload."""

    interval = _resolve_interval(request.interval_minutes)
    df = _download_candles(request.symbol, interval, request)
    if df.empty:
        raise CandleDataError(
            f"No candle data returned for symbol '{request.symbol}' "
            f"with interval '{interval}'."
        )

    df = df.sort_index()
    truncated = df.tail(request.limit)

    candles = [_row_to_candle(row) for _, row in truncated.iterrows()]
    indicator_payload = _build_indicators(truncated, request.indicators)

    return CandleDataResponse(
        symbol=request.symbol.upper(),
        interval_minutes=request.interval_minutes,
        candles=candles,
        indicators=indicator_payload,
    )


def _resolve_interval(minutes: int) -> str:
    try:
        return _MINUTE_TO_YF_INTERVAL[minutes]
    except KeyError as exc:
        raise CandleDataError(
            f"Interval '{minutes}' minutes is not supported by Yahoo Finance."
        ) from exc


def _download_candles(
    symbol: str, interval: str, request: CandleDataRequest
) -> pd.DataFrame:
    params: Dict[str, object] = {
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
    }

    if request.start:
        params["start"] = _ensure_timezone(request.start)
    if request.end:
        params["end"] = _ensure_timezone(request.end)

    if "start" not in params and "end" not in params:
        params["period"] = _infer_period(request.interval_minutes, request.limit)

    data = yf.download(symbol, **params)
    if data.empty:
        return data

    # The download may use a timezone-aware index. Convert to UTC to standardise.
    index = data.index
    if hasattr(index, "tz") and index.tz is not None:
        data.index = index.tz_convert(timezone.utc)
    else:
        data.index = pd.DatetimeIndex(index).tz_localize(timezone.utc)
    return data


def _infer_period(interval_minutes: int, limit: int) -> str:
    total_minutes = interval_minutes * limit
    # Yahoo Finance accepted periods for minute data: 1d,5d,1mo,3mo,6mo,1y,2y
    if total_minutes <= 390:
        return "1d"
    if total_minutes <= 1950:
        return "5d"
    if total_minutes <= 11700:
        return "1mo"
    if total_minutes <= 35100:
        return "3mo"
    if total_minutes <= 70200:
        return "6mo"
    if total_minutes <= 140400:
        return "1y"
    return "2y"


def _row_to_candle(row: pd.Series) -> CandleRecord:
    timestamp = row.name.to_pydatetime()
    return CandleRecord(
        timestamp=timestamp,
        open=_require_float(row["Open"]),
        high=_require_float(row["High"]),
        low=_require_float(row["Low"]),
        close=_require_float(row["Close"]),
        volume=_optional_float(row.get("Volume", np.nan)) or 0.0,
    )


def _build_indicators(
    df: pd.DataFrame, indicators: Iterable[IndicatorSpec]
) -> Dict[str, List[IndicatorPoint]]:
    payload: Dict[str, List[IndicatorPoint]] = {}
    close = df["Close"]

    for spec in indicators:
        name = spec.name
        if name == IndicatorName.SMA:
            series = _sma(close, window=spec.window or 20)
            payload[name.value] = _series_to_points(df.index, series, ["value"])
        elif name == IndicatorName.EMA:
            series = _ema(close, window=spec.window or 20)
            payload[name.value] = _series_to_points(df.index, series, ["value"])
        elif name == IndicatorName.RSI:
            series = _rsi(close, window=spec.window or 14)
            payload[name.value] = _series_to_points(df.index, series, ["value"])
        elif name == IndicatorName.MACD:
            macd, signal, hist = _macd(close)
            payload[name.value] = _multi_series_to_points(
                df.index,
                {
                    "macd": macd,
                    "signal": signal,
                    "histogram": hist,
                },
            )
        elif name == IndicatorName.BBANDS:
            upper, middle, lower = _bollinger_bands(close, window=spec.window or 20)
            payload[name.value] = _multi_series_to_points(
                df.index,
                {"upper": upper, "middle": middle, "lower": lower},
            )
        else:
            payload[name.value] = []

    return payload


def _series_to_points(
    index: pd.Index, series: pd.Series, labels: List[str]
) -> List[IndicatorPoint]:
    points: List[IndicatorPoint] = []
    for ts, value in zip(index, series):
        timestamp = ts.to_pydatetime()
        val = _optional_float(value)
        points.append(IndicatorPoint(timestamp=timestamp, values={labels[0]: val}))
    return points


def _multi_series_to_points(
    index: pd.Index, series_map: Dict[str, pd.Series]
) -> List[IndicatorPoint]:
    points: List[IndicatorPoint] = []
    labels = list(series_map.keys())
    for idx, ts in enumerate(index):
        timestamp = ts.to_pydatetime()
        values = {}
        for label in labels:
            value = series_map[label].iloc[idx]
            values[label] = _optional_float(value)
        points.append(IndicatorPoint(timestamp=timestamp, values=values))
    return points


def _extract_scalar(value: object) -> Optional[object]:
    """Pandas가 반환하는 단일 원소 시리즈/배열을 스칼라로 변환."""
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        return value.iloc[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.item()
    return value


def _require_float(value: object) -> float:
    scalar = _extract_scalar(value)
    if scalar is None or pd.isna(scalar):
        raise CandleDataError("필수 가격 데이터에 결측치가 포함돼 있습니다.")
    return float(scalar)


def _optional_float(value: object) -> Optional[float]:
    scalar = _extract_scalar(value)
    if scalar is None or pd.isna(scalar):
        return None
    return float(scalar)


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(series, window=12)
    slow = _ema(series, window=26)
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    middle = _sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


def _ensure_timezone(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
