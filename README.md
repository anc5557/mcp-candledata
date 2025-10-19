# mcp-candledata

분 단위 OHLCV 캔들과 보조지표 값을 MCP(Model Context Protocol) 인터페이스로
제공하는 HTTP 서버입니다. 단타용 워크플로우에서 티커, 분봉 간격, 보조지표
설정을 입력하면 실시간에 가까운 가격/지표 데이터를 스트리밍 방식으로 받을 수 있습니다.

## 사전 준비
- Python 3.11 이상
- 패키지 관리 도구 [uv](https://docs.astral.sh/uv/)
- 서버 실행 시 Yahoo Finance API( `yfinance` )에 접근 가능한 네트워크
- MCP 호환 클라이언트(OpenAI Desktop, VSCode MCP 브리지 등)

## 빠른 시작
```bash
# 가상환경 생성 및 의존성 설치
uv sync

# MCP HTTP 서버 실행 (기본: 0.0.0.0:8765, 스트리밍 활성화)
uv run mcp-candledata
```

선택적 환경 변수:
- `MCP_CANDLEDATA_HOST` – 바인딩 주소 (기본값 `0.0.0.0`)
- `MCP_CANDLEDATA_PORT` – 포트 (기본값 `8765`)
- `MCP_CANDLEDATA_STREAMING` – `0`으로 설정하면 스트리밍 비활성화

## 도구 계약(Tool Contract)

`get_candle_data` 는 다음과 같은 JSON 입력을 받습니다.
```jsonc
{
  "ticker": "TSLA",              // alias: "symbol"
  "provider": "yahoo",           // alias: "source", "vendor", "exchange" (기본값 auto)
  "interval": "5m",              // alias: "interval_minutes"
  "limit": 120,
  "indicators": [
    { "name": "sma", "window": 20 },
    { "name": "rsi" }
  ]
}
```

`symbol`/`interval_minutes` 필드 이름과 정수 분 단위 입력도 그대로 지원합니다.

가상화폐(업비트) 시세를 조회하고 싶다면 `provider` 를 `"upbit"` 으로 지정하거나
티커를 `KRW-BTC`, `USDT-ETH` 처럼 업비트 마켓 코드 형식으로 전달하세요.
지원 간격은 업비트 REST API 규격(분봉: 1/3/5/10/15/30/60/240, 일봉: 1d)에 맞춰집니다.
예시:
```json
{
  "symbol": "KRW-BTC",
  "provider": "upbit",
  "interval": 1,
  "limit": 100
}
```

지원하는 보조지표 이름: `sma`, `ema`, `rsi`, `macd`, `bbands`

응답에는 요청한 분봉에 맞춰 정렬된 최근 캔들과 각 보조지표의 시계열 데이터가 포함됩니다.

## 터널링 / 원격 접속

`ngrok` 등 터널링 도구로 로컬 HTTP 엔드포인트를 노출할 수 있습니다.
```bash
uv run mcp-candledata &
ngrok http http://127.0.0.1:8765
```
전달받은 퍼블릭 URL을 MCP 클라이언트 설정에 입력하면 외부에서도 동일하게 접근할 수 있습니다.
