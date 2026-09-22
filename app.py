import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="BTC Interval Monitor",
    page_icon="₿",
    layout="wide",
)

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_CLOCK_OFFSET_SECONDS = 20


def get_btc_candles(limit=90):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - pd.Timedelta(minutes=limit)

    response = requests.get(
        "https://api.exchange.coinbase.com/products/BTC-USD/candles",
        params={
            "granularity": 60,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        headers={"User-Agent": "btc-kalshi-dashboard/1.0"},
        timeout=15,
    )
    response.raise_for_status()

    rows = response.json()

    if not rows:
        raise ValueError("Coinbase returned no BTC candle data.")

    df = pd.DataFrame(
        rows,
        columns=["open_time", "low", "high", "open", "close", "volume"],
    )

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df["time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)

    return df.sort_values("time").reset_index(drop=True)


def get_kalshi_market(ticker):
    if not ticker.strip():
        return None, None, None

    try:
        market_response = requests.get(
            f"{KALSHI_BASE}/markets/{ticker.strip()}",
            timeout=10,
        )
        market_response.raise_for_status()

        market = market_response.json().get(
            "market",
            market_response.json(),
        )

        book_response = requests.get(
            f"{KALSHI_BASE}/markets/{ticker.strip()}/orderbook",
            timeout=10,
        )
        book_response.raise_for_status()

        orderbook = book_response.json().get(
            "orderbook",
            book_response.json(),
        )

        return market, orderbook, None

    except requests.RequestException as error:
        return None, None, str(error)


def normal_cdf(value):
    return 0.5 * math.erfc(-value / math.sqrt(2))


def calculate_model(df, strike, minutes_left):
    closes = df["close"].to_numpy()
    current_price = closes[-1]

    recent_returns = np.diff(np.log(closes[-31:]))
    sigma_1m = float(np.std(recent_returns, ddof=1))

    horizon_volatility = max(
        sigma_1m * math.sqrt(max(minutes_left, 1)),
        0.00015,
    )

    momentum_5m = current_price / closes[-6] - 1
    momentum_drift = 0.20 * momentum_5m * min(minutes_left, 10)

    z_score = (
        math.log(strike) - (math.log(current_price) + momentum_drift)
    ) / horizon_volatility

    probability_above = float(
        np.clip(1 - normal_cdf(z_score), 0.02, 0.98)
    )

    return {
        "price": current_price,
        "distance": current_price - strike,
        "momentum_5m": momentum_5m,
        "volatility_1m": sigma_1m,
        "above": probability_above,
        "below": 1 - probability_above,
    }


st.title("₿ BTC 15-Minute Interval Monitor")
st.caption(
    "Read-only research dashboard. It does not connect to your Kalshi "
    "account and cannot place trades."
)

with st.sidebar:
    st.header("Contract details")

    kalshi_ticker = st.text_input(
        "Kalshi market ticker (optional)",
        placeholder="KXBTC...",
    )

    strike = st.number_input(
        "BTC threshold / strike",
        min_value=1.0,
        value=85513.00,
        step=0.01,
    )

    settlement_input = st.text_input(
        "Settlement time (UTC)",
        value="2026-09-22T03:30:00Z",
        help="Example: 10:30 PM CDT = 03:30 UTC the following day.",
    )

    st.button("Refresh data", type="primary")

try:
    settlement_time = datetime.fromisoformat(
        settlement_input.replace("Z", "+00:00")
    )

    minutes_left = max(
        (
            settlement_time
            - datetime.now(timezone.utc)
        ).total_seconds() / 60
        - KALSHI_CLOCK_OFFSET_SECONDS / 60,
        0,
    )

except ValueError:
    st.error("Use this UTC time format: 2026-09-22T03:30:00Z")
    st.stop()

try:
    candles = get_btc_candles()
except requests.RequestException as error:
    st.error(f"Could not load BTC candle data: {error}")
    st.stop()

model = calculate_model(candles, strike, minutes_left)
market, orderbook, kalshi_error = get_kalshi_market(kalshi_ticker)

col1, col2, col3, col4 = st.columns(4)

col1.metric("BTC spot", f"${model['price']:,.2f}")
col2.metric("Strike", f"${strike:,.2f}")
col3.metric("Distance from strike", f"${model['distance']:,.2f}")
col4.metric("Minutes remaining", f"{minutes_left:.1f}")

left_column, right_column = st.columns([2, 1])

with left_column:
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=candles["time"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="BTC/USDT",
        )
    )

    figure.add_hline(
        y=strike,
        line_dash="dash",
        line_color="#f5c542",
        annotation_text=f"Strike ${strike:,.2f}",
    )

    figure.update_layout(
        title="Recent BTC 1-Minute Candles",
        height=540,
        xaxis_rangeslider_visible=False,
        yaxis_title="BTC price",
    )

    st.plotly_chart(figure, use_container_width=True)

with right_column:
    st.subheader("Model estimate")

    st.metric("Above probability", f"{model['above'] * 100:.1f}%")
    st.metric("Below probability", f"{model['below'] * 100:.1f}%")
    st.metric(
        "5-minute momentum",
        f"{model['momentum_5m'] * 100:.3f}%",
    )
    st.metric(
        "1-minute volatility",
        f"{model['volatility_1m'] * 100:.3f}%",
    )

    if model["above"] > 0.60:
        st.success("Model leans above. This is not a guarantee.")
    elif model["above"] < 0.40:
        st.warning("Model leans below. This is not a guarantee.")
    else:
        st.info("Near coin-flip range. No strong model edge.")

st.divider()
st.subheader("Kalshi public market data")

if kalshi_ticker.strip() and kalshi_error:
    st.error(f"Could not load Kalshi data: {kalshi_error}")

elif market:
    market_col1, market_col2, market_col3 = st.columns(3)

    market_col1.metric(
        "Market title",
        market.get("title", "Not provided"),
    )
    market_col2.metric(
        "YES bid",
        f"{market.get('yes_bid', 'N/A')}¢",
    )
    market_col3.metric(
        "YES ask",
        f"{market.get('yes_ask', 'N/A')}¢",
    )

    with st.expander("Show public Kalshi order-book data"):
        st.json(orderbook)

else:
    st.info(
        "Enter a Kalshi ticker to load public market and order-book data."
    )

st.divider()
st.warning(
    "This is analysis only, not financial or betting advice. "
    "The estimate can be wrong, especially near settlement. "
    "Kalshi contract rules, settlement source, and timestamp decide the outcome."
)
