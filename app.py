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


def calculate_rsi(closes, period=14):
    deltas = closes.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi.fillna(50.0)


def calculate_atr(df, period=5):
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period, min_periods=period).mean()


def calculate_chart_bot(df, strike, minutes_left):
    bot_df = df.copy()
    bot_df["ema_5"] = bot_df["close"].ewm(span=5, adjust=False).mean()
    bot_df["ema_12"] = bot_df["close"].ewm(span=12, adjust=False).mean()
    bot_df["rsi_14"] = calculate_rsi(bot_df["close"], period=14)
    bot_df["atr_5"] = calculate_atr(bot_df, period=5)

    current_price = float(bot_df["close"].iloc[-1])
    ema_5 = float(bot_df["ema_5"].iloc[-1])
    ema_12 = float(bot_df["ema_12"].iloc[-1])
    rsi_14 = float(bot_df["rsi_14"].iloc[-1])
    atr_5 = float(bot_df["atr_5"].iloc[-1])

    recent_15 = bot_df.tail(15)
    range_high = float(recent_15["high"].max())
    range_low = float(recent_15["low"].min())
    range_midpoint = (range_high + range_low) / 2

    last_three = bot_df.tail(3)
    green_candles = int((last_three["close"] > last_three["open"]).sum())
    red_candles = int((last_three["close"] < last_three["open"]).sum())

    distance_dollars = abs(current_price - strike)
    distance_pct = ((current_price - strike) / strike) * 100
    atr_multiple = distance_dollars / atr_5 if atr_5 > 0 else 0.0

    above_points = 0
    below_points = 0
    signal_rows = []

    if current_price > strike:
        above_points += 1
        signal_rows.append(("Price vs strike", "ABOVE", "Price is above strike"))
    elif current_price < strike:
        below_points += 1
        signal_rows.append(("Price vs strike", "BELOW", "Price is below strike"))
    else:
        signal_rows.append(("Price vs strike", "NEUTRAL", "Price equals strike"))

    if ema_5 > ema_12:
        above_points += 1
        signal_rows.append(("EMA trend", "ABOVE", "5 EMA is above 12 EMA"))
    elif ema_5 < ema_12:
        below_points += 1
        signal_rows.append(("EMA trend", "BELOW", "5 EMA is below 12 EMA"))
    else:
        signal_rows.append(("EMA trend", "NEUTRAL", "EMAs are equal"))

    if rsi_14 >= 55:
        above_points += 1
        signal_rows.append(("RSI 14", "ABOVE", f"RSI is {rsi_14:.1f}"))
    elif rsi_14 <= 45:
        below_points += 1
        signal_rows.append(("RSI 14", "BELOW", f"RSI is {rsi_14:.1f}"))
    else:
        signal_rows.append(("RSI 14", "NEUTRAL", f"RSI is {rsi_14:.1f}"))

    if green_candles >= 2:
        above_points += 1
        signal_rows.append(("Last 3 candles", "ABOVE", f"{green_candles} green candles"))
    elif red_candles >= 2:
        below_points += 1
        signal_rows.append(("Last 3 candles", "BELOW", f"{red_candles} red candles"))
    else:
        signal_rows.append(("Last 3 candles", "NEUTRAL", "Mixed candles"))

    if current_price > range_midpoint:
        above_points += 1
        signal_rows.append(("15-minute range", "ABOVE", "Price is above range midpoint"))
    elif current_price < range_midpoint:
        below_points += 1
        signal_rows.append(("15-minute range", "BELOW", "Price is below range midpoint"))
    else:
        signal_rows.append(("15-minute range", "NEUTRAL", "Price is at range midpoint"))

    if atr_multiple >= 2:
        cushion_label = "STRONG"
        cushion_text = "Distance is at least 2× recent 5-minute ATR"
    elif atr_multiple >= 1:
        cushion_label = "MODERATE"
        cushion_text = "Distance is at least 1× recent 5-minute ATR"
    else:
        cushion_label = "TOO CLOSE"
        cushion_text = "Distance is less than recent 5-minute ATR"

    signal_rows.append(("ATR cushion", cushion_label, cushion_text))

    time_ok = 3 <= minutes_left <= 10
    close_to_strike = abs(distance_pct) < 0.05
    atr_filter_passes = atr_multiple >= 1

    if close_to_strike:
        label = "NO CLEAR EDGE"
        confidence = "Low"
        summary = "BTC is too close to the strike price."
    elif not time_ok:
        label = "NO CLEAR EDGE"
        confidence = "Low"
        summary = "Bot only scores the 3–10 minute window."
    elif not atr_filter_passes:
        label = "NO CLEAR EDGE"
        confidence = "Low"
        summary = "BTC is too close to strike compared with recent volatility."
    elif above_points >= 4 and above_points > below_points:
        label = "LEAN ABOVE"
        confidence = "Medium"
        summary = f"{above_points}/5 chart signals lean above with an ATR cushion."
    elif below_points >= 4 and below_points > above_points:
        label = "LEAN BELOW"
        confidence = "Medium"
        summary = f"{below_points}/5 chart signals lean below with an ATR cushion."
    else:
        label = "NO CLEAR EDGE"
        confidence = "Low"
        summary = "Chart signals are mixed."

    return {
        "df": bot_df,
        "label": label,
        "confidence": confidence,
        "summary": summary,
        "above_points": above_points,
        "below_points": below_points,
        "current_price": current_price,
        "distance_dollars": distance_dollars,
        "distance_pct": distance_pct,
        "ema_5": ema_5,
        "ema_12": ema_12,
        "rsi_14": rsi_14,
        "atr_5": atr_5,
        "atr_multiple": atr_multiple,
        "cushion_label": cushion_label,
        "green_candles": green_candles,
        "red_candles": red_candles,
        "range_high": range_high,
        "range_low": range_low,
        "range_midpoint": range_midpoint,
        "signals": signal_rows,
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
except (requests.RequestException, ValueError) as error:
    st.error(f"Could not load BTC candle data: {error}")
    st.stop()

if len(candles) < 31:
    st.error("Not enough BTC candles returned. Refresh and try again.")
    st.stop()

model = calculate_model(candles, strike, minutes_left)
chart_bot = calculate_chart_bot(candles, strike, minutes_left)
market, orderbook, kalshi_error = get_kalshi_market(kalshi_ticker)

col1, col2, col3, col4 = st.columns(4)

col1.metric("BTC spot", f"${model['price']:,.2f}")
col2.metric("Strike", f"${strike:,.2f}")
col3.metric("Distance from strike", f"${model['distance']:,.2f}")
col4.metric("Minutes remaining", f"{minutes_left:.1f}")

st.divider()
st.subheader("Chart research bot")

bot_col1, bot_col2, bot_col3, bot_col4, bot_col5 = st.columns(5)
bot_col1.metric("Bot result", chart_bot["label"])
bot_col2.metric("Confidence", chart_bot["confidence"])
bot_col3.metric("Above signals", f"{chart_bot['above_points']} / 5")
bot_col4.metric("Below signals", f"{chart_bot['below_points']} / 5")
bot_col5.metric("ATR cushion", chart_bot["cushion_label"])
st.caption(chart_bot["summary"])

with st.expander("Show chart-bot signals"):
    signal_df = pd.DataFrame(
        chart_bot["signals"],
        columns=["Indicator", "Signal", "Explanation"],
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
    detail_col1.metric("EMA 5", f"${chart_bot['ema_5']:,.2f}")
    detail_col2.metric("EMA 12", f"${chart_bot['ema_12']:,.2f}")
    detail_col3.metric("RSI 14", f"{chart_bot['rsi_14']:.1f}")
    detail_col4.metric("5-min ATR", f"${chart_bot['atr_5']:,.2f}")

    st.caption(
        f"Distance from strike: ${chart_bot['distance_dollars']:,.2f} | "
        f"ATR multiple: {chart_bot['atr_multiple']:.2f}× | "
        f"15-minute range: ${chart_bot['range_low']:,.2f} to "
        f"${chart_bot['range_high']:,.2f}"
    )

left_column, right_column = st.columns([2, 1])

with left_column:
    chart_df = chart_bot["df"]
    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=chart_df["time"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="BTC/USD",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema_5"],
            mode="lines",
            name="EMA 5",
            line=dict(color="#00cc96", width=1.5),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["ema_12"],
            mode="lines",
            name="EMA 12",
            line=dict(color="#636efa", width=1.5),
        )
    )

    figure.add_hline(
        y=strike,
        line_dash="dash",
        line_color="#f5c542",
        annotation_text=f"Strike ${strike:,.2f}",
    )

    figure.update_layout(
        title="Recent BTC 1-Minute Candles with EMA Lines",
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
    "Chart research only, not financial or betting advice. This bot uses "
    "historical Coinbase candles and can be wrong. Kalshi contract rules, "
    "settlement source, and timestamp decide the outcome."
)
