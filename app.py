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
    initial_sidebar_state="collapsed",
)

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_CLOCK_OFFSET_SECONDS = 20

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 0.55rem;
            padding: 0.45rem 0.65rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.15rem;
        }
        h1 {
            font-size: 1.65rem;
            margin-bottom: 0.1rem;
        }
        h2, h3 {
            font-size: 1.05rem;
            margin-top: 0.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def calculate_setup_checker(df, strike, minutes_left, model):
    current_price = model["price"]
    distance_pct = ((current_price - strike) / strike) * 100

    recent_prices = df["close"].tail(5)
    recent_high = float(recent_prices.max())
    recent_low = float(recent_prices.min())
    momentum_pct = model["momentum_5m"] * 100

    if current_price > strike:
        direction = "ABOVE"
    elif current_price < strike:
        direction = "BELOW"
    else:
        direction = "NEUTRAL"

    far_enough_from_strike = abs(distance_pct) >= 0.10
    momentum_agrees = (
        (direction == "ABOVE" and momentum_pct > 0)
        or (direction == "BELOW" and momentum_pct < 0)
    )

    if direction == "ABOVE":
        near_recent_barrier = current_price >= recent_high * 0.9995
    elif direction == "BELOW":
        near_recent_barrier = current_price <= recent_low * 1.0005
    else:
        near_recent_barrier = True

    right_time_window = 4.5 <= minutes_left <= 8.5
    checks_passed = sum(
        [
            direction != "NEUTRAL",
            far_enough_from_strike,
            momentum_agrees,
            not near_recent_barrier,
            right_time_window,
        ]
    )

    if (
        direction != "NEUTRAL"
        and far_enough_from_strike
        and momentum_agrees
        and not near_recent_barrier
        and right_time_window
    ):
        label = f"CONSIDER {direction}"
        explanation = "All setup rules passed. This is a filter, not a prediction."
    else:
        label = "NO TRADE"
        reasons = []

        if not far_enough_from_strike:
            reasons.append("too close to strike")
        if not momentum_agrees:
            reasons.append("momentum disagrees")
        if near_recent_barrier:
            reasons.append("near recent high/low")
        if not right_time_window:
            reasons.append("outside 5–8 min window")

        explanation = " • ".join(reasons) or "No clear setup."

    return {
        "label": label,
        "direction": direction,
        "distance_pct": distance_pct,
        "momentum_pct": momentum_pct,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "far_enough": far_enough_from_strike,
        "momentum_agrees": momentum_agrees,
        "near_barrier": near_recent_barrier,
        "right_time": right_time_window,
        "checks_passed": checks_passed,
        "explanation": explanation,
    }


def calculate_risk_cap(setup, model, minutes_left, risk_bankroll):
    if not 4.5 <= minutes_left <= 5.5:
        return {
            "amount": 0.0,
            "percent": 0.0,
            "reason": "Sizing appears only with about 5 minutes left.",
        }

    if setup["label"] == "NO TRADE":
        return {
            "amount": 0.0,
            "percent": 0.0,
            "reason": "No amount: the setup checker says NO TRADE.",
        }

    confidence = max(model["above"], model["below"]) * 100

    if confidence < 60:
        risk_percent = 0.0
    elif confidence < 65:
        risk_percent = 0.5
    elif confidence < 70:
        risk_percent = 1.0
    elif confidence < 75:
        risk_percent = 1.5
    else:
        risk_percent = 2.0

    risk_amount = risk_bankroll * (risk_percent / 100)

    return {
        "amount": risk_amount,
        "percent": risk_percent,
        "reason": (
            f"Conservative maximum based on {confidence:.1f}% model confidence. "
            "It is not an instruction to trade."
        ),
    }


st.title("₿ BTC 15-Minute Monitor")
st.caption("Read-only research dashboard — no account connection and no order placement.")

with st.sidebar:
    st.subheader("Contract inputs")

    kalshi_ticker = st.text_input(
        "Kalshi ticker (optional)",
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

    risk_bankroll = st.number_input(
        "Risk bankroll for sizing only ($)",
        min_value=0.0,
        value=25.0,
        step=5.0,
        help="Use only money you can afford to lose. This is a conservative cap, not a recommendation.",
    )

    st.button("Refresh data", type="primary", use_container_width=True)

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
    st.error("Use UTC format like: 2026-09-22T03:30:00Z")
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
setup = calculate_setup_checker(candles, strike, minutes_left, model)
risk_cap = calculate_risk_cap(setup, model, minutes_left, risk_bankroll)
market, orderbook, kalshi_error = get_kalshi_market(kalshi_ticker)

spot_col, strike_col, distance_col, time_col = st.columns(4)
spot_col.metric("BTC spot", f"${model['price']:,.2f}")
strike_col.metric("Strike", f"${strike:,.2f}")
distance_col.metric("Distance", f"${model['distance']:,.2f}")
time_col.metric("Minutes left", f"{minutes_left:.1f}")

setup_left, setup_right = st.columns([3, 2])

with setup_left:
    st.subheader("Setup checker")
    label_col, direction_col, checks_col = st.columns(3)
    label_col.metric("Status", setup["label"])
    direction_col.metric("Direction", setup["direction"])
    checks_col.metric("Rules", f"{setup['checks_passed']} / 5")
    st.caption(setup["explanation"])

with setup_right:
    st.subheader("5-minute risk cap")
    amount_col, percent_col = st.columns(2)
    amount_col.metric("Maximum risk", f"${risk_cap['amount']:,.2f}")
    percent_col.metric("Bankroll cap", f"{risk_cap['percent']:.1f}%")
    st.caption(risk_cap["reason"])

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
            name="BTC/USD",
        )
    )
    figure.add_hline(
        y=strike,
        line_dash="dash",
        line_color="#f5c542",
        annotation_text=f"Strike ${strike:,.2f}",
    )
    figure.update_layout(
        title="BTC 1-Minute Candles",
        height=410,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        yaxis_title="BTC price",
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)

with right_column:
    st.subheader("Model")
    st.metric("Above", f"{model['above'] * 100:.1f}%")
    st.metric("Below", f"{model['below'] * 100:.1f}%")
    st.metric("5-min momentum", f"{model['momentum_5m'] * 100:.3f}%")
    st.metric("1-min volatility", f"{model['volatility_1m'] * 100:.3f}%")

    if model["above"] > 0.60:
        st.success("Model leans above — not a guarantee.")
    elif model["above"] < 0.40:
        st.warning("Model leans below — not a guarantee.")
    else:
        st.info("Near coin-flip range.")

with st.expander("Setup details"):
    setup_details = pd.DataFrame(
        [
            {
                "Rule": "At least 0.10% from strike",
                "Pass": "Yes" if setup["far_enough"] else "No",
            },
            {
                "Rule": "5-minute momentum agrees",
                "Pass": "Yes" if setup["momentum_agrees"] else "No",
            },
            {
                "Rule": "Not near recent high/low",
                "Pass": "Yes" if not setup["near_barrier"] else "No",
            },
            {
                "Rule": "Between 5 and 8 minutes left",
                "Pass": "Yes" if setup["right_time"] else "No",
            },
        ]
    )
    st.dataframe(setup_details, use_container_width=True, hide_index=True)
    st.caption(
        f"Recent 5-minute high: ${setup['recent_high']:,.2f} | "
        f"Recent 5-minute low: ${setup['recent_low']:,.2f}"
    )

if kalshi_ticker.strip():
    st.subheader("Kalshi public market data")

    if kalshi_error:
        st.error(f"Could not load Kalshi data: {kalshi_error}")
    elif market:
        market_col1, market_col2, market_col3 = st.columns(3)
        market_col1.metric("Market", market.get("title", "Not provided"))
        market_col2.metric("YES bid", f"{market.get('yes_bid', 'N/A')}¢")
        market_col3.metric("YES ask", f"{market.get('yes_ask', 'N/A')}¢")

        with st.expander("Order-book data"):
            st.json(orderbook)

st.caption(
    "Analysis only, not financial or betting advice. The model can be wrong. "
    "Contract rules, settlement source, and timestamp decide the outcome."
)
