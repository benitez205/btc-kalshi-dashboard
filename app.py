import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="BTC Live Interval Monitor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
CHART_REFRESH_SECONDS = 30

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 0.9rem;
            padding-bottom: 0.9rem;
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
            font-size: 1.12rem;
        }
        h1 {
            font-size: 1.6rem;
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


def get_live_btc_price():
    response = requests.get(
        COINBASE_TICKER_URL,
        headers={"User-Agent": "btc-kalshi-dashboard/1.0"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "price": float(data["price"]),
        "time": pd.to_datetime(data["time"], utc=True),
    }


@st.cache_data(ttl=25, show_spinner=False)
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


@st.cache_data(ttl=10, show_spinner=False)
def get_kalshi_market(ticker):
    if not ticker.strip():
        return None, None, None

    try:
        market_response = requests.get(
            f"{KALSHI_BASE}/markets/{ticker.strip()}",
            timeout=10,
        )
        market_response.raise_for_status()
        market = market_response.json().get("market", market_response.json())

        book_response = requests.get(
            f"{KALSHI_BASE}/markets/{ticker.strip()}/orderbook",
            timeout=10,
        )
        book_response.raise_for_status()
        orderbook = book_response.json().get("orderbook", book_response.json())
        return market, orderbook, None
    except requests.RequestException as error:
        return None, None, str(error)


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
    return (100 - (100 / (1 + relative_strength))).fillna(50.0)


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


def calculate_chart_bot(df, strike, live_price):
    bot_df = df.copy()
    bot_df["ema_5"] = bot_df["close"].ewm(span=5, adjust=False).mean()
    bot_df["ema_12"] = bot_df["close"].ewm(span=12, adjust=False).mean()
    bot_df["rsi_14"] = calculate_rsi(bot_df["close"], period=14)
    bot_df["atr_5"] = calculate_atr(bot_df, period=5)

    current_price = live_price
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
    atr_multiple = distance_dollars / atr_5 if atr_5 > 0 else 0.0

    price_above_strike = current_price > strike
    price_below_strike = current_price < strike
    bullish_signals = 0
    bearish_signals = 0
    signal_rows = []

    if price_above_strike:
        signal_rows.append(("Price vs strike", "ABOVE", "Live price is above strike"))
    elif price_below_strike:
        signal_rows.append(("Price vs strike", "BELOW", "Live price is below strike"))
    else:
        signal_rows.append(("Price vs strike", "NEUTRAL", "Live price equals strike"))

    if ema_5 > ema_12:
        bullish_signals += 1
        signal_rows.append(("EMA trend", "BULLISH", "5 EMA is above 12 EMA"))
    elif ema_5 < ema_12:
        bearish_signals += 1
        signal_rows.append(("EMA trend", "BEARISH", "5 EMA is below 12 EMA"))
    else:
        signal_rows.append(("EMA trend", "NEUTRAL", "EMAs are equal"))

    if rsi_14 >= 55:
        bullish_signals += 1
        signal_rows.append(("RSI 14", "BULLISH", f"RSI is {rsi_14:.1f}"))
    elif rsi_14 <= 45:
        bearish_signals += 1
        signal_rows.append(("RSI 14", "BEARISH", f"RSI is {rsi_14:.1f}"))
    else:
        signal_rows.append(("RSI 14", "NEUTRAL", f"RSI is {rsi_14:.1f}"))

    if green_candles >= 2:
        bullish_signals += 1
        signal_rows.append(("Last 3 candles", "BULLISH", f"{green_candles} green candles"))
    elif red_candles >= 2:
        bearish_signals += 1
        signal_rows.append(("Last 3 candles", "BEARISH", f"{red_candles} red candles"))
    else:
        signal_rows.append(("Last 3 candles", "NEUTRAL", "Mixed candles"))

    if current_price > range_midpoint:
        bullish_signals += 1
        signal_rows.append(("15-minute range", "BULLISH", "Live price is above range midpoint"))
    elif current_price < range_midpoint:
        bearish_signals += 1
        signal_rows.append(("15-minute range", "BEARISH", "Live price is below range midpoint"))
    else:
        signal_rows.append(("15-minute range", "NEUTRAL", "Live price is at range midpoint"))

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

    bullish_pressure = bullish_signals >= 3
    bearish_pressure = bearish_signals >= 3
    reversal_watch = False
    reversal_reason = "No strong conflict between price location and momentum."

    if price_above_strike and bearish_pressure:
        reversal_watch = True
        reversal_reason = (
            "Price is above strike, but at least three short-term signals are bearish."
        )
    elif price_below_strike and bullish_pressure:
        reversal_watch = True
        reversal_reason = (
            "Price is below strike, but at least three short-term signals are bullish."
        )

    if atr_multiple < 1:
        label = "TOO CLOSE / NO CLEAR EDGE"
        confidence = "Low"
        summary = (
            "Price is within one recent ATR of the strike. Ordinary BTC volatility "
            "could change the above/below result."
        )
    elif reversal_watch and price_above_strike:
        label = "ABOVE — WEAKENING / REVERSAL WATCH"
        confidence = "Low"
        summary = reversal_reason
    elif reversal_watch and price_below_strike:
        label = "BELOW — WEAKENING / REVERSAL WATCH"
        confidence = "Low"
        summary = reversal_reason
    elif price_above_strike and bullish_pressure:
        label = "ABOVE — TREND CONFIRMED"
        confidence = "Medium"
        summary = "Price is above strike and at least three short-term signals are bullish."
    elif price_below_strike and bearish_pressure:
        label = "BELOW — TREND CONFIRMED"
        confidence = "Medium"
        summary = "Price is below strike and at least three short-term signals are bearish."
    else:
        label = "TOO CLOSE / NO CLEAR EDGE"
        confidence = "Low"
        summary = "Price location and short-term momentum do not agree strongly enough."

    return {
        "df": bot_df,
        "label": label,
        "confidence": confidence,
        "summary": summary,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "distance_dollars": distance_dollars,
        "ema_5": ema_5,
        "ema_12": ema_12,
        "rsi_14": rsi_14,
        "atr_5": atr_5,
        "atr_multiple": atr_multiple,
        "cushion_label": cushion_label,
        "range_high": range_high,
        "range_low": range_low,
        "reversal_watch": reversal_watch,
        "reversal_reason": reversal_reason,
        "signals": signal_rows,
    }


def calculate_budget_plan(
    bankroll,
    reserve_cash,
    risk_per_trade,
    buy_limit_cents,
    sell_limit_cents,
):
    available_for_orders = max(bankroll - reserve_cash, 0)
    buy_price = buy_limit_cents / 100
    sell_price = sell_limit_cents / 100

    contracts_by_risk = math.floor(risk_per_trade / buy_price) if buy_price > 0 else 0
    contracts_by_cash = math.floor(available_for_orders / buy_price) if buy_price > 0 else 0
    contracts_allowed = max(min(contracts_by_risk, contracts_by_cash), 0)

    maximum_cost = contracts_allowed * buy_price
    potential_sale_proceeds = contracts_allowed * sell_price
    potential_profit = potential_sale_proceeds - maximum_cost
    maximum_settlement_value = float(contracts_allowed)
    maximum_settlement_profit = maximum_settlement_value - maximum_cost

    warnings = []
    if reserve_cash >= bankroll:
        warnings.append("Reserve cash is equal to or larger than the bankroll, so no order budget remains.")
    if risk_per_trade > available_for_orders:
        warnings.append("Risk cap exceeds cash available after your reserve.")
    if buy_limit_cents <= 0 or buy_limit_cents >= 100:
        warnings.append("Buy limit must be between 1¢ and 99¢.")
    if sell_limit_cents <= 0 or sell_limit_cents >= 100:
        warnings.append("Sell limit must be between 1¢ and 99¢.")
    if sell_limit_cents <= buy_limit_cents:
        warnings.append("Your sell limit is not above your buy limit, so it does not lock in a gross gain.")
    if contracts_allowed == 0 and buy_price > 0:
        warnings.append("Risk cap or available cash is too low to purchase one contract at this limit.")
    if risk_per_trade > bankroll * 0.05:
        warnings.append("Risk per trade exceeds 5% of the bankroll. Consider a smaller fixed cap.")

    return {
        "available_for_orders": available_for_orders,
        "contracts_allowed": contracts_allowed,
        "maximum_cost": maximum_cost,
        "potential_sale_proceeds": potential_sale_proceeds,
        "potential_profit": potential_profit,
        "maximum_settlement_value": maximum_settlement_value,
        "maximum_settlement_profit": maximum_settlement_profit,
        "warnings": warnings,
    }


st.title("₿ BTC Live 15-Minute Monitor")
st.caption(
    "Live Coinbase quote + 1-minute chart research. Read-only; it cannot place, close, or modify Kalshi orders."
)

with st.sidebar:
    st.header("Contract inputs")
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
    refresh_seconds = st.selectbox(
        "Live panel interval",
        options=[1, 2, 5, 10],
        index=1,
        help="The live dashboard panel refreshes automatically at this interval.",
    )
    st.caption(f"Candles and indicators are cached for {CHART_REFRESH_SECONDS} seconds.")

    st.divider()
    st.header("Budget & limit planner")
    bankroll = st.number_input(
        "Current bankroll ($)",
        min_value=0.0,
        value=69.00,
        step=1.00,
        key="bankroll",
    )
    reserve_cash = st.number_input(
        "Cash reserve ($)",
        min_value=0.0,
        value=24.00,
        step=1.00,
        key="reserve_cash",
    )
    risk_per_trade = st.number_input(
        "Maximum risk per trade ($)",
        min_value=0.0,
        value=2.00,
        step=0.50,
        key="risk_per_trade",
    )
    contract_side = st.selectbox(
        "Contract side you are evaluating",
        options=["YES", "NO"],
        help="This planner calculates cash limits only. It does not choose a side for you.",
    )
    buy_limit_cents = st.number_input(
        "Your maximum buy limit (¢)",
        min_value=1,
        max_value=99,
        value=40,
        step=1,
        key="buy_limit_cents",
    )
    sell_limit_cents = st.number_input(
        "Your planned take-profit limit (¢)",
        min_value=1,
        max_value=99,
        value=55,
        step=1,
        key="sell_limit_cents",
    )

try:
    settlement_time = datetime.fromisoformat(
        settlement_input.replace("Z", "+00:00")
    )
except ValueError:
    st.error("Use UTC format like: 2026-09-22T03:30:00Z")
    st.stop()

budget_plan = calculate_budget_plan(
    bankroll,
    reserve_cash,
    risk_per_trade,
    buy_limit_cents,
    sell_limit_cents,
)


@st.fragment(run_every=f"{refresh_seconds}s")
def live_dashboard():
    try:
        live_quote = get_live_btc_price()
        candles = get_btc_candles()
    except (requests.RequestException, ValueError) as error:
        st.error(f"Could not load live market data: {error}")
        return

    if len(candles) < 31:
        st.error("Not enough BTC candles returned. Please wait for the next refresh.")
        return

    now_utc = datetime.now(timezone.utc)
    minutes_left = max((settlement_time - now_utc).total_seconds() / 60, 0)
    chart_bot = calculate_chart_bot(candles, strike, live_quote["price"])

    spot_col, strike_col, distance_col, time_col, quote_col = st.columns(5)
    spot_col.metric("Live BTC spot", f"${live_quote['price']:,.2f}")
    strike_col.metric("Strike", f"${strike:,.2f}")
    distance_col.metric("Distance", f"${live_quote['price'] - strike:,.2f}")
    time_col.metric("Minutes left", f"{minutes_left:.2f}")
    quote_col.metric("Quote time", live_quote["time"].strftime("%H:%M:%S UTC"))

    st.caption(
        f"Live panel updates every {refresh_seconds} second(s). "
        f"Candles, EMA, RSI, and ATR refresh at most every {CHART_REFRESH_SECONDS} seconds."
    )

    st.divider()
    st.subheader("Chart research bot")

    bot_col1, bot_col2, bot_col3, bot_col4, bot_col5 = st.columns(5)
    bot_col1.metric("Bot result", chart_bot["label"])
    bot_col2.metric("Confidence", chart_bot["confidence"])
    bot_col3.metric("Bullish signals", f"{chart_bot['bullish_signals']} / 4")
    bot_col4.metric("Bearish signals", f"{chart_bot['bearish_signals']} / 4")
    bot_col5.metric("ATR cushion", chart_bot["cushion_label"])
    st.caption(chart_bot["summary"])

    if chart_bot["reversal_watch"]:
        st.warning(
            f"REVERSAL WATCH: {chart_bot['reversal_reason']} "
            "This is a research warning, not a recommendation to increase position size."
        )
    elif chart_bot["atr_multiple"] < 1:
        st.info(
            "STRIKE RISK: Price is close to the strike relative to recent volatility. "
            "A normal move could change the above/below result."
        )
    else:
        st.success("No strong conflict between current price location and short-term momentum.")

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
        figure.add_hline(
            y=live_quote["price"],
            line_dash="dot",
            line_color="#00cc96",
            annotation_text=f"Live ${live_quote['price']:,.2f}",
        )
        figure.update_layout(
            title="BTC 1-Minute Candles with Live Price and EMA Lines",
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
            yaxis_title="BTC price",
        )
        st.plotly_chart(figure, use_container_width=True)

    with right_column:
        st.subheader("Signal interpretation")
        st.metric("Signal label", chart_bot["label"])
        st.metric("Distance vs ATR", f"{chart_bot['atr_multiple']:.2f}×")
        st.write(
            "The dashboard is informational only. It does not recommend adding money, "
            "opening a trade, closing a trade, or setting an auto-sell price."
        )


live_dashboard()

st.divider()
st.subheader("Budget & limit planner")
st.caption(
    "Enter your own limits first. This calculator checks size and cash exposure; it does not generate a trade, buy price, or sell price."
)

plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)
plan_col1.metric("Side evaluated", contract_side)
plan_col2.metric("Available after reserve", f"${budget_plan['available_for_orders']:,.2f}")
plan_col3.metric("Maximum contracts", str(budget_plan["contracts_allowed"]))
plan_col4.metric("Maximum order cost", f"${budget_plan['maximum_cost']:,.2f}")

plan_col5, plan_col6, plan_col7, plan_col8 = st.columns(4)
plan_col5.metric("Buy limit", f"{buy_limit_cents}¢")
plan_col6.metric("Take-profit limit", f"{sell_limit_cents}¢")
plan_col7.metric("Proceeds at target", f"${budget_plan['potential_sale_proceeds']:,.2f}")
plan_col8.metric("Gross target profit", f"${budget_plan['potential_profit']:,.2f}")

st.caption(
    f"If {budget_plan['contracts_allowed']} contract(s) settle in your favor instead of being sold early, "
    f"the maximum gross settlement profit would be ${budget_plan['maximum_settlement_profit']:,.2f} before fees."
)

if budget_plan["warnings"]:
    for warning in budget_plan["warnings"]:
        st.warning(warning)
else:
    st.success(
        "Planner check passed: proposed order stays within your stated trade-risk cap and protects the stated cash reserve."
    )

with st.expander("How this planner calculates size"):
    st.code(
        "maximum contracts = floor(min(risk per trade, bankroll − reserve) ÷ buy limit price)",
        language="text",
    )
    st.write(
        "It assumes a binary contract bought at your entered limit. It shows gross figures only; "
        "fees, partial fills, and unfilled limit orders can change real results."
    )

if kalshi_ticker.strip():
    st.divider()
    st.subheader("Kalshi public market data")
    market, orderbook, kalshi_error = get_kalshi_market(kalshi_ticker)

    if kalshi_error:
        st.error(f"Could not load Kalshi data: {kalshi_error}")
    elif market:
        market_col1, market_col2, market_col3 = st.columns(3)
        market_col1.metric("Market", market.get("title", "Not provided"))
        market_col2.metric("YES bid", f"{market.get('yes_bid', 'N/A')}¢")
        market_col3.metric("YES ask", f"{market.get('yes_ask', 'N/A')}¢")

        with st.expander("Show public Kalshi order-book data"):
            st.json(orderbook)

st.caption(
    "Research only, not financial or betting advice. A Coinbase quote can differ "
    "from the source and timestamp used for Kalshi settlement. Limit orders may not fill "
    "or may fill only partially."
)


