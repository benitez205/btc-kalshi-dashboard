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
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
CHART_REFRESH_SECONDS = 30
STALE_QUOTE_SECONDS = 15
STALE_CANDLE_SECONDS = 120
WIDE_SPREAD_CENTS = 4
EXTREME_LOW_CENTS = 10
EXTREME_HIGH_CENTS = 90
LOSS_STREAK_LIMIT = 2
SESSION_LOSS_LIMIT = -5.00
TRADE_LOG_COLUMNS = [
    "logged_at_utc", "settlement_time_utc", "strike", "snapshot_btc_price",
    "snapshot_quote_age_seconds", "snapshot_candle_age_seconds", "snapshot_data_status",
    "snapshot_bot_label", "snapshot_confidence", "snapshot_market_state",
    "snapshot_bullish_signals", "snapshot_bearish_signals", "snapshot_atr_multiple",
    "snapshot_ema_5", "snapshot_ema_12", "snapshot_rsi_14", "snapshot_range_high",
    "snapshot_range_low", "snapshot_yes_bid", "snapshot_yes_ask", "snapshot_no_bid",
    "snapshot_no_ask", "snapshot_selected_bid", "snapshot_selected_ask",
    "snapshot_selected_spread_cents", "snapshot_book_top3_bid_contracts",
    "snapshot_book_top3_ask_contracts", "side", "contracts", "entry_cents", "exit_cents",
    "fees", "cost", "proceeds", "net_pnl", "budget_cap", "within_budget_cap",
    "partial_fill", "notes",
]


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
        "retrieved_at": datetime.now(timezone.utc),
    }


@st.cache_data(ttl=25, show_spinner=False)
def get_btc_candles(limit=90):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - pd.Timedelta(minutes=limit)
    response = requests.get(
        COINBASE_CANDLES_URL,
        params={"granularity": 60, "start": start_time.isoformat(), "end": end_time.isoformat()},
        headers={"User-Agent": "btc-kalshi-dashboard/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise ValueError("Coinbase returned no BTC candle data.")
    df = pd.DataFrame(rows, columns=["open_time", "low", "high", "open", "close", "volume"])
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])
    df["time"] = pd.to_datetime(df["open_time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


@st.cache_data(ttl=10, show_spinner=False)
def get_kalshi_market(ticker):
    if not ticker.strip():
        return None, None, None
    try:
        market_response = requests.get(f"{KALSHI_BASE}/markets/{ticker.strip()}", timeout=10)
        market_response.raise_for_status()
        market = market_response.json().get("market", market_response.json())
        book_response = requests.get(f"{KALSHI_BASE}/markets/{ticker.strip()}/orderbook", timeout=10)
        book_response.raise_for_status()
        orderbook = book_response.json().get("orderbook", book_response.json())
        return market, orderbook, None
    except requests.RequestException as error:
        return None, None, str(error)


def extract_market_prices(market, side):
    if not market:
        return None, None
    if side == "YES":
        return safe_int(market.get("yes_bid")), safe_int(market.get("yes_ask"))
    return safe_int(market.get("no_bid")), safe_int(market.get("no_ask"))


def parse_orderbook_levels(orderbook, side):
    if not orderbook:
        return [], []
    yes_levels = orderbook.get("yes", []) if isinstance(orderbook, dict) else []
    no_levels = orderbook.get("no", []) if isinstance(orderbook, dict) else []
    raw_levels = yes_levels if side == "YES" else no_levels
    bid_levels = []
    ask_levels = []
    for level in raw_levels:
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            price, quantity = safe_int(level[0]), safe_int(level[1])
        elif isinstance(level, dict):
            price = safe_int(level.get("price") or level.get("price_cents"))
            quantity = safe_int(level.get("quantity") or level.get("count") or level.get("size"))
        else:
            continue
        if price is not None and quantity is not None:
            bid_levels.append((price, quantity))
    bid_levels = sorted(bid_levels, key=lambda item: item[0], reverse=True)
    if side == "YES":
        for price, quantity in no_levels:
            no_price, no_quantity = safe_int(price), safe_int(quantity)
            if no_price is not None and no_quantity is not None:
                ask_levels.append((100 - no_price, no_quantity))
    else:
        for price, quantity in yes_levels:
            yes_price, yes_quantity = safe_int(price), safe_int(quantity)
            if yes_price is not None and yes_quantity is not None:
                ask_levels.append((100 - yes_price, yes_quantity))
    ask_levels = sorted(ask_levels, key=lambda item: item[0])
    return bid_levels[:3], ask_levels[:3]


def calculate_rsi(closes, period=14):
    deltas = closes.diff()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
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


def calculate_market_state(df, atr_5):
    recent = df.tail(15)
    range_size = float(recent["high"].max() - recent["low"].min())
    net_move = abs(float(recent["close"].iloc[-1] - recent["close"].iloc[0]))
    efficiency = net_move / range_size if range_size > 0 else 0.0
    ema_gap = abs(float(df["ema_5"].iloc[-1] - df["ema_12"].iloc[-1]))
    normalized_ema_gap = ema_gap / atr_5 if atr_5 > 0 else 0.0
    if efficiency >= 0.60 and normalized_ema_gap >= 0.25:
        return "TRENDING", efficiency, normalized_ema_gap
    if efficiency <= 0.35 and normalized_ema_gap <= 0.25:
        return "CHOPPY", efficiency, normalized_ema_gap
    return "UNCLEAR", efficiency, normalized_ema_gap


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
    market_state, trend_efficiency, normalized_ema_gap = calculate_market_state(bot_df, atr_5)
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
    signal_rows.append(("Market state", market_state, f"Trend efficiency {trend_efficiency:.2f}; EMA-gap/ATR {normalized_ema_gap:.2f}"))

    bullish_pressure = bullish_signals >= 3
    bearish_pressure = bearish_signals >= 3
    reversal_watch = False
    reversal_reason = "No strong conflict between price location and momentum."
    if price_above_strike and bearish_pressure:
        reversal_watch = True
        reversal_reason = "Price is above strike, but at least three short-term signals are bearish."
    elif price_below_strike and bullish_pressure:
        reversal_watch = True
        reversal_reason = "Price is below strike, but at least three short-term signals are bullish."

    if atr_multiple < 1:
        label = "TOO CLOSE / NO CLEAR EDGE"
        confidence = "Low"
        summary = "Price is within one recent ATR of the strike. Ordinary BTC volatility could change the above/below result."
    elif market_state == "CHOPPY":
        label = "CHOPPY / NO CLEAR EDGE"
        confidence = "Low"
        summary = "Recent movement is range-like rather than directional. Treat momentum labels cautiously."
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
        "market_state": market_state,
        "trend_efficiency": trend_efficiency,
        "normalized_ema_gap": normalized_ema_gap,
        "reversal_watch": reversal_watch,
        "reversal_reason": reversal_reason,
        "signals": signal_rows,
    }


def calculate_budget_plan(bankroll, reserve_cash, risk_per_trade, buy_limit_cents, sell_limit_cents):
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


def new_empty_trade_log():
    return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def calculate_session_metrics(trade_log):
    if trade_log.empty:
        return {"count": 0, "pnl": 0.0, "loss_streak": 0, "win_rate": None}
    working = trade_log.copy()
    working["logged_at_utc"] = pd.to_datetime(working["logged_at_utc"], utc=True, errors="coerce")
    working["net_pnl"] = pd.to_numeric(working["net_pnl"], errors="coerce").fillna(0.0)
    today_utc = datetime.now(timezone.utc).date()
    today = working[working["logged_at_utc"].dt.date == today_utc].sort_values("logged_at_utc")
    pnl = float(today["net_pnl"].sum()) if not today.empty else 0.0
    loss_streak = 0
    for value in reversed(today["net_pnl"].tolist()):
        if value < 0:
            loss_streak += 1
        else:
            break
    win_rate = float((today["net_pnl"] > 0).mean() * 100) if not today.empty else None
    return {"count": len(today), "pnl": pnl, "loss_streak": loss_streak, "win_rate": win_rate}


def data_status(quote_age_seconds, candle_age_seconds):
    issues = []
    if quote_age_seconds is not None and quote_age_seconds > STALE_QUOTE_SECONDS:
        issues.append("Coinbase quote is stale")
    if candle_age_seconds is not None and candle_age_seconds > STALE_CANDLE_SECONDS:
        issues.append("latest candle is stale")
    return "CHECK DATA" if issues else "CURRENT", issues


def create_trade_snapshot(settlement_time, strike, live_quote, chart_bot, quote_age_seconds, candle_age_seconds, market_context):
    status, _ = data_status(quote_age_seconds, candle_age_seconds)
    return {
        "settlement_time_utc": settlement_time.isoformat(),
        "strike": round(float(strike), 2),
        "snapshot_btc_price": round(float(live_quote["price"]), 2),
        "snapshot_quote_age_seconds": round(float(quote_age_seconds), 1),
        "snapshot_candle_age_seconds": round(float(candle_age_seconds), 1),
        "snapshot_data_status": status,
        "snapshot_bot_label": chart_bot["label"],
        "snapshot_confidence": chart_bot["confidence"],
        "snapshot_market_state": chart_bot["market_state"],
        "snapshot_bullish_signals": chart_bot["bullish_signals"],
        "snapshot_bearish_signals": chart_bot["bearish_signals"],
        "snapshot_atr_multiple": round(float(chart_bot["atr_multiple"]), 3),
        "snapshot_ema_5": round(float(chart_bot["ema_5"]), 2),
        "snapshot_ema_12": round(float(chart_bot["ema_12"]), 2),
        "snapshot_rsi_14": round(float(chart_bot["rsi_14"]), 2),
        "snapshot_range_high": round(float(chart_bot["range_high"]), 2),
        "snapshot_range_low": round(float(chart_bot["range_low"]), 2),
        "snapshot_yes_bid": market_context.get("yes_bid"),
        "snapshot_yes_ask": market_context.get("yes_ask"),
        "snapshot_no_bid": market_context.get("no_bid"),
        "snapshot_no_ask": market_context.get("no_ask"),
        "snapshot_selected_bid": market_context.get("selected_bid"),
        "snapshot_selected_ask": market_context.get("selected_ask"),
        "snapshot_selected_spread_cents": market_context.get("spread"),
        "snapshot_book_top3_bid_contracts": market_context.get("top3_bid_contracts"),
        "snapshot_book_top3_ask_contracts": market_context.get("top3_ask_contracts"),
    }


def add_trade_to_log(snapshot, side, contracts, entry_cents, exit_cents, fees, partial_fill, notes, budget_cap):
    cost = contracts * (entry_cents / 100)
    proceeds = contracts * (exit_cents / 100)
    net_pnl = proceeds - cost - fees
    row = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **snapshot,
        "side": side,
        "contracts": contracts,
        "entry_cents": entry_cents,
        "exit_cents": exit_cents,
        "fees": round(fees, 2),
        "cost": round(cost, 2),
        "proceeds": round(proceeds, 2),
        "net_pnl": round(net_pnl, 2),
        "budget_cap": round(budget_cap, 2),
        "within_budget_cap": cost <= budget_cap,
        "partial_fill": partial_fill,
        "notes": notes,
    }
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, pd.DataFrame([row])], ignore_index=True)


if "trade_log" not in st.session_state:
    st.session_state.trade_log = new_empty_trade_log()
if "latest_snapshot" not in st.session_state:
    st.session_state.latest_snapshot = None
if "auto_snapshot_key" not in st.session_state:
    st.session_state.auto_snapshot_key = None

st.title("₿ BTC Live 15-Minute Monitor")
st.caption("Live Coinbase quote + 1-minute chart research. Read-only; it cannot place, close, or modify Kalshi orders.")

with st.sidebar:
    st.header("Contract inputs")
    kalshi_ticker = st.text_input("Kalshi market ticker (optional)", placeholder="KXBTC...")
    strike = st.number_input("BTC threshold / strike", min_value=1.0, value=85513.00, step=0.01)
    settlement_input = st.text_input("Settlement time (UTC)", value="2026-09-22T03:30:00Z", help="Example: 10:30 PM CDT = 03:30 UTC the following day.")
    refresh_seconds = st.selectbox("Live panel interval", options=[1, 2, 5, 10], index=1, help="The live dashboard panel refreshes automatically at this interval.")
    st.caption(f"Candles and indicators are cached for {CHART_REFRESH_SECONDS} seconds.")
    st.divider()
    st.header("Budget & limit planner")
    bankroll = st.number_input("Current bankroll ($)", min_value=0.0, value=69.00, step=1.00)
    reserve_cash = st.number_input("Cash reserve ($)", min_value=0.0, value=24.00, step=1.00)
    risk_per_trade = st.number_input("Maximum risk per trade ($)", min_value=0.0, value=2.00, step=0.50)
    contract_side = st.selectbox("Contract side you are evaluating", options=["YES", "NO"])
    buy_limit_cents = st.number_input("Your maximum buy limit (¢)", min_value=1, max_value=99, value=40, step=1)
    sell_limit_cents = st.number_input("Your planned take-profit limit (¢)", min_value=1, max_value=99, value=55, step=1)

try:
    settlement_time = datetime.fromisoformat(settlement_input.replace("Z", "+00:00"))
except ValueError:
    st.error("Use UTC format like: 2026-09-22T03:30:00Z")
    st.stop()

budget_plan = calculate_budget_plan(bankroll, reserve_cash, risk_per_trade, buy_limit_cents, sell_limit_cents)

market, orderbook, kalshi_error = (None, None, None)
market_context = {
    "yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None,
    "selected_bid": None, "selected_ask": None, "spread": None,
    "top3_bid_contracts": None, "top3_ask_contracts": None,
}
if kalshi_ticker.strip():
    market, orderbook, kalshi_error = get_kalshi_market(kalshi_ticker)
    if market:
        market_context["yes_bid"], market_context["yes_ask"] = extract_market_prices(market, "YES")
        market_context["no_bid"], market_context["no_ask"] = extract_market_prices(market, "NO")
        selected_bid, selected_ask = extract_market_prices(market, contract_side)
        market_context["selected_bid"] = selected_bid
        market_context["selected_ask"] = selected_ask
        if selected_bid is not None and selected_ask is not None:
            market_context["spread"] = selected_ask - selected_bid
        bid_levels, ask_levels = parse_orderbook_levels(orderbook, contract_side)
        market_context["top3_bid_contracts"] = sum(quantity for _, quantity in bid_levels) if bid_levels else 0
        market_context["top3_ask_contracts"] = sum(quantity for _, quantity in ask_levels) if ask_levels else 0


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
    seconds_left = max((settlement_time - now_utc).total_seconds(), 0)
    minutes_left = seconds_left / 60
    quote_age_seconds = max((now_utc - live_quote["time"].to_pydatetime()).total_seconds(), 0)
    candle_age_seconds = max((now_utc - candles["time"].iloc[-1].to_pydatetime()).total_seconds(), 0)
    status, data_issues = data_status(quote_age_seconds, candle_age_seconds)
    chart_bot = calculate_chart_bot(candles, strike, live_quote["price"])
    current_snapshot = create_trade_snapshot(
        settlement_time, strike, live_quote, chart_bot, quote_age_seconds,
        candle_age_seconds, market_context,
    )
    st.session_state.latest_snapshot = current_snapshot
    snapshot_key = f"{settlement_time.isoformat()}|{strike:.2f}"
    if seconds_left <= 0 and st.session_state.auto_snapshot_key != snapshot_key:
        st.session_state.auto_snapshot_key = snapshot_key
        st.session_state.latest_snapshot = current_snapshot

    spot_col, strike_col, distance_col, time_col, quote_col = st.columns(5)
    spot_col.metric("Live BTC spot", f"${live_quote['price']:,.2f}")
    strike_col.metric("Strike", f"${strike:,.2f}")
    distance_col.metric("Distance", f"${live_quote['price'] - strike:,.2f}")
    time_col.metric("Minutes left", f"{minutes_left:.2f}")
    quote_col.metric("Quote age", f"{quote_age_seconds:.1f}s")
    st.caption(f"Live panel updates every {refresh_seconds} second(s). Candles, EMA, RSI, and ATR refresh at most every {CHART_REFRESH_SECONDS} seconds.")

    st.divider()
    st.subheader("Data quality")
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    quality_col1.metric("Data status", status)
    quality_col2.metric("Coinbase quote age", f"{quote_age_seconds:.1f}s")
    quality_col3.metric("Latest candle age", f"{candle_age_seconds:.1f}s")
    if data_issues:
        st.warning("DATA WARNING: " + "; ".join(data_issues) + ". Verify the displayed data before relying on it.")
    else:
        st.success("Data-age check passed using the dashboard thresholds.")

    st.divider()
    st.subheader("Chart research bot")
    bot_col1, bot_col2, bot_col3, bot_col4, bot_col5 = st.columns(5)
    bot_col1.metric("Bot result", chart_bot["label"])
    bot_col2.metric("Confidence", chart_bot["confidence"])
    bot_col3.metric("Market state", chart_bot["market_state"])
    bot_col4.metric("Bullish signals", f"{chart_bot['bullish_signals']} / 4")
    bot_col5.metric("ATR cushion", chart_bot["cushion_label"])
    st.caption(chart_bot["summary"])

    if chart_bot["market_state"] == "CHOPPY":
        st.warning("MARKET STATE: Recent price action is choppy. Trend and momentum signals can be less reliable in this regime.")
    if chart_bot["reversal_watch"]:
        st.warning(f"REVERSAL WATCH: {chart_bot['reversal_reason']} This is a research warning, not a recommendation to increase position size.")
    elif chart_bot["atr_multiple"] < 1:
        st.info("STRIKE RISK: Price is close to the strike relative to recent volatility. A normal move could change the above/below result.")
    else:
        st.success("No strong conflict between current price location and short-term momentum.")

    with st.expander("Show chart-bot signals"):
        signal_df = pd.DataFrame(chart_bot["signals"], columns=["Indicator", "Signal", "Explanation"])
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
        detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
        detail_col1.metric("EMA 5", f"${chart_bot['ema_5']:,.2f}")
        detail_col2.metric("EMA 12", f"${chart_bot['ema_12']:,.2f}")
        detail_col3.metric("RSI 14", f"{chart_bot['rsi_14']:.1f}")
        detail_col4.metric("5-min ATR", f"${chart_bot['atr_5']:,.2f}")
        st.caption(
            f"Distance from strike: ${chart_bot['distance_dollars']:,.2f} | "
            f"ATR multiple: {chart_bot['atr_multiple']:.2f}× | "
            f"15-minute range: ${chart_bot['range_low']:,.2f} to ${chart_bot['range_high']:,.2f} | "
            f"Trend efficiency: {chart_bot['trend_efficiency']:.2f}"
        )

    left_column, right_column = st.columns([2, 1])
    with left_column:
        chart_df = chart_bot["df"]
        figure = go.Figure()
        figure.add_trace(go.Candlestick(
            x=chart_df["time"], open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"], close=chart_df["close"], name="BTC/USD",
        ))
        figure.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["ema_5"], mode="lines", name="EMA 5", line=dict(color="#00cc96", width=1.5)))
        figure.add_trace(go.Scatter(x=chart_df["time"], y=chart_df["ema_12"], mode="lines", name="EMA 12", line=dict(color="#636efa", width=1.5)))
        figure.add_hline(y=strike, line_dash="dash", line_color="#f5c542", annotation_text=f"Strike ${strike:,.2f}")
        figure.add_hline(y=live_quote["price"], line_dash="dot", line_color="#00cc96", annotation_text=f"Live ${live_quote['price']:,.2f}")
        figure.update_layout(
            title="BTC 1-Minute Candles with Live Price and EMA Lines", height=480,
            margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False,
            yaxis_title="BTC price",
        )
        st.plotly_chart(figure, use_container_width=True)
        if seconds_left <= 0:
            st.info("Clock has reached zero. The most recent dashboard data is saved as the trade-log snapshot. Confirm settlement using the market's official rules and source.")

    with right_column:
        st.subheader("Signal interpretation")
        st.metric("Signal label", chart_bot["label"])
        st.metric("Market state", chart_bot["market_state"])
        st.metric("Distance vs ATR", f"{chart_bot['atr_multiple']:.2f}×")
        st.write("The dashboard is informational only. It does not recommend adding money, opening a trade, closing a trade, or setting an auto-sell price.")


live_dashboard()

st.divider()
st.subheader("Session guardrails")
st.caption("Displays warnings from the completed-trade log only. It does not block access, submit orders, or change any account setting.")
session_metrics = calculate_session_metrics(st.session_state.trade_log)
guard_col1, guard_col2, guard_col3, guard_col4 = st.columns(4)
guard_col1.metric("Today’s logged trades", str(session_metrics["count"]))
guard_col2.metric("Today’s net P&L", f"${session_metrics['pnl']:,.2f}")
guard_col3.metric("Consecutive losses", str(session_metrics["loss_streak"]))
guard_col4.metric("Guardrail limits", f"{LOSS_STREAK_LIMIT} losses / ${abs(SESSION_LOSS_LIMIT):.2f}")
if session_metrics["loss_streak"] >= LOSS_STREAK_LIMIT or session_metrics["pnl"] <= SESSION_LOSS_LIMIT:
    reasons = []
    if session_metrics["loss_streak"] >= LOSS_STREAK_LIMIT:
        reasons.append(f"{session_metrics['loss_streak']} consecutive logged losses")
    if session_metrics["pnl"] <= SESSION_LOSS_LIMIT:
        reasons.append(f"today’s logged P&L is ${session_metrics['pnl']:,.2f}")
    st.error("SESSION GUARDRAIL: " + " and ".join(reasons) + ". Pause and review the logged conditions; this dashboard will not place or prevent any order.")
else:
    st.success("Session guardrail has not been triggered by the completed trades logged today.")

st.divider()
st.subheader("Budget & limit planner")
st.caption("Enter your own limits first. This calculator checks size and cash exposure; it does not generate a trade, buy price, or sell price.")
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
if budget_plan["warnings"]:
    for warning in budget_plan["warnings"]:
        st.warning(warning)
else:
    st.success("Planner check passed: proposed order stays within your stated trade-risk cap and protects the stated cash reserve.")

st.divider()
st.subheader("Live market context")
st.caption("Read-only comparison of your manual limits with the public Kalshi market. It does not generate trade instructions or submit orders.")

if not kalshi_ticker.strip():
    st.info("Enter a Kalshi market ticker in the sidebar to display live bid/ask context.")
elif kalshi_error:
    st.error(f"Could not load Kalshi market context: {kalshi_error}")
elif market:
    live_bid = market_context["selected_bid"]
    live_ask = market_context["selected_ask"]
    spread = market_context["spread"]
    context_col1, context_col2, context_col3, context_col4 = st.columns(4)
    context_col1.metric(f"Live {contract_side} bid", f"{live_bid}¢" if live_bid is not None else "N/A")
    context_col2.metric(f"Live {contract_side} ask", f"{live_ask}¢" if live_ask is not None else "N/A")
    context_col3.metric("Bid / ask spread", f"{spread}¢" if spread is not None else "N/A")
    context_col4.metric("Your manual buy ceiling", f"{buy_limit_cents}¢")

    depth_col1, depth_col2, depth_col3 = st.columns(3)
    depth_col1.metric("Top 3 bid depth", str(market_context["top3_bid_contracts"]) if market_context["top3_bid_contracts"] is not None else "N/A")
    depth_col2.metric("Top 3 ask depth", str(market_context["top3_ask_contracts"]) if market_context["top3_ask_contracts"] is not None else "N/A")
    depth_col3.metric("Pricing status", "WIDE SPREAD" if spread is not None and spread >= WIDE_SPREAD_CENTS else "NORMAL / UNKNOWN")

    if spread is not None and spread >= WIDE_SPREAD_CENTS:
        st.warning(f"SPREAD WARNING: The displayed {contract_side} spread is {spread}¢, at or above the {WIDE_SPREAD_CENTS}¢ dashboard warning threshold. Wide spreads can make apparent value disappear after execution.")
    if live_ask is not None and (live_ask < EXTREME_LOW_CENTS or live_ask > EXTREME_HIGH_CENTS):
        st.warning(f"EXTREME-PRICE WARNING: The displayed {contract_side} ask is {live_ask}¢. Prices below {EXTREME_LOW_CENTS}¢ or above {EXTREME_HIGH_CENTS}¢ can have asymmetric payoff and execution risk.")
    if live_bid is not None and (live_bid < EXTREME_LOW_CENTS or live_bid > EXTREME_HIGH_CENTS):
        st.info(f"The displayed {contract_side} bid is also in an extreme-price range: {live_bid}¢.")

    if live_ask is None:
        st.info("No current ask was returned for the selected side. Check the order book and market status.")
    elif live_ask <= buy_limit_cents:
        st.success(f"Price comparison only: the displayed {contract_side} ask ({live_ask}¢) is at or below your manual ceiling ({buy_limit_cents}¢). Review the contract rules, chart context, and available size yourself.")
    else:
        difference = live_ask - buy_limit_cents
        st.info(f"Price comparison only: the displayed {contract_side} ask is {difference}¢ above your manual ceiling. A limit order at your ceiling would generally rest unless a seller matches it.")

    sell_col1, sell_col2, sell_col3 = st.columns(3)
    sell_col1.metric("Your manual sell target", f"{sell_limit_cents}¢")
    sell_col2.metric(f"Live {contract_side} bid", f"{live_bid}¢" if live_bid is not None else "N/A")
    target_distance = sell_limit_cents - live_bid if live_bid is not None else None
    sell_col3.metric("Distance to target", f"{target_distance}¢" if target_distance is not None else "N/A")
    if live_bid is not None:
        if live_bid >= sell_limit_cents:
            st.success("Price comparison only: the displayed bid is at or above your manual sell target.")
        else:
            st.caption("The displayed bid is below your manual sell target. A limit sale at your target may remain resting until a buyer matches it.")

    with st.expander("Show public Kalshi order-book data"):
        st.json(orderbook)

st.divider()
st.subheader("Trade log")
st.caption("When the clock reaches zero, the dashboard keeps the latest market snapshot. Enter the actual fills manually, then download the CSV after your session.")
if st.session_state.latest_snapshot:
    snapshot = st.session_state.latest_snapshot
    snapshot_col1, snapshot_col2, snapshot_col3, snapshot_col4 = st.columns(4)
    snapshot_col1.metric("Snapshot BTC", f"${snapshot['snapshot_btc_price']:,.2f}")
    snapshot_col2.metric("Snapshot label", snapshot["snapshot_bot_label"])
    snapshot_col3.metric("Snapshot state", snapshot["snapshot_market_state"])
    snapshot_col4.metric("Snapshot spread", f"{snapshot['snapshot_selected_spread_cents']}¢" if snapshot["snapshot_selected_spread_cents"] is not None else "N/A")
else:
    st.info("Waiting for the live dashboard to collect the first trade-log snapshot.")

with st.form("trade_log_form", clear_on_submit=True):
    st.write("Add a completed trade")
    form_col1, form_col2, form_col3, form_col4 = st.columns(4)
    logged_side = form_col1.selectbox("Side traded", options=["YES", "NO"])
    logged_contracts = form_col2.number_input("Contracts", min_value=1, value=1, step=1)
    logged_entry = form_col3.number_input("Actual entry (¢)", min_value=1, max_value=99, value=40, step=1)
    logged_exit = form_col4.number_input("Actual exit / settlement (¢)", min_value=0, max_value=100, value=55, step=1)
    logged_fees = st.number_input("Total fees ($)", min_value=0.0, value=0.00, step=0.01)
    partial_fill = st.checkbox("Partial fill or execution issue")
    logged_notes = st.text_area("Notes (optional)", placeholder="Example: Followed plan; sell limit filled before settlement.")
    save_trade = st.form_submit_button("Save completed trade")

if save_trade:
    if st.session_state.latest_snapshot is None:
        st.error("No market snapshot is available yet. Wait for the live panel to load.")
    else:
        add_trade_to_log(
            st.session_state.latest_snapshot, logged_side, int(logged_contracts),
            int(logged_entry), int(logged_exit), float(logged_fees), partial_fill,
            logged_notes, risk_per_trade,
        )
        st.success("Trade saved to this browser session's log.")

trade_log = st.session_state.trade_log.copy()
if not trade_log.empty:
    trade_log["net_pnl"] = pd.to_numeric(trade_log["net_pnl"], errors="coerce")
    log_col1, log_col2, log_col3, log_col4 = st.columns(4)
    log_col1.metric("Completed trades", str(len(trade_log)))
    log_col2.metric("Net P&L", f"${trade_log['net_pnl'].sum():,.2f}")
    log_col3.metric("Average net P&L", f"${trade_log['net_pnl'].mean():,.2f}")
    log_col4.metric("Win rate", f"{(trade_log['net_pnl'] > 0).mean() * 100:.1f}%")
    st.dataframe(trade_log, use_container_width=True, hide_index=True)
    csv_data = trade_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download trade log CSV", data=csv_data, file_name="btc_trade_log.csv", mime="text/csv")
else:
    st.info("No completed trades logged yet.")

st.caption("Research only, not financial or betting advice. The trade log is stored only in the current browser session. Download the CSV regularly; a page reload, browser close, or Streamlit redeploy can clear it. Limit orders may not fill or may fill only partially.")
