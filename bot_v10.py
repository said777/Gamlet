"""
Bot V10 — Лучшее из V7 + V9.1
Сканер сигналов + SQLite + Telegram + Score 0-10
Расширенные фильтры: Funding, Delta, Retest, False BO, Squeeze, Accumulation, Stop Hunt
"""
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
import logging
import os
from datetime import datetime, timezone
from models import TradeLogger, Trade
from telegram_notifier import TelegramNotifier
from signal_filter import SignalFilter
from dotenv import load_dotenv

# =============================
# CONFIG
# =============================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SL_MULTIPLIER = 5.5       # ATR множитель для SL (широкий — меньше ложных стопов)
MIN_SCORE = 4              # Минимальный Score для отправки сигнала
SCAN_INTERVAL = 120        # Секунд между сканами

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "DOT/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT",
    "INJ/USDT:USDT", "SUI/USDT:USDT", "SEI/USDT:USDT", "HYPE/USDT:USDT",
    "LTC/USDT:USDT", "UNI/USDT:USDT", "AAVE/USDT:USDT", "FARTCOIN/USDT:USDT"
]

# =============================
# EXCHANGES (read-only)
# =============================

exchanges = {}
try:
    exchanges["MEXC"] = ccxt.mexc({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
except:
    log.warning("MEXC init failed")

try:
    exchanges["BINGX"] = ccxt.bingx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
except:
    log.warning("BINGX init failed")

# =============================
# INIT
# =============================

notifier = TelegramNotifier(TELEGRAM_TOKEN, CHAT_ID)
db = TradeLogger()
db.init_db()
signal_filter = SignalFilter(min_hours_between=4)

# Анти-дублирование по цене (из V7)
last_prices = {}   # symbol -> last entry price

# =============================
# БАЗОВЫЕ ИНДИКАТОРЫ
# =============================

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_adx(df, period=14):
    try:
        result = ta.adx(df["h"], df["l"], df["c"], length=period)
        col = f"ADX_{period}"
        return result[col] if result is not None and col in result.columns else pd.Series([float("nan")] * len(df), index=df.index)
    except:
        return pd.Series([float("nan")] * len(df), index=df.index)

def calc_atr(df, period=14):
    try:
        result = ta.atr(df["h"], df["l"], df["c"], length=period)
        return result if result is not None else pd.Series([float("nan")] * len(df), index=df.index)
    except:
        return pd.Series([float("nan")] * len(df), index=df.index)

# =============================
# СВЕЧНЫЕ ПАТТЕРНЫ
# =============================

def is_bullish_engulfing(prev, last):
    return (prev["c"] < prev["o"] and last["c"] > last["o"] and
            last["o"] < prev["c"] and last["c"] > prev["o"])

def is_bearish_engulfing(prev, last):
    return (prev["c"] > prev["o"] and last["c"] < last["o"] and
            last["o"] > prev["c"] and last["c"] < prev["o"])

def is_bullish_pin_bar(candle):
    body = abs(candle["c"] - candle["o"])
    if body == 0: return False
    lower_wick = min(candle["c"], candle["o"]) - candle["l"]
    upper_wick = candle["h"] - max(candle["c"], candle["o"])
    return lower_wick >= 2 * body and upper_wick <= body * 0.5

def is_bearish_pin_bar(candle):
    body = abs(candle["c"] - candle["o"])
    if body == 0: return False
    lower_wick = min(candle["c"], candle["o"]) - candle["l"]
    upper_wick = candle["h"] - max(candle["c"], candle["o"])
    return upper_wick >= 2 * body and lower_wick <= body * 0.5

# =============================
# RSI ДИВЕРГЕНЦИЯ (из V7 — улучшенная)
# =============================

def detect_rsi_divergence(df, lookback=20):
    """
    bearish: цена выше хая, RSI нет → плохой LONG
    bullish: цена ниже лоя, RSI нет → плохой SHORT
    """
    if len(df) < lookback + 2:
        return None
    window = df.iloc[-(lookback + 2):-1]
    last = df.iloc[-1]

    if last["h"] > window["h"].max() and last["rsi"] < window["rsi"].max():
        return "bearish"
    if last["l"] < window["l"].min() and last["rsi"] > window["rsi"].min():
        return "bullish"
    return None

# =============================
# ЛОЖНЫЙ ПРОБОЙ (из V7)
# =============================

def detect_false_breakout(df, lookback=10):
    if len(df) < lookback + 3:
        return None
    window = df.iloc[-(lookback + 3):-2]
    prev = df.iloc[-2]

    level_high = window["h"].max()
    level_low = window["l"].min()

    if prev["l"] < level_low and prev["c"] > level_low:
        return "false_bear"
    if prev["h"] > level_high and prev["c"] < level_high:
        return "false_bull"
    return None

# =============================
# РЕТЕСТ УРОВНЯ (из V7)
# =============================

def check_retest(df, signal, lookback=20):
    if len(df) < lookback + 3:
        return False
    window = df.iloc[-(lookback + 3):-3]
    recent = df.iloc[-3:-1]
    last = df.iloc[-1]

    if signal == "LONG":
        support = window["l"].min()
        touched = recent["l"].min() <= support * 1.005
        bounced = last["c"] > support
        return bool(touched and bounced)
    else:
        resistance = window["h"].max()
        touched = recent["h"].max() >= resistance * 0.995
        rejected = last["c"] < resistance
        return bool(touched and rejected)

# =============================
# СЖАТИЕ ATR (из V7)
# =============================

def detect_range_compression(df, short=5, long=20):
    if len(df) < long + 2:
        return False
    atr_short = df["atr"].iloc[-short:].mean()
    atr_long = df["atr"].iloc[-long:].mean()
    if pd.isna(atr_short) or pd.isna(atr_long) or atr_long == 0:
        return False
    return bool(atr_short < atr_long * 0.75)

# =============================
# СКРЫТОЕ НАКОПЛЕНИЕ (из V7)
# =============================

def detect_hidden_accumulation(df, lookback=10):
    if len(df) < lookback + 2:
        return False
    window = df.iloc[-(lookback + 2):-2]
    last5_price = df["c"].iloc[-5:]
    last5_vol = df["v"].iloc[-5:]

    price_flat = (last5_price.max() - last5_price.min()) / last5_price.mean() < 0.02
    vol_growing = last5_vol.mean() > window["v"].mean() * 1.3
    return bool(price_flat and vol_growing)

# =============================
# СТОП-ХАНТ (из V7)
# =============================

def detect_stop_hunt(df, signal):
    if len(df) < 6:
        return False
    prev = df.iloc[-2]
    window = df.iloc[-6:-2]

    avg_range = (window["h"] - window["l"]).mean()
    if avg_range == 0:
        return False

    candle_range = prev["h"] - prev["l"]
    if signal == "LONG":
        spike_down = (prev["o"] - prev["l"]) > avg_range * 1.5
        recover = prev["c"] > (prev["l"] + candle_range * 0.5)
        return bool(spike_down and recover)
    else:
        spike_up = (prev["h"] - prev["o"]) > avg_range * 1.5
        recover = prev["c"] < (prev["l"] + candle_range * 0.5)
        return bool(spike_up and recover)

# =============================
# ДЕЛЬТА — аппроксимация покупок/продаж (из V7)
# =============================

def calc_delta(df, lookback=5):
    recent = df.iloc[-lookback:]
    delta = 0.0
    for _, row in recent.iterrows():
        rng = row["h"] - row["l"]
        if rng == 0:
            continue
        mid = (row["h"] + row["l"]) / 2
        if row["c"] > mid:
            delta += row["v"] * ((row["c"] - mid) / rng)
        else:
            delta -= row["v"] * ((mid - row["c"]) / rng)
    return delta

# =============================
# ORDER BOOK (из V9)
# =============================

def get_order_book_signal(exchange, symbol):
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(item[1] for item in ob["bids"])
        ask_vol = sum(item[1] for item in ob["asks"])
        total = bid_vol + ask_vol
        if total == 0:
            return None, 0.5
        bid_ratio = bid_vol / total
        if bid_ratio > 0.60:
            return "BUY", round(bid_ratio, 3)
        elif bid_ratio < 0.40:
            return "SELL", round(bid_ratio, 3)
        return None, round(bid_ratio, 3)
    except Exception as e:
        log.warning(f"Order book error {symbol}: {e}")
        return None, 0.5

# =============================
# ПЛОТНОСТЬ ЛИКВИДНОСТИ (из V7)
# =============================

def get_liquidity_density(exchange, symbol, price):
    try:
        ob = exchange.fetch_order_book(symbol, limit=50)
        zone = price * 0.005

        bid_cluster = sum(v for p, v in ob["bids"] if p >= price - zone)
        ask_cluster = sum(v for p, v in ob["asks"] if p <= price + zone)
        total = bid_cluster + ask_cluster

        if total == 0:
            return None, None, "neutral"

        bid_d = bid_cluster / total
        ask_d = ask_cluster / total

        if bid_d > 0.60:
            side = "buy_pressure"
        elif ask_d > 0.60:
            side = "sell_pressure"
        else:
            side = "neutral"

        return round(bid_d, 3), round(ask_d, 3), side
    except Exception as e:
        log.warning(f"Liquidity density error {symbol}: {e}")
        return None, None, "neutral"

# =============================
# FUNDING RATE (из V7)
# =============================

def get_funding_rate(exchange, symbol):
    try:
        fr = exchange.fetch_funding_rate(symbol)
        rate = fr.get("fundingRate", None)
        return float(rate) if rate is not None else None
    except:
        return None

def check_funding_rate(rate, signal):
    if rate is None:
        return True, "➖ N/A"
    rate_pct = rate * 100
    if signal == "LONG" and rate > 0.001:
        return False, f"⚠️ Лонг перегрет ({rate_pct:.3f}%)"
    if signal == "SHORT" and rate < -0.001:
        return False, f"⚠️ Шорт перегрет ({rate_pct:.3f}%)"
    return True, f"✅ {rate_pct:.3f}%"

# =============================
# ПЕРЕГРЕВ РЫНКА (из V7)
# =============================

def check_market_overheating(df, signal):
    rsi = df["rsi"].iloc[-1]
    if signal == "LONG" and rsi > 75:
        return False, f"⚠️ RSI перегрет ({rsi:.1f})"
    if signal == "SHORT" and rsi < 25:
        return False, f"⚠️ RSI перепродан ({rsi:.1f})"
    return True, f"✅ RSI {rsi:.1f}"

# =============================
# РЕЙТИНГ СИГНАЛА 0-10 (из V7)
# =============================

def calc_signal_score(factors: dict) -> int:
    weights = {
        "rsi_signal": 1,
        "adx_ok": 1,
        "volume_ok": 1,
        "ob_match": 1,
        "pattern": 1,
        "funding_ok": 1,
        "range_compression": 1,
        "hidden_accum": 1,
        "stop_hunt": 1,
        "retest": 1,
    }
    return sum(weights[k] for k in weights if factors.get(k))

# =============================
# WICK MANIPULATION FILTER (V10.1)
# =============================

def detect_wick_manipulation(df, lookback=5):
    """
    Обнаруживает манипулятивные свечи: длинные тени без тела.
    Если 2+ свечей из последних lookback имеют wick_ratio > 3.0 — манипуляция.
    """
    if len(df) < lookback + 1:
        return False, 0
    window = df.iloc[-(lookback + 1):-1]
    manip_count = 0
    for _, row in window.iterrows():
        body = abs(row["c"] - row["o"])
        if body == 0:
            body = 0.0001
        upper_wick = row["h"] - max(row["c"], row["o"])
        lower_wick = min(row["c"], row["o"]) - row["l"]
        total_wick = upper_wick + lower_wick
        if total_wick / body > 3.0:
            manip_count += 1
    return manip_count >= 2, manip_count

# =============================
# MOMENTUM ALIGNMENT (V10.1)
# =============================

def check_momentum_alignment(df, signal, lookback=3):
    """
    Проверяет, совпадает ли направление последних N свечей с сигналом.
    Возвращает (aligned, count_with, count_against).
    """
    if len(df) < lookback + 1:
        return True, 0, 0
    recent = df.iloc[-lookback:]
    with_signal = 0
    against_signal = 0
    for _, row in recent.iterrows():
        if signal == "LONG":
            if row["c"] > row["o"]:
                with_signal += 1
            else:
                against_signal += 1
        else:
            if row["c"] < row["o"]:
                with_signal += 1
            else:
                against_signal += 1
    aligned = against_signal < lookback  # не все свечи против
    return aligned, with_signal, against_signal

# =============================
# STRONG DELTA CHECK (V10.1)
# =============================

def is_delta_strongly_against(df, signal, lookback=10):
    """
    Проверяет, сильно ли дельта против сигнала.
    Если текущая дельта против И её абсолютное значение > медианы за lookback × 2 — сильно против.
    """
    if len(df) < lookback + 5:
        return False
    deltas = []
    for i in range(lookback):
        start = -(lookback + 5 - i)
        end = start + 5
        window = df.iloc[start:end]
        d = 0.0
        for _, row in window.iterrows():
            rng = row["h"] - row["l"]
            if rng == 0:
                continue
            mid = (row["h"] + row["l"]) / 2
            if row["c"] > mid:
                d += row["v"] * ((row["c"] - mid) / rng)
            else:
                d -= row["v"] * ((mid - row["c"]) / rng)
        deltas.append(abs(d))

    current_delta = calc_delta(df)
    median_delta = sorted(deltas)[len(deltas) // 2] if deltas else 0

    if median_delta == 0:
        return False

    against = (signal == "LONG" and current_delta < 0) or (signal == "SHORT" and current_delta > 0)
    strong = abs(current_delta) > median_delta * 1.5

    return against and strong

# =============================
# СТРУКТУРА / SWEEP / СВЕЧА (из V9)
# =============================

def market_structure_ok(df, signal):
    try:
        if len(df) < 5: return False
        last = df.iloc[-1]
        prev = df.iloc[-3]
        return last["l"] > prev["l"] if signal == "LONG" else last["h"] < prev["h"]
    except:
        return False

def liquidity_sweep(df, signal):
    try:
        if len(df) < 3: return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        return last["l"] < prev["l"] if signal == "LONG" else last["h"] > prev["h"]
    except:
        return False

def candle_confirmation(prev, last, signal):
    try:
        return last["c"] > prev["c"] if signal == "LONG" else last["c"] < prev["c"]
    except:
        return False

# =============================
# LOG CLOSED TRADE
# =============================

def log_closed_trade(signal_id, exit_price, profit):
    """
    Закрыть позицию вручную:
    >>> from models import TradeLogger
    >>> t = TradeLogger()
    >>> t.close_trade(ID, цена_выхода, профит)
    """
    try:
        trade = db.get_trade_by_id(signal_id)
        if not trade:
            log.error(f"❌ Сделка ID {signal_id} не найдена")
            return False

        db.close_trade(signal_id, exit_price, profit)

        entry_price = trade['entry_price']
        symbol = trade['symbol']
        side = trade['side']
        roi = ((exit_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
        if side == "SHORT":
            roi = -roi

        emoji = "✅" if profit >= 0 else "❌"
        msg = f"""{emoji} *ПОЗИЦИЯ ЗАКРЫТА* — {symbol}

Сторона: {side}
Вход: `{entry_price:.4f}`
Выход: `{exit_price:.4f}`

Профит: `{profit:.2f} USDT`
ROI: `{roi:.2f}%`

_{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")} UTC_
"""
        notifier.send_signal(msg)
        log.info(f"✅ Сделка #{signal_id} закрыта: {profit:.2f} USDT ({roi:.2f}%)")
        return True
    except Exception as e:
        log.error(f"❌ Ошибка: {e}")
        return False

# =============================
# SCAN
# =============================

def scan():
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 3:
        log.debug("Азиатский флэт (00-03 UTC) — пропуск")
        return

    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    log.info(f"=== Сканирование {now_str} ===")
    log.info(f"Монеты: {len(SYMBOLS)} | Биржи: {', '.join(exchanges.keys())}")
    signals_found = 0

    for name, exchange in exchanges.items():
        log.info(f"📊 {name}: сканирую {len(SYMBOLS)} пар...")
        for symbol in SYMBOLS:
            try:
                log.info(f"🔎 {symbol} [{name}]")

                # ===== 4H ТРЕНД =====
                df4 = pd.DataFrame(
                    exchange.fetch_ohlcv(symbol, "4h", limit=220),
                    columns=["t", "o", "h", "l", "c", "v"]
                )
                if len(df4) < 50:
                    continue
                df4["ema50"] = calc_ema(df4["c"], 50)
                df4["ema200"] = calc_ema(df4["c"], 200)
                last4 = df4.iloc[-1]

                if last4["ema50"] > last4["ema200"]:
                    trend4 = "LONG"
                elif last4["ema50"] < last4["ema200"]:
                    trend4 = "SHORT"
                else:
                    continue

                # ===== 1H ДАННЫЕ =====
                df1 = pd.DataFrame(
                    exchange.fetch_ohlcv(symbol, "1h", limit=120),
                    columns=["t", "o", "h", "l", "c", "v"]
                )
                if len(df1) < 50:
                    continue
                df1["rsi"] = calc_rsi(df1["c"])
                df1["adx"] = calc_adx(df1)
                df1["atr"] = calc_atr(df1)
                df1["vol_ma"] = df1["v"].rolling(20).mean()

                last1 = df1.iloc[-1]
                prev1 = df1.iloc[-2]

                # ===== БАЗОВЫЕ ФИЛЬТРЫ =====
                volume_ok = bool(last1["v"] > last1["vol_ma"] * 1.05)
                adx_ok = bool(12 < last1["adx"] < 60)
                if not volume_ok or not adx_ok:
                    continue

                # ===== RSI СИГНАЛ (38/62) =====
                signal = None
                if trend4 == "LONG" and prev1["rsi"] < 38 and last1["rsi"] >= 38:
                    signal = "LONG"
                if trend4 == "SHORT" and prev1["rsi"] > 62 and last1["rsi"] <= 62:
                    signal = "SHORT"
                if signal is None:
                    continue

                # ===== КУЛДАУН (персистентный) =====
                if not signal_filter.is_valid_signal(exchange=name, symbol=symbol):
                    continue

                # ===== ДИВЕРГЕНЦИЯ RSI =====
                divergence = detect_rsi_divergence(df1)
                if signal == "LONG" and divergence == "bearish":
                    log.info(f"⛔ {symbol}: медвежья дивергенция → LONG отклонён")
                    continue
                if signal == "SHORT" and divergence == "bullish":
                    log.info(f"⛔ {symbol}: бычья дивергенция → SHORT отклонён")
                    continue

                div_note = "✅ Бычья" if divergence == "bullish" else "⚠️ Медвежья" if divergence == "bearish" else "➖ Нет"

                # ===== ПЕРЕГРЕВ РЫНКА =====
                heat_ok, heat_note = check_market_overheating(df1, signal)
                if not heat_ok:
                    log.info(f"⛔ {symbol}: {heat_note}")
                    continue

                # ===== ЛОЖНЫЙ ПРОБОЙ =====
                false_bo = detect_false_breakout(df1)
                false_bo_against = (
                    (signal == "LONG" and false_bo == "false_bull") or
                    (signal == "SHORT" and false_bo == "false_bear")
                )
                false_bo_note = f"⚠️ {false_bo}" if false_bo_against else "✅ Нет"

                # ===== FUNDING RATE =====
                funding = get_funding_rate(exchange, symbol)
                funding_ok, funding_note = check_funding_rate(funding, signal)

                # ===== АНТИ-ДУБЛЬ ПО ЦЕНЕ (<0.5%) =====
                entry = last1["c"]
                if symbol in last_prices and abs(entry - last_prices[symbol]) / entry < 0.005:
                    log.info(f"⛔ {symbol}: дубль по цене — пропуск")
                    continue

                # ===== СВЕЧНЫЕ ПАТТЕРНЫ =====
                pattern = None
                if signal == "LONG":
                    if is_bullish_engulfing(prev1, last1): pattern = "🕯 Бычий Engulfing"
                    elif is_bullish_pin_bar(last1): pattern = "📍 Бычий Pin Bar"
                elif signal == "SHORT":
                    if is_bearish_engulfing(prev1, last1): pattern = "🕯 Медвежий Engulfing"
                    elif is_bearish_pin_bar(last1): pattern = "📍 Медвежий Pin Bar"

                # ===== ORDER BOOK =====
                ob_signal, bid_ratio = get_order_book_signal(exchange, symbol)
                ob_match = (signal == "LONG" and ob_signal == "BUY") or (signal == "SHORT" and ob_signal == "SELL")
                ask_ratio = round(1 - bid_ratio, 3)
                ob_note = f"{'✅' if ob_match else '➖'} Покупатели {bid_ratio*100:.1f}% / Продавцы {ask_ratio*100:.1f}%"

                # ===== ПЛОТНОСТЬ ЛИКВИДНОСТИ =====
                bid_density, ask_density, liq_side = get_liquidity_density(exchange, symbol, entry)
                liq_match = (signal == "LONG" and liq_side == "buy_pressure") or (signal == "SHORT" and liq_side == "sell_pressure")
                if bid_density is not None:
                    liq_note = f"{'✅' if liq_match else '➖'} Ниже: {bid_density*100:.1f}% | Выше: {ask_density*100:.1f}%"
                else:
                    liq_note = "➖ N/A"

                # ===== ДЕЛЬТА =====
                delta = calc_delta(df1)
                delta_ok = (signal == "LONG" and delta > 0) or (signal == "SHORT" and delta < 0)
                delta_note = f"{'✅' if delta_ok else '➖'} {'+' if delta >= 0 else ''}{delta:.1f}"

                # ===== БЛОКИРОВКА: стакан + дельта оба против сигнала =====
                ob_against = (signal == "LONG" and ob_signal == "SELL") or (signal == "SHORT" and ob_signal == "BUY")
                delta_against = not delta_ok
                if ob_against and delta_against:
                    log.info(f"⛔ {symbol}: стакан + дельта против {signal} — блокировка")
                    continue

                # ===== БЛОКИРОВКА: сильная дельта против (V10.1) =====
                if is_delta_strongly_against(df1, signal):
                    log.info(f"⛔ {symbol}: дельта сильно против {signal} — блокировка")
                    continue



                # ===== РАСШИРЕННЫЕ ФИЛЬТРЫ =====
                structure = market_structure_ok(df1, signal)
                sweep = liquidity_sweep(df1, signal)
                confirm = candle_confirmation(prev1, last1, signal)
                range_compression = detect_range_compression(df1)
                hidden_accum = detect_hidden_accumulation(df1)
                stop_hunt = detect_stop_hunt(df1, signal)
                retest = check_retest(df1, signal)

                # ===== SCORE 0-10 (бонусы - штрафы) =====
                factors = {
                    "rsi_signal": True,
                    "adx_ok": adx_ok,
                    "volume_ok": volume_ok,
                    "ob_match": ob_match,
                    "pattern": pattern is not None,
                    "funding_ok": funding_ok,
                    "range_compression": range_compression,
                    "hidden_accum": hidden_accum,
                    "stop_hunt": stop_hunt,
                    "retest": retest,
                }
                score = calc_signal_score(factors)

                # Штрафы за противоречие
                if ob_against:
                    score -= 1  # стакан против
                if delta_against:
                    score -= 1  # дельта против
                if not structure:
                    score -= 2  # структура не подтверждена (V10.1: было -1)
                # ...без антиманипуляционных штрафов...
                score = max(0, score)

                score_bar = "⭐" * score + "☆" * (10 - score)

                if score < MIN_SCORE:
                    log.info(f"⛔ {symbol}: Score {score}/10 < {MIN_SCORE} — слабый сигнал")
                    continue

                # ===== ATR SL / TP =====
                atr = last1["atr"]
                if pd.isna(atr) or atr == 0:
                    atr = entry * 0.01

                if signal == "LONG":
                    sl = entry - SL_MULTIPLIER * atr
                    tp1 = entry + 1.0 * atr
                    tp2 = entry + 2.5 * atr
                    tp3 = entry + 6.5 * atr
                else:
                    sl = entry + SL_MULTIPLIER * atr
                    tp1 = entry - 1.0 * atr
                    tp2 = entry - 2.5 * atr
                    tp3 = entry - 6.5 * atr

                rr = round(abs(tp2 - entry) / abs(sl - entry), 2)
                emoji = "🟢" if signal == "LONG" else "🔴"
                pattern_line = f"Паттерн: {pattern}\n" if pattern else ""

                rc_note = "✅ Сжатие" if range_compression else "➖"
                ha_note = "✅ Обнаружено" if hidden_accum else "➖"
                sh_note = "✅ Обнаружен" if stop_hunt else "➖"
                rt_note = "✅ Подтверждён" if retest else "➖"

                # ===== RECORD + DB + SEND =====
                signal_filter.record_signal(exchange=name, symbol=symbol)
                last_prices[symbol] = entry

                trade = Trade(
                    timestamp=datetime.now(),
                    exchange=name,
                    symbol=symbol,
                    side=signal,
                    entry_price=entry,
                    sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
                    ai_score=f"{score}/10"
                )
                trade_id = db.add_trade(trade)
                if trade_id == -1:
                    trade_id = 0

                dt_local = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

                msg = f"""{emoji} *{signal} СИГНАЛ* — {symbol} ID: {trade_id}

Биржа: `{name}`
Вход: `{entry:.4f}`
Stop Loss: `{sl:.4f}` _(ATR ×{SL_MULTIPLIER})_

TP1: `{tp1:.4f}`
TP2: `{tp2:.4f}`
TP3: `{tp3:.4f}`

R:R (к TP2): `{rr}` | ATR: `{atr:.4f}`

📊 *Индикаторы*
Тренд 4H: {trend4} | {heat_note} | ADX: `{last1["adx"]:.1f}`
Дивергенция: {div_note}

📖 *Стакан и ликвидность*
Order Book: {ob_note}
Ликвидность: {liq_note}
Дельта: {delta_note}
Funding: {funding_note}

🔍 *Контекст*
{pattern_line}Структура: {"✅" if structure else "❌"} | Sweep: {"✅" if sweep else "❌"} | Свеча: {"✅" if confirm else "❌"}
Ложный пробой: {false_bo_note}
Стоп-хант: {sh_note}
Ретест: {rt_note}
Сжатие ATR: {rc_note}
Накопление: {ha_note}

🏆 *Score: {score}/10*
{score_bar}

_{dt_local} UTC_

*Закрыть позицию:*
`from models import TradeLogger; TradeLogger().close_trade({trade_id}, цена, профит)`
"""
                notifier.send_signal(msg)
                log.info(f"✅ Сигнал #{trade_id}: {signal} {symbol} @ {name} | Score={score}/10")
                signals_found += 1

            except Exception as e:
                log.error(f"Ошибка [{name}] {symbol}: {e}")

    log.info(f"=== Завершено | {len(exchanges) * len(SYMBOLS)} пар | Сигналов: {signals_found} ===")

# =============================
# START
# =============================

if __name__ == "__main__":
    log.info("Bot V10 STARTED — Best of V7 + V9.1")
    notifier.send_signal(
        "⚡ *Bot V10 Запущен*\n\n"
        "📍 Режим: Сканер + Ручная торговля\n"
        f"✅ {len(SYMBOLS)} пар | {len(exchanges)} бирж\n"
        "✅ Score 0-10 | Фильтр слабых сигналов\n"
        "✅ Funding | Delta | Ликвидность\n"
        "✅ SQLite + Telegram\n"
        f"⚙️ SL ×{SL_MULTIPLIER} | MIN Score: {MIN_SCORE}/10"
    )

    while True:
        try:
            scan()
        except Exception as e:
            log.critical(f"Критическая ошибка: {e}")
            notifier.send_signal(f"🚨 *Критическая ошибка:* `{e}`")
        time.sleep(SCAN_INTERVAL)
