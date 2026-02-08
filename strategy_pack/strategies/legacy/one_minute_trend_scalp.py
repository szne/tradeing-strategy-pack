from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from trading_sdk.base_strategy import BaseStrategy
from trading_sdk.structs import AccountData, AccountSnapshot, Action, OrderSignal


class OneMinuteTrendScalpStrategy(BaseStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "ema_fast": 20,
            "ema_slow": 60,
            "adx_period": 14,
            "adx_min": 16.0,
            "atr_min_pct": 0.001,
            "trend_score_lookback": 12,
            "trend_score_threshold": 4,
            "slope_min": 0.0,
            "confirm_hold_bars": 0,
            "max_counter_bars": 3,
            "atr_period": 14,
            "atr_stop_mult": 1.2,
            "atr_trail_mult": 1.6,
            "atr_expand_mult": 1.0,
            "atr_ma_period": 20,
            "breakout_lookback": 8,
            "breakout_buffer_pct": 0.0,
            "risk_pct": 0.01,
            "max_position_pct": 0.2,
            "min_qty": 0.0,
            "max_leverage": 50.0,
            "leverage_cap": 200.0,
            "max_drawdown": 0.2,
            "slippage_bps": 2.0,
            "max_holding_bars": 180,
            "daily_trade_limit": 2,
            "min_trade_gap_bars": 300,
        }

    def next_signal(self, market_data: Any, account_data: AccountData) -> OrderSignal:
        df = market_data
        required = {"Close", "High", "Low"}
        if df is None or not required.issubset(df.columns):
            raise ValueError("market_data must include High, Low, Close columns")

        ema_fast = int(self.params.get("ema_fast", 20))
        ema_slow = int(self.params.get("ema_slow", 60))
        adx_period = int(self.params.get("adx_period", 14))
        atr_period = int(self.params.get("atr_period", 14))
        lookback = int(self.params.get("breakout_lookback", 10))
        if min(ema_fast, ema_slow, adx_period, atr_period, lookback) <= 1:
            raise ValueError("indicator windows must be positive")

        min_rows = max(ema_slow, adx_period + 1, atr_period + 1, lookback + 1) + 1
        if len(df) < min_rows:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")

        ema_f = close.ewm(span=ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=ema_slow, adjust=False).mean()
        atr = self._calc_atr(high, low, close, atr_period)
        adx = self._calc_adx(high, low, close, adx_period)
        atr_ma = atr.rolling(int(self.params.get("atr_ma_period", 20))).mean()

        price = float(close.iloc[-1])
        prev_fast, curr_fast = ema_f.iloc[-2], ema_f.iloc[-1]
        prev_slow, curr_slow = ema_s.iloc[-2], ema_s.iloc[-1]
        curr_atr = atr.iloc[-1]
        curr_adx = adx.iloc[-1]
        curr_atr_ma = atr_ma.iloc[-1]

        if any(
            pd.isna(val)
            for val in (
                prev_fast,
                prev_slow,
                curr_fast,
                curr_slow,
                curr_atr,
                curr_adx,
                curr_atr_ma,
            )
        ):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        if not self._trade_limit_ok(df):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="trade limit")

        snapshot = self._resolve_account_snapshot(account_data)
        if self._is_drawdown_stop(snapshot):
            return self._exit_signal(snapshot, "max drawdown stop")

        entry_index = self._update_position_state(snapshot, len(df))
        max_holding = int(self.params.get("max_holding_bars", 0))
        if max_holding > 0 and entry_index is not None:
            bars_in = (len(df) - 1) - entry_index
            if bars_in >= max_holding:
                return self._exit_signal(snapshot, "max holding time")

        trailing_exit = self._maybe_trailing_exit(snapshot, price, curr_atr)
        if trailing_exit is not None:
            return trailing_exit

        adx_min = float(self.params.get("adx_min", 0.0))
        if adx_min > 0 and curr_adx < adx_min:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="weak trend")

        if curr_atr_ma <= 0:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="invalid atr_ma")
        expand_mult = float(self.params.get("atr_expand_mult", 1.0))
        if expand_mult > 0 and curr_atr < curr_atr_ma * expand_mult:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="volatility not expanded")

        if not self._trend_state_ok(close, curr_atr, ema_s):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="trend state not ok")

        counter_exit = self._maybe_counter_exit(snapshot, close, ema_s)
        if counter_exit is not None:
            return counter_exit

        prev_high = high.iloc[-(lookback + 1) : -1].max()
        prev_low = low.iloc[-(lookback + 1) : -1].min()
        buffer_pct = float(self.params.get("breakout_buffer_pct", 0.0))
        long_break = prev_high * (1 + max(buffer_pct, 0.0))
        short_break = prev_low * (1 - max(buffer_pct, 0.0))

        action: Action = "WAIT"
        reasoning = "no setup"
        if prev_fast <= prev_slow and curr_fast > curr_slow and self._confirm_breakout(close, long_break):
            action = "BUY"
            reasoning = "1m trend breakout long"
        elif prev_fast >= prev_slow and curr_fast < curr_slow and self._confirm_breakout(close, short_break, is_long=False):
            action = "SELL"
            reasoning = "1m trend breakout short"

        if action == "WAIT":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning=reasoning)

        if snapshot.position_qty > 0 and action == "BUY":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already long")
        if snapshot.position_qty < 0 and action == "SELL":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already short")

        stop_mult = float(self.params.get("atr_stop_mult", 1.2))
        if stop_mult <= 0:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="invalid stop params")

        if action == "BUY":
            sl = price - curr_atr * stop_mult
            stop_distance = price - sl
        else:
            sl = price + curr_atr * stop_mult
            stop_distance = sl - price

        quantity = self._calc_qty(price, stop_distance, snapshot, action)
        min_qty = float(self.params.get("min_qty", 0.0))
        if quantity <= min_qty:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="quantity below minimum")

        self._record_trade_marker(df)
        return OrderSignal(
            action=action,
            quantity=quantity,
            type="MARKET",
            sl=sl,
            reasoning=reasoning,
        )

    def _calc_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def _calc_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        if period <= 1:
            return pd.Series([np.nan] * len(close), index=close.index, dtype="float64")
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0).astype("float64")
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0).astype("float64")
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        atr = atr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = (abs(plus_di - minus_di) / denom) * 100
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    def _calc_qty(
        self, price: float, stop_distance: float, snapshot: AccountSnapshot, action: Action
    ) -> float:
        balance = self._resolve_balance_value(snapshot)
        cash_value = snapshot.cash if snapshot.cash > 0 else balance
        risk_pct = float(self.params.get("risk_pct", 0.0))
        max_position_pct = float(self.params.get("max_position_pct", 0.0))
        max_leverage = float(self.params.get("max_leverage", 0.0))
        leverage_cap = float(self.params.get("leverage_cap", 0.0))
        fee_rate = self._resolve_fee_rate()
        slippage_bps = float(self.params.get("slippage_bps", 0.0))

        if (
            price <= 0
            or balance <= 0
            or cash_value <= 0
            or risk_pct <= 0
            or stop_distance <= 0
        ):
            return 0.0

        effective_price = price
        if slippage_bps > 0:
            slip = slippage_bps / 10_000
            effective_price = price * (1 + slip) if action == "BUY" else price * (1 - slip)

        risk_qty = (balance * risk_pct) / stop_distance
        max_qty_cash = cash_value / (effective_price * (1 + fee_rate))
        max_qty_position = (
            (balance * max_position_pct) / effective_price
            if max_position_pct > 0
            else max_qty_cash
        )
        leverage_limit = leverage_cap if leverage_cap > 0 else max_leverage
        leverage = min(max_leverage, leverage_limit) if max_leverage > 0 else leverage_limit
        if leverage > 0:
            max_qty_leverage = (balance * leverage) / effective_price
        else:
            max_qty_leverage = max_qty_cash
        quantity = min(risk_qty, max_qty_cash, max_qty_position, max_qty_leverage)
        if quantity <= 0:
            return 0.0
        return quantity

    def _resolve_account_snapshot(self, account_data: AccountData) -> AccountSnapshot:
        has_values = False
        if isinstance(account_data, AccountSnapshot):
            has_values = True
        elif isinstance(account_data, dict):
            has_values = any(key in account_data for key in ("balance", "equity", "cash"))
        elif account_data is not None:
            has_values = any(hasattr(account_data, key) for key in ("balance", "equity", "cash"))

        snapshot = AccountSnapshot.from_account_data(account_data)
        if has_values:
            return snapshot

        config = self.config.get("backtest", {}) if self.config else {}
        if "initial_balance" in config:
            value = float(config["initial_balance"])
            return AccountSnapshot(
                balance=value,
                equity=value,
                cash=value,
                position_qty=snapshot.position_qty,
                position_avg_price=snapshot.position_avg_price,
            )

        return snapshot

    def _resolve_balance_value(self, snapshot: AccountSnapshot) -> float:
        for value in (snapshot.balance, snapshot.cash, snapshot.equity):
            if value > 0:
                return float(value)
        return 0.0

    def _resolve_fee_rate(self) -> float:
        fees_cfg = self.config.get("fees", {}) if self.config else {}
        if isinstance(fees_cfg, dict) and "taker" in fees_cfg:
            try:
                value = float(fees_cfg.get("taker", 0.0))
                return value if value > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0

        backtest_cfg = self.config.get("backtest", {}) if self.config else {}
        if isinstance(backtest_cfg, dict) and "fee_rate" in backtest_cfg:
            try:
                value = float(backtest_cfg.get("fee_rate", 0.0))
                return value if value > 0 else 0.0
            except (TypeError, ValueError):
                return 0.0

        return 0.0

    def _is_drawdown_stop(self, snapshot: AccountSnapshot) -> bool:
        max_drawdown = float(self.params.get("max_drawdown", 0.0))
        if max_drawdown <= 0:
            return False

        equity = snapshot.equity if snapshot.equity > 0 else snapshot.balance
        if equity <= 0:
            return False

        peak = self.state.get("peak_equity", 0.0)
        if equity > peak:
            self.state["peak_equity"] = equity
            peak = equity

        if peak <= 0:
            return False
        return equity <= peak * (1 - max_drawdown)

    def _update_position_state(self, snapshot: AccountSnapshot, bar_index: int) -> int | None:
        prev_qty = float(self.state.get("position_qty", 0.0))
        curr_qty = float(snapshot.position_qty)
        entry_index = self.state.get("entry_index")

        if prev_qty == 0 and curr_qty != 0:
            entry_index = bar_index - 1
            self.state["entry_index"] = entry_index
        elif curr_qty == 0 and prev_qty != 0:
            self.state.pop("entry_index", None)
            entry_index = None

        self.state["position_qty"] = curr_qty
        return entry_index

    def _maybe_trailing_exit(
        self, snapshot: AccountSnapshot, price: float, atr_value: float
    ) -> OrderSignal | None:
        if atr_value <= 0:
            return None
        qty = float(snapshot.position_qty)
        if qty == 0:
            self.state.pop("trail_price", None)
            return None
        trail_mult = float(self.params.get("atr_trail_mult", 0.0))
        if trail_mult <= 0:
            return None
        trail_price = self.state.get("trail_price")
        if qty > 0:
            new_trail = price - atr_value * trail_mult
            if trail_price is None or new_trail > trail_price:
                self.state["trail_price"] = new_trail
                trail_price = new_trail
            if trail_price is not None and price <= float(trail_price):
                return OrderSignal(action="SELL", quantity=abs(qty), reasoning="atr trailing stop")
        else:
            new_trail = price + atr_value * trail_mult
            if trail_price is None or new_trail < trail_price:
                self.state["trail_price"] = new_trail
                trail_price = new_trail
            if trail_price is not None and price >= float(trail_price):
                return OrderSignal(action="BUY", quantity=abs(qty), reasoning="atr trailing stop")
        return None

    def _exit_signal(self, snapshot: AccountSnapshot, reason: str) -> OrderSignal:
        qty = float(snapshot.position_qty)
        if qty > 0:
            return OrderSignal(action="SELL", quantity=abs(qty), reasoning=reason)
        if qty < 0:
            return OrderSignal(action="BUY", quantity=abs(qty), reasoning=reason)
        return OrderSignal(action="WAIT", quantity=0.0, reasoning=reason)

    def _trend_state_ok(self, close: pd.Series, atr_value: float, ema_s: pd.Series) -> bool:
        price = float(close.iloc[-1])
        if price <= 0 or atr_value <= 0:
            return False
        atr_min_pct = float(self.params.get("atr_min_pct", 0.0))
        if atr_min_pct > 0 and (atr_value / price) < atr_min_pct:
            return False
        lookback = int(self.params.get("trend_score_lookback", 12))
        if lookback > 1:
            trend_score = int(np.sign(close.diff().iloc[-lookback:]).sum())
            threshold = int(self.params.get("trend_score_threshold", 0))
            if threshold > 0 and abs(trend_score) < threshold:
                return False
        slope_min = float(self.params.get("slope_min", 0.0))
        if slope_min > 0:
            slope = float(ema_s.iloc[-1] - ema_s.iloc[-2])
            if abs(slope) < slope_min:
                return False
        return True

    def _confirm_breakout(self, close: pd.Series, level: float, is_long: bool = True) -> bool:
        hold_bars = int(self.params.get("confirm_hold_bars", 1))
        if hold_bars <= 0:
            return (close.iloc[-1] > level) if is_long else (close.iloc[-1] < level)
        if len(close) < hold_bars + 1:
            return False
        recent = close.iloc[-(hold_bars + 1) :]
        if is_long:
            return bool((recent > level).all())
        return bool((recent < level).all())

    def _maybe_counter_exit(
        self, snapshot: AccountSnapshot, close: pd.Series, ema_s: pd.Series
    ) -> OrderSignal | None:
        qty = float(snapshot.position_qty)
        if qty == 0:
            self.state.pop("counter_bars", None)
            return None
        slope = float(ema_s.iloc[-1] - ema_s.iloc[-2])
        if (qty > 0 and slope <= 0) or (qty < 0 and slope >= 0):
            return self._exit_signal(snapshot, "trend slope reversal")

        max_counter = int(self.params.get("max_counter_bars", 0))
        if max_counter <= 0:
            return None
        diff = float(close.iloc[-1] - close.iloc[-2])
        against = (diff < 0) if qty > 0 else (diff > 0)
        counter = int(self.state.get("counter_bars", 0))
        counter = counter + 1 if against else 0
        self.state["counter_bars"] = counter
        if counter >= max_counter:
            return self._exit_signal(snapshot, "counter bars exit")
        return None

    def _trade_limit_ok(self, df: pd.DataFrame) -> bool:
        limit = int(self.params.get("daily_trade_limit", 0))
        gap_bars = int(self.params.get("min_trade_gap_bars", 0))
        if limit <= 0 and gap_bars <= 0:
            return True

        last_trade_bar = self.state.get("last_trade_bar")
        if gap_bars > 0 and last_trade_bar is not None:
            if (len(df) - 1) - int(last_trade_bar) < gap_bars:
                return False

        if limit <= 0:
            return True

        index = df.index
        if not hasattr(index, "date"):
            return True
        try:
            current_date = index[-1].date()
        except Exception:
            return True
        daily_counts = self.state.setdefault("daily_trade_counts", {})
        return int(daily_counts.get(str(current_date), 0)) < limit

    def _record_trade_marker(self, df: pd.DataFrame) -> None:
        self.state["last_trade_bar"] = len(df) - 1
        index = df.index
        if not hasattr(index, "date"):
            return
        try:
            current_date = index[-1].date()
        except Exception:
            return
        daily_counts = self.state.setdefault("daily_trade_counts", {})
        key = str(current_date)
        daily_counts[key] = int(daily_counts.get(key, 0)) + 1
