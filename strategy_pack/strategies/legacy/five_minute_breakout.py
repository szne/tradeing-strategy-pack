from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np

from trading_sdk.base_strategy import BaseStrategy
from trading_sdk.structs import AccountData, AccountSnapshot, Action, OrderSignal


class FiveMinuteBreakoutStrategy(BaseStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "fast_ema": 12,
            "slow_ema": 36,
            "rsi_period": 14,
            "rsi_long": 55,
            "rsi_short": 45,
            "breakout_lookback": 20,
            "breakout_buffer_pct": 0.0005,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "atr_take_mult": 3.0,
            "atr_trail_mult": 2.0,
            "use_trailing_exit": True,
            "use_mean_reversion": True,
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_band_buffer": 0.0,
            "mean_rev_rsi": 30,
            "mean_rev_exit_rsi": 50,
            "use_volatility_filter": True,
            "atr_pct_min": 0.001,
            "atr_pct_max": 0.01,
            "use_time_filter": True,
            "time_filter_hours": [12, 13, 14, 15, 16, 17, 18, 19, 20],
            "time_filter_tz": "UTC",
            "use_trend_filter": True,
            "adx_period": 14,
            "adx_min": 20.0,
            "use_regime_switch": True,
            "regime_adx": 18.0,
            "use_vola_breakout": True,
            "vola_lookback": 20,
            "vola_expand_mult": 1.5,
            "risk_pct": 0.02,
            "max_position_pct": 0.3,
            "min_qty": 0.0,
            "max_leverage": 50.0,
            "leverage_cap": 200.0,
            "max_drawdown": 0.2,
            "slippage_bps": 2.0,
            "max_holding_bars": 288,
        }

    def next_signal(self, market_data: Any, account_data: AccountData) -> OrderSignal:
        df = market_data
        required = {"Close", "High", "Low"}
        if df is None or not required.issubset(df.columns):
            raise ValueError("market_data must include High, Low, Close columns")

        fast_ema = int(self.params.get("fast_ema", 12))
        slow_ema = int(self.params.get("slow_ema", 36))
        rsi_period = int(self.params.get("rsi_period", 14))
        lookback = int(self.params.get("breakout_lookback", 20))
        atr_period = int(self.params.get("atr_period", 14))
        if min(fast_ema, slow_ema, rsi_period, lookback, atr_period) <= 0:
            raise ValueError("indicator windows must be positive")

        min_rows = max(slow_ema, rsi_period + 1, lookback + 1, atr_period + 1) + 1
        if len(df) < min_rows:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")

        fast = close.ewm(span=fast_ema, adjust=False).mean()
        slow = close.ewm(span=slow_ema, adjust=False).mean()
        rsi = self._calc_rsi(close, rsi_period)
        atr = self._calc_atr(high, low, close, atr_period)
        adx = self._calc_adx(high, low, close, int(self.params.get("adx_period", 14)))

        prev_fast, curr_fast = fast.iloc[-2], fast.iloc[-1]
        prev_slow, curr_slow = slow.iloc[-2], slow.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_atr = atr.iloc[-1]
        curr_adx = adx.iloc[-1]
        price = float(close.iloc[-1])
        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")
        if pd.isna(curr_rsi) or pd.isna(curr_atr) or pd.isna(curr_adx):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")
        if not self._volatility_gate(price, curr_atr):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="volatility gate")
        if not self._time_gate(df):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="time filter")
        if not self._trend_gate(curr_adx):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="trend filter")

        rsi_long = float(self.params.get("rsi_long", 55))
        rsi_short = float(self.params.get("rsi_short", 45))
        breakout_lookback = int(self.params.get("breakout_lookback", 20))

        prev_high = high.iloc[-(breakout_lookback + 1) : -1].max()
        prev_low = low.iloc[-(breakout_lookback + 1) : -1].min()

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

        in_trend_regime = self._is_trend_regime(curr_adx)
        action: Action = "WAIT"
        reasoning = "no setup"
        buffer_pct = float(self.params.get("breakout_buffer_pct", 0.0))
        long_break = prev_high * (1 + max(buffer_pct, 0.0))
        short_break = prev_low * (1 - max(buffer_pct, 0.0))
        if prev_fast <= prev_slow and curr_fast > curr_slow and curr_rsi >= rsi_long and price > long_break:
            action = "BUY"
            reasoning = "bullish ema cross with breakout"
        elif (
            prev_fast >= prev_slow
            and curr_fast < curr_slow
            and curr_rsi <= rsi_short
            and price < short_break
        ):
            action = "SELL"
            reasoning = "bearish ema cross with breakdown"

        if action == "WAIT" and not in_trend_regime:
            mean_reversion = self._mean_reversion_signal(close, curr_rsi)
            if mean_reversion is not None:
                action, reasoning = mean_reversion

        if action == "WAIT" and in_trend_regime:
            vola_breakout = self._vola_breakout_signal(high, low, close, curr_atr)
            if vola_breakout is not None:
                action, reasoning = vola_breakout

        if action == "WAIT":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning=reasoning)
        if snapshot.position_qty > 0 and action == "BUY":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already long")
        if snapshot.position_qty < 0 and action == "SELL":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already short")

        stop_mult = float(self.params.get("atr_stop_mult", 1.5))
        take_mult = float(self.params.get("atr_take_mult", 3.0))
        if stop_mult <= 0 or take_mult <= 0:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="invalid risk params")

        if action == "BUY":
            sl = price - curr_atr * stop_mult
            tp = price + curr_atr * take_mult
            stop_distance = price - sl
        else:
            sl = price + curr_atr * stop_mult
            tp = price - curr_atr * take_mult
            stop_distance = sl - price

        quantity = self._calc_qty(price, stop_distance, snapshot, action)
        min_qty = float(self.params.get("min_qty", 0.0))
        if quantity <= min_qty:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="quantity below minimum")

        return OrderSignal(
            action=action,
            quantity=quantity,
            type="MARKET",
            sl=sl,
            tp=tp,
            reasoning=reasoning,
        )

    def _mean_reversion_signal(
        self, close: pd.Series, curr_rsi: float
    ) -> tuple[Action, str] | None:
        if not bool(self.params.get("use_mean_reversion", False)):
            return None
        bb_period = int(self.params.get("bb_period", 20))
        bb_std = float(self.params.get("bb_std", 2.0))
        if bb_period <= 1 or bb_std <= 0:
            return None
        mean = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = mean + bb_std * std
        lower = mean - bb_std * std
        buffer = float(self.params.get("bb_band_buffer", 0.0))
        price = float(close.iloc[-1])
        upper_level = float(upper.iloc[-1]) * (1 - max(buffer, 0.0))
        lower_level = float(lower.iloc[-1]) * (1 + max(buffer, 0.0))
        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return None
        rsi_entry = float(self.params.get("mean_rev_rsi", 30))
        if price <= lower_level and curr_rsi <= rsi_entry:
            return ("BUY", "mean reversion buy")
        if price >= upper_level and curr_rsi >= 100 - rsi_entry:
            return ("SELL", "mean reversion sell")
        return None

    def _calc_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

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
            if action == "BUY":
                effective_price = price * (1 + slip)
            else:
                effective_price = price * (1 - slip)

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
        if not bool(self.params.get("use_trailing_exit", False)):
            return None
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

    def _volatility_gate(self, price: float, atr_value: float) -> bool:
        if not bool(self.params.get("use_volatility_filter", False)):
            return True
        if price <= 0 or atr_value <= 0:
            return False
        atr_pct = atr_value / price
        atr_min = float(self.params.get("atr_pct_min", 0.0))
        atr_max = float(self.params.get("atr_pct_max", 0.0))
        if atr_min > 0 and atr_pct < atr_min:
            return False
        if atr_max > 0 and atr_pct > atr_max:
            return False
        return True

    def _trend_gate(self, adx_value: float) -> bool:
        if not bool(self.params.get("use_trend_filter", False)):
            return True
        adx_min = float(self.params.get("adx_min", 0.0))
        if adx_min <= 0:
            return True
        return adx_value >= adx_min

    def _is_trend_regime(self, adx_value: float) -> bool:
        if not bool(self.params.get("use_regime_switch", False)):
            return True
        threshold = float(self.params.get("regime_adx", 0.0))
        if threshold <= 0:
            return True
        return adx_value >= threshold

    def _time_gate(self, df: pd.DataFrame) -> bool:
        if not bool(self.params.get("use_time_filter", False)):
            return True
        if df is None or df.empty:
            return False
        index = df.index
        if not hasattr(index, "tz_localize"):
            return True
        try:
            tz = str(self.params.get("time_filter_tz", "UTC"))
            if index.tz is None:
                localized = index.tz_localize("UTC")
            else:
                localized = index.tz_convert("UTC")
            if tz and tz.upper() != "UTC":
                localized = localized.tz_convert(tz)
            hour = int(localized[-1].hour)
        except Exception:
            return True
        hours = self.params.get("time_filter_hours", [])
        if not hours:
            return True
        return hour in {int(h) for h in hours}

    def _vola_breakout_signal(
        self, high: pd.Series, low: pd.Series, close: pd.Series, atr_value: float
    ) -> tuple[Action, str] | None:
        if not bool(self.params.get("use_vola_breakout", False)):
            return None
        lookback = int(self.params.get("vola_lookback", 20))
        if lookback <= 2:
            return None
        prev_high = high.iloc[-(lookback + 1) : -1].max()
        prev_low = low.iloc[-(lookback + 1) : -1].min()
        prev_atr = self._calc_atr(high, low, close, int(self.params.get("atr_period", 14)))
        prev_atr_val = prev_atr.iloc[-2]
        if pd.isna(prev_atr_val) or prev_atr_val <= 0:
            return None
        expand_mult = float(self.params.get("vola_expand_mult", 1.5))
        if expand_mult <= 0:
            return None
        price = float(close.iloc[-1])
        if atr_value >= prev_atr_val * expand_mult and price > prev_high:
            return ("BUY", "volatility expansion breakout")
        if atr_value >= prev_atr_val * expand_mult and price < prev_low:
            return ("SELL", "volatility expansion breakdown")
        return None

    def _exit_signal(self, snapshot: AccountSnapshot, reason: str) -> OrderSignal:
        qty = float(snapshot.position_qty)
        if qty > 0:
            return OrderSignal(action="SELL", quantity=abs(qty), reasoning=reason)
        if qty < 0:
            return OrderSignal(action="BUY", quantity=abs(qty), reasoning=reason)
        return OrderSignal(action="WAIT", quantity=0.0, reasoning=reason)
