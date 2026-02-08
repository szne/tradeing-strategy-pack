from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from trading_sdk.base_strategy import BaseStrategy
from trading_sdk.structs import AccountData, AccountSnapshot, Action, OrderSignal


class FiveMinuteReversionStrategy(BaseStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "atr_take_mult": 1.0,
            "exit_rsi": 50,
            "max_holding_bars": 96,
            "risk_pct": 0.02,
            "max_position_pct": 0.3,
            "min_qty": 0.0,
            "max_leverage": 50.0,
            "leverage_cap": 200.0,
            "max_drawdown": 0.2,
            "slippage_bps": 2.0,
        }

    def next_signal(self, market_data: Any, account_data: AccountData) -> OrderSignal:
        df = market_data
        required = {"Close", "High", "Low"}
        if df is None or not required.issubset(df.columns):
            raise ValueError("market_data must include High, Low, Close columns")

        bb_period = int(self.params.get("bb_period", 20))
        rsi_period = int(self.params.get("rsi_period", 14))
        atr_period = int(self.params.get("atr_period", 14))
        if min(bb_period, rsi_period, atr_period) <= 1:
            raise ValueError("indicator windows must be positive")

        min_rows = max(bb_period + 1, rsi_period + 1, atr_period + 1) + 1
        if len(df) < min_rows:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        close = pd.to_numeric(df["Close"], errors="coerce")
        high = pd.to_numeric(df["High"], errors="coerce")
        low = pd.to_numeric(df["Low"], errors="coerce")

        mean = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        upper = mean + float(self.params.get("bb_std", 2.0)) * std
        lower = mean - float(self.params.get("bb_std", 2.0)) * std
        rsi = self._calc_rsi(close, rsi_period)
        atr = self._calc_atr(high, low, close, atr_period)

        price = float(close.iloc[-1])
        curr_rsi = rsi.iloc[-1]
        curr_atr = atr.iloc[-1]
        mid = mean.iloc[-1]
        up = upper.iloc[-1]
        lo = lower.iloc[-1]

        if any(pd.isna(val) for val in (curr_rsi, curr_atr, mid, up, lo)):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        snapshot = self._resolve_account_snapshot(account_data)
        if self._is_drawdown_stop(snapshot):
            return self._exit_signal(snapshot, "max drawdown stop")

        entry_index = self._update_position_state(snapshot, len(df))
        max_holding = int(self.params.get("max_holding_bars", 0))
        if max_holding > 0 and entry_index is not None:
            bars_in = (len(df) - 1) - entry_index
            if bars_in >= max_holding:
                return self._exit_signal(snapshot, "max holding time")

        exit_rsi = float(self.params.get("exit_rsi", 50))
        if snapshot.position_qty > 0 and (price >= mid or curr_rsi >= exit_rsi):
            return self._exit_signal(snapshot, "mean revert exit")
        if snapshot.position_qty < 0 and (price <= mid or curr_rsi <= 100 - exit_rsi):
            return self._exit_signal(snapshot, "mean revert exit")

        action: Action = "WAIT"
        reasoning = "no setup"
        oversold = float(self.params.get("rsi_oversold", 30))
        overbought = float(self.params.get("rsi_overbought", 70))
        if price <= lo and curr_rsi <= oversold:
            action = "BUY"
            reasoning = "mean reversion long"
        elif price >= up and curr_rsi >= overbought:
            action = "SELL"
            reasoning = "mean reversion short"

        if action == "WAIT":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning=reasoning)

        if snapshot.position_qty > 0 and action == "BUY":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already long")
        if snapshot.position_qty < 0 and action == "SELL":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="already short")

        stop_mult = float(self.params.get("atr_stop_mult", 1.5))
        take_mult = float(self.params.get("atr_take_mult", 1.0))
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

    def _exit_signal(self, snapshot: AccountSnapshot, reason: str) -> OrderSignal:
        qty = float(snapshot.position_qty)
        if qty > 0:
            return OrderSignal(action="SELL", quantity=abs(qty), reasoning=reason)
        if qty < 0:
            return OrderSignal(action="BUY", quantity=abs(qty), reasoning=reason)
        return OrderSignal(action="WAIT", quantity=0.0, reasoning=reason)


class FiveMinuteAggressiveReversionStrategy(FiveMinuteReversionStrategy):
    def default_params(self) -> Dict[str, Any]:
        params = super().default_params()
        params.update(
            {
                "risk_pct": 0.05,
                "max_leverage": 100.0,
                "max_position_pct": 0.6,
                "atr_take_mult": 1.5,
            }
        )
        return params
