from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from trading_sdk.base_strategy import BaseStrategy
from trading_sdk.structs import AccountData, AccountSnapshot, Action, OrderSignal


class SampleStrategy(BaseStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "fast_window": 10,
            "slow_window": 30,
            "risk_pct": 0.01,
            "stop_loss_pct": 0.01,
            "min_qty": 0.0,
        }

    def next_signal(self, market_data: Any, account_data: AccountData) -> OrderSignal:
        df = market_data
        if df is None or "Close" not in df.columns:
            raise ValueError("market_data must include 'Close' column")

        fast_window = int(self.params.get("fast_window", 10))
        slow_window = int(self.params.get("slow_window", 30))
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("moving average windows must be positive")

        if len(df) < slow_window + 1:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        close = pd.to_numeric(df["Close"], errors="coerce")
        fast_ma = close.rolling(fast_window).mean()
        slow_ma = close.rolling(slow_window).mean()

        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]

        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        action: Action = "WAIT"
        reasoning = "no signal"

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            action = "BUY"
            reasoning = "fast_ma crossed above slow_ma"
        elif prev_fast >= prev_slow and curr_fast < curr_slow:
            action = "SELL"
            reasoning = "fast_ma crossed below slow_ma"

        if action == "WAIT":
            return OrderSignal(action="WAIT", quantity=0.0, reasoning=reasoning)

        price = float(close.iloc[-1])
        quantity = self._calc_qty(price, account_data)
        min_qty = float(self.params.get("min_qty", 0.0))
        if quantity <= min_qty:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="quantity below minimum")

        return OrderSignal(action=action, quantity=quantity, type="MARKET", reasoning=reasoning)

    def _calc_qty(self, price: float, account_data: AccountData) -> float:
        snapshot = self._resolve_account_snapshot(account_data)
        balance = self._resolve_balance_value(snapshot)
        cash_value = snapshot.cash if snapshot.cash > 0 else balance
        risk_pct = float(self.params.get("risk_pct", 0.0))
        stop_loss_pct = float(self.params.get("stop_loss_pct", 0.0))
        fee_rate = self._resolve_fee_rate()

        if (
            price <= 0
            or balance <= 0
            or cash_value <= 0
            or risk_pct <= 0
            or stop_loss_pct <= 0
        ):
            return 0.0

        risk_qty = (balance * risk_pct) / (price * stop_loss_pct)
        max_qty = cash_value / (price * (1 + fee_rate))
        quantity = min(risk_qty, max_qty)
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
