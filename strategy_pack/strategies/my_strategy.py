from __future__ import annotations

from typing import Any, Dict

from trading_sdk.base_strategy import BaseStrategy
from trading_sdk.structs import AccountSnapshot, OrderSignal


class MyStrategy(BaseStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "risk_pct": 0.01,
            "min_qty": 0.0,
        }

    def next_signal(self, market_data: Any, account_data: Any) -> OrderSignal:
        if market_data is None or "Close" not in market_data.columns or len(market_data) < 2:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="insufficient data")

        close = market_data["Close"]
        prev_price = float(close.iloc[-2])
        curr_price = float(close.iloc[-1])
        if curr_price <= 0:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="invalid price")

        snapshot = AccountSnapshot.from_account_data(account_data)
        balance = snapshot.cash if snapshot.cash > 0 else snapshot.balance
        fee_rate = self._resolve_fee_rate()
        risk_pct = float(self.params.get("risk_pct", 0.0))
        min_qty = float(self.params.get("min_qty", 0.0))
        if balance <= 0 or risk_pct <= 0:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="invalid account or params")

        quantity = (balance * risk_pct) / (curr_price * (1 + fee_rate))
        if quantity <= min_qty:
            return OrderSignal(action="WAIT", quantity=0.0, reasoning="quantity below minimum")

        action = "BUY" if curr_price > prev_price else "SELL"
        reason = "price up" if action == "BUY" else "price down"
        return OrderSignal(action=action, quantity=quantity, type="MARKET", reasoning=reason)

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
