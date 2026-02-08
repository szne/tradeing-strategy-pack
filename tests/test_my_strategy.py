from __future__ import annotations

import pandas as pd
from strategy_pack.strategies.my_strategy import MyStrategy


def test_my_strategy_returns_signal() -> None:
    strategy = MyStrategy(params={"risk_pct": 0.01, "min_qty": 0.0})
    strategy.setup({"backtest": {"fee_rate": 0.001}})

    market = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    account = {"balance": 1000.0, "cash": 1000.0}

    signal = strategy.next_signal(market, account)

    assert signal.action in {"BUY", "SELL", "WAIT"}
