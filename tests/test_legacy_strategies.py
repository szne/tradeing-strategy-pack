from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_pack.strategies.legacy.five_minute_breakout import FiveMinuteBreakoutStrategy
from strategy_pack.strategies.legacy.five_minute_impulse import FiveMinuteImpulseStrategy
from strategy_pack.strategies.legacy.five_minute_momentum import FiveMinuteMomentumStrategy
from strategy_pack.strategies.legacy.five_minute_momentum_12pct import (
    FiveMinuteMomentum12PctStrategy,
)
from strategy_pack.strategies.legacy.five_minute_regime import FiveMinuteRegimeStrategy
from strategy_pack.strategies.legacy.five_minute_reversion import FiveMinuteReversionStrategy
from strategy_pack.strategies.legacy.one_minute_trend_scalp import OneMinuteTrendScalpStrategy
from strategy_pack.strategies.legacy.sample_strategy import SampleStrategy


def _market_frame(rows: int = 320, freq: str = "5min") -> pd.DataFrame:
    base = np.linspace(100.0, 120.0, rows)
    wave = np.sin(np.linspace(0.0, 10.0, rows))
    close = base + wave
    index = pd.date_range("2025-01-01", periods=rows, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1000.0),
        },
        index=index,
    )


def test_legacy_sample_strategy_smoke() -> None:
    strategy = SampleStrategy(params={"risk_pct": 0.01, "stop_loss_pct": 0.01})
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_momentum_strategy_smoke() -> None:
    strategy = FiveMinuteMomentumStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_reversion_strategy_smoke() -> None:
    strategy = FiveMinuteReversionStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_breakout_strategy_smoke() -> None:
    strategy = FiveMinuteBreakoutStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_impulse_strategy_smoke() -> None:
    strategy = FiveMinuteImpulseStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_regime_strategy_smoke() -> None:
    strategy = FiveMinuteRegimeStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_momentum_12pct_strategy_smoke() -> None:
    strategy = FiveMinuteMomentum12PctStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(_market_frame(), {"balance": 1000.0, "cash": 1000.0})
    assert signal.action in {"BUY", "SELL", "WAIT"}


def test_legacy_one_minute_trend_scalp_strategy_smoke() -> None:
    strategy = OneMinuteTrendScalpStrategy()
    strategy.setup({"backtest": {"fee_rate": 0.001}})
    signal = strategy.next_signal(
        _market_frame(rows=600, freq="1min"),
        {"balance": 1000.0, "cash": 1000.0},
    )
    assert signal.action in {"BUY", "SELL", "WAIT"}
