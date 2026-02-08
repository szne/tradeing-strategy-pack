from __future__ import annotations

from typing import Any, Dict

from trading_sdk.base_strategy import BaseStrategy
from strategy_pack.strategies.legacy.five_minute_momentum import FiveMinuteMomentumStrategy


class FiveMinuteMomentum12PctStrategy(FiveMinuteMomentumStrategy):
    def default_params(self) -> Dict[str, Any]:
        return {
            "ema_fast": 12,
            "ema_slow": 36,
            "rsi_period": 14,
            "rsi_momentum": 55,
            "donchian_lookback": 20,
            "breakout_buffer_pct": 0.0,
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "atr_trail_mult": 3.0,
            "atr_ma_period": 20,
            "atr_expand_mult": 1.1,
            "adx_period": 14,
            "adx_min": 17.0,
            "use_time_filter": True,
            "time_filter_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "time_filter_tz": "UTC",
            "risk_pct": 0.02,
            "max_position_pct": 0.3,
            "min_qty": 0.0,
            "max_leverage": 200.0,
            "leverage_cap": 200.0,
            "max_drawdown": 0.2,
            "slippage_bps": 2.0,
            "max_holding_bars": 288,
        }
