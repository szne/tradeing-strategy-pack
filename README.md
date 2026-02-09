# strategy-pack

External strategy package for the trading engine.

## Install

```sh
pip install -e ../trading-sdk
pip install -e .
```

## Entry point

- group: `trading_system.strategies`
- name: `my_strategy`
- additional legacy entries:
  - `legacy_sample`
  - `legacy_5m_momentum`
  - `legacy_5m_reversion`
  - `legacy_5m_breakout`
  - `legacy_5m_impulse`
  - `legacy_5m_regime`
  - `legacy_5m_momentum_12pct`
  - `legacy_1m_trend_scalp`

## Legacy Strategies (migrated from engine)
- `strategy_pack/strategies/legacy/sample_strategy.py`
- `strategy_pack/strategies/legacy/five_minute_momentum.py`
- `strategy_pack/strategies/legacy/five_minute_reversion.py`
- `strategy_pack/strategies/legacy/five_minute_breakout.py`
- `strategy_pack/strategies/legacy/five_minute_impulse.py`
- `strategy_pack/strategies/legacy/five_minute_regime.py`
- `strategy_pack/strategies/legacy/five_minute_momentum_12pct.py`
- `strategy_pack/strategies/legacy/one_minute_trend_scalp.py`

## SDK 互換範囲
- `trading-sdk>=0.1,<0.2` を前提にしています。

## Coding Agent運用
- エージェント運用ルールは `CODEX.md` を参照してください。

## リリース
- 手順は `RELEASE.md` を参照してください。
