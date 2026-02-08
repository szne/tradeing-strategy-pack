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

## Legacy Strategies (migrated from engine)
- `strategy_pack/strategies/legacy/sample_strategy.py`
- `strategy_pack/strategies/legacy/five_minute_momentum.py`
- `strategy_pack/strategies/legacy/five_minute_reversion.py`

## SDK 互換範囲
- `trading-sdk>=0.1,<0.2` を前提にしています。

## リリース
- 手順は `RELEASE.md` を参照してください。
