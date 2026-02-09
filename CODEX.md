# CODEX.md (strategy-pack)

## Purpose
`strategy-pack` で Strategy を高速に改善するための、コーディングエージェント向け運用ルール。

## Scope
- 主対象: `strategy_pack/strategies/<strategy_name>.py`
- 必要時のみ: `tests/test_<strategy_name>.py`

上記以外のファイル変更は、ユーザーが明示した場合のみ行う。

## Hard Rules
1. Strategy実装は `trading_sdk` 契約に従うこと（`BaseStrategy`, `OrderSignal`）。
2. `ccxt` / `core.*` / `trading-engine` 内部モジュールを import しないこと。
3. 1イテレーションでの変更対象は「1戦略ファイル」を基本とする。
4. 変更後は必ず `validate -> test -> backtest` の順で確認する。

## Standard Commands
```sh
# strategy-pack root
strategy validate --path .
strategy test --path .

# engine連携backtest
strategy backtest \
  --engine-root ../trading-engine \
  --config configs/default.yaml \
  --strategy strategy_pack.strategies.<strategy_name>:<ClassName> \
  --source synthetic \
  --rows 500
```

## Iteration Workflow
1. 目的を1行で定義（例: DD削減、勝率改善）。
2. 変更仮説を最大3つに絞る。
3. Strategyファイルのみ編集。
4. 上記コマンドで検証。
5. 差分と結果を短く記録して次ループへ。

## Commit Policy
- 1改善ループ = 1コミット
- メッセージは意図が分かる粒度にする
  - 例: `✨ feat(strategy): tighten entry filter for breakout`
