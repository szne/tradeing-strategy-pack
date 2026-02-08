# RELEASE Guide (strategy-pack)

## 1. タグ規約
- `vX.Y.Z` 形式のみ使用する。

## 2. 事前チェック
- `main` 最新化: `git checkout main && git pull origin main`
- 作業ツリー確認: `git status`
- 検証:
  - `strategy validate --path .`
  - `python -m pytest -q`

## 3. リリース手順
1. `pyproject.toml` の `version` を更新
2. リリースノートを作成
3. コミット: `git commit -am "🔖 release: vX.Y.Z"`
4. タグ: `git tag vX.Y.Z`
5. push: `git push origin main && git push origin vX.Y.Z`
6. GitHub Release を作成

## 4. リリースノート雛形
- [workspace-meta template](https://github.com/szne/tradeing-workspace-meta/blob/main/docs/templates/release_notes_template.md)

