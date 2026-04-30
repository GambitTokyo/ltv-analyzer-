# LTV Analyzer - Development Notes

## 日本語フォント文字化け再発防止チェックリスト

**新規 chart 描画関数を追加する際、必ず以下を実施:**

- [ ] 関数の冒頭または `plt.subplots(...)` の直前で **`setup_japanese_font()`**(`app_pro.py`)または **`_init()`**(`pptx_export.py`)を呼ぶ
- [ ] 英日両言語でチャートを生成して文字化けしないか確認
- [ ] PPT/PDF エクスポートでも文字化けしないか確認(セグメント別含む)

## 過去に文字化けが発生した箇所(再発防止のため記録)

| 媒体 | 場所 | 関数 | 修正日 |
|---|---|---|---|
| PPT P5 | 暫定 LTV - 観測期間別グラフ | `pptx_export._make_ltv_graph` | 2026-04 |
| PPT P7+ | セグメント別 LTV∞ 比較 | `pptx_export._make_bar_graph` | 2026-04 |
| PPT P7+ | セグメント別 Survival/Weibull(2 グラフ横並び) | `pptx_export.generate_pptx` line 520/535 | **2026-04-30** |
| PDF P4 | 暫定 LTV - 観測期間別グラフ | `app_pro.py` line 3353 | 2026-04 |
| PDF P5+ | セグメント別 LTV∞ バー | `app_pro.py` line 3501 | 2026-04 |
| PDF P5+ | セグメント別 Survival/Weibull | `app_pro.py` line 3625 | 2026-04 |

## フォント設定の仕組み

### app_pro.py 側

- **モジュール初期化**:line 2367-2381 で `_JP_FONT_PATH` 探索 + `fontManager.addfont(...)` で matplotlib に登録
- **共通ヘルパー**:`setup_japanese_font()`(line 2390)— rcParams 強制再適用、冪等
- **使用パターン**:全 chart 描画前に呼ぶ

### pptx_export.py 側

- **モジュール初期化**:line 19 `_init()` 関数内で同等処理
- **使用パターン**:`_make_ltv_graph` / `_make_bar_graph` / セグメント別 KM/Weibull で **必ず `_init()` 呼出**(冪等)

## Streamlit Cloud / Railway デプロイ時の必須要件

`packages.txt` に以下を含める:

```
fonts-ipafont-gothic
fonts-noto-cjk
```

これがないと Linux 環境で日本語フォントが見つからず、`setup_japanese_font()` が None を返してフォールバック失敗 → 文字化け再発。

## 自動テストの推奨(将来)

`tests/test_charts.py`:
```python
def test_all_charts_have_japanese_font():
    """全 chart 関数が setup_japanese_font() / _init() を呼んでいるか AST で確認"""
    import ast, inspect
    # ... AST 解析実装
```

## バージョン管理

- `app_pro.py` のバージョン番号を機能改修時に更新(現状 v369)
- `lang.py` の冒頭に `# -*- coding: utf-8 -*-` を必ず維持

## Gambit テンプレートの維持

PPT 生成では必ず `Presentation('LTV-analyzer.pptx')` で既存テンプレートを開くこと。新規作成すると スライドマスター・背景・カラー・フォント設定が失われる。
