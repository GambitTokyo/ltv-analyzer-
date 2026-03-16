# LTV Analyzer

Kaplan–Meier × Weibull モデルによる LTV∞ 自動算出ツール

## セットアップ（初回のみ）

```bash
# Python 3.9以上が必要です
pip install -r requirements.txt
```

## 起動方法

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 が自動的に開きます。

## 入力CSVの形式

| 列名 | 内容 | 形式 |
|------|------|------|
| customer_id | 顧客ID | 任意の文字列 |
| start_date | 契約開始日 | YYYY-MM-DD |
| end_date | 解約日（継続中は空欄） | YYYY-MM-DD |
| revenue | 売上 | 数値 |

※ 列名は完全一致不要。`start`・`end`・`revenue`を含む列名は自動認識します。

## 機能

- KM生存曲線 → Weibullフィッティング → LTV∞ の自動算出
- CAC上限の算出（3モード選択）
- 観測期間別の暫定LTVテーブル
- Excel / PowerPoint / PDF エクスポート
- AIへの質問プロンプト自動生成（Claude・ChatGPT・Gemini対応）
