# -*- coding: utf-8 -*-
# lang.py — LTV Analyzer 言語/通貨/テキスト辞書
# ═══════════════════════════════════════════════════════════════
# v329: T() 関数 + フラットキー辞書 追加

# ── 通貨定義 ──────────────────────────────────────────────────

CURRENCIES = {
    'JPY': {'symbol': '¥',  'prefix': True,  'decimal': 0},
    'USD': {'symbol': '$',  'prefix': True,  'decimal': 0},
    'EUR': {'symbol': '€',  'prefix': True,  'decimal': 0},
    'GBP': {'symbol': '£',  'prefix': True,  'decimal': 0},
    'KRW': {'symbol': '₩',  'prefix': True,  'decimal': 0},
    'CNY': {'symbol': '¥',  'prefix': True,  'decimal': 0},
    'TWD': {'symbol': 'NT$','prefix': True,  'decimal': 0},
    'INR': {'symbol': '₹',  'prefix': True,  'decimal': 0},
    'none':{'symbol': '',   'prefix': True,  'decimal': 0},
}

LANG_DEFAULTS = {
    'ja': 'JPY',
    'en': 'USD',
}

# ── 通貨フォーマッター ────────────────────────────────────────

def fmt_c(val, cur='JPY', dp=None):
    """通貨フォーマット — fmt_c(12345, 'JPY') → '¥12,345'  /  fmt_c(123.45, 'USD', 2) → '$123.45'"""
    c = CURRENCIES.get(cur, CURRENCIES['JPY'])
    d = dp if dp is not None else c['decimal']
    s = c['symbol']
    num = f'{val:,.{d}f}'
    if c['prefix']:
        return f'{s}{num}'
    else:
        return f'{num}{s}'


def cur_symbol(cur='JPY'):
    """通貨記号だけ返す — Plotly tickprefix等に使用"""
    return CURRENCIES.get(cur, CURRENCIES['JPY'])['symbol']


def cur_decimal(cur='JPY'):
    """小数桁数を返す — 条件分岐用"""
    return CURRENCIES.get(cur, CURRENCIES['JPY'])['decimal']


# ══════════════════════════════════════════════════════════════
# テキスト辞書 — T(key) で言語切り替え
# ══════════════════════════════════════════════════════════════

_LANG = 'ja'   # グローバル言語設定（set_lang() で変更）

def set_lang(lang: str):
    """言語をセット — app起動時に1回呼ぶ"""
    global _LANG
    _LANG = lang

def get_lang() -> str:
    return _LANG

def T(key: str, **kwargs) -> str:
    """テキスト辞書から取得。f-string的な埋め込みは kwargs で渡す。
    例: T('sidebar_gpm_caption', gpm='50%')
    キーが見つからない場合はキー名をそのまま返す（開発中のフォールバック）。
    """
    entry = _DICT.get(key)
    if entry is None:
        return key  # fallback: キー名をそのまま表示
    text = entry.get(_LANG, entry.get('en', key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass  # フォーマット失敗時はテンプレートのまま返す
    return text


# ══════════════════════════════════════════════════════════════
# ビジネスタイプ・課金サイクルの内部キー定数
# ══════════════════════════════════════════════════════════════

BIZ_SUBSCRIPTION = 'subscription'
BIZ_SPOT         = 'spot'

BILLING_CALENDAR_MONTHLY = 'calendar_monthly'
BILLING_ANNUAL_365       = 'annual_365'
BILLING_CUSTOM_DAYS      = 'custom_days'
BILLING_FIXED_30         = 'fixed_30'
BILLING_DAILY_SPOT       = 'daily_spot'


# ══════════════════════════════════════════════════════════════
# 辞書本体（フラットキー方式）
# ══════════════════════════════════════════════════════════════
# 命名規則: {セクション}_{概要}
#   sidebar_*   — サイドバーUI
#   main_*      — メインエリアUI
#   summary_*   — 分析結果サマリー・結論文
#   chart_*     — Plotlyグラフラベル
#   export_*    — エクスポート関連
#   seg_*       — セグメント分析
#   prompt_*    — AI Prompt Generator
#   sample_*    — サンプルデータ関連
#   common_*    — 共通（日/年 等）
# ══════════════════════════════════════════════════════════════

_DICT = {

    # ── 共通 ──────────────────────────────────────────────────
    'common_days':        {'ja': '日', 'en': 'd'},
    'common_years':       {'ja': '年', 'en': 'yr'},
    'common_days_unit':   {'ja': '{n}日', 'en': '{n}d'},
    'common_years_paren': {'ja': '約{y:.1f}年', 'en': '~{y:.1f}yr'},
    'common_customers':   {'ja': '件', 'en': ''},
    'common_none':        {'ja': 'なし', 'en': 'None'},
    'common_on':          {'ja': 'ON', 'en': 'ON'},
    'common_off':         {'ja': 'OFF', 'en': 'OFF'},
    'common_acquisition': {'ja': '契約', 'en': 'acquisition'},
    'common_first_purchase': {'ja': '初回購入', 'en': 'first purchase'},
    'common_start_date':    {'ja': '契約開始日', 'en': 'start date'},
    'common_first_purchase_date': {'ja': '初回購入日', 'en': 'first purchase date'},

    # ── サイドバー: 言語/通貨 ─────────────────────────────────
    'sidebar_lang_currency':    {'ja': '### 言語 / 通貨', 'en': '### Language / Currency'},
    'sidebar_lang_label':       {'ja': '言語 / Language', 'en': 'Language'},
    'sidebar_cur_label':        {'ja': '通貨 / Currency', 'en': 'Currency'},

    # ── サイドバー: データ入力 ─────────────────────────────────
    'sidebar_data_input':       {'ja': '### データ入力', 'en': '### Data Input'},
    'sidebar_sample_hint':      {'ja': 'サンプルデータを選択してお試しください。', 'en': 'Select sample data to try it out.'},
    'sidebar_sample_placeholder': {'ja': '（選択してください）', 'en': '(Select)'},
    'sidebar_sample_select':    {'ja': 'サンプルデータを選択', 'en': 'Select sample data'},
    'sidebar_sample_dl':        {'ja': 'サンプルCSVをダウンロード（データフォーマット確認用）', 'en': 'Download sample CSV (check data format)'},
    'sidebar_upload_csv':       {'ja': 'CSVをアップロード', 'en': 'Upload CSV'},

    # ── サイドバー: サンプルデータ名 ──────────────────────────
    'sample_elearn':   {'ja': 'サブスク型：動画学習（日割りOFF）', 'en': 'Subscription: e-Learning (no prorate)'},
    'sample_cowork':   {'ja': 'サブスク型：コワーキング（日割りON）', 'en': 'Subscription: Coworking (prorate ON)'},
    'sample_skincare': {'ja': '都度購入型：コスメ系EC', 'en': 'Spot Purchase: Skincare EC'},

    # ── サイドバー: サンプルデータのレポート情報 ───────────────
    'sample_elearn_title':    {'ja': '動画学習プラットフォーム 顧客LTV分析', 'en': 'e-Learning Platform Customer LTV Analysis'},
    'sample_elearn_client':   {'ja': 'LearnPlus株式会社', 'en': 'LearnPlus Inc.'},
    'sample_elearn_analyst':  {'ja': 'マーケティング部', 'en': 'Marketing Dept.'},
    'sample_cowork_title':    {'ja': 'コワーキングスペース 会員LTV分析', 'en': 'Coworking Space Member LTV Analysis'},
    'sample_cowork_client':   {'ja': 'WorkHub株式会社', 'en': 'WorkHub Inc.'},
    'sample_cowork_analyst':  {'ja': '事業企画部', 'en': 'Business Planning Dept.'},
    'sample_skincare_title':  {'ja': 'コスメ系EC 顧客LTV分析', 'en': 'Skincare EC Customer LTV Analysis'},
    'sample_skincare_client': {'ja': 'GlowSkin株式会社', 'en': 'GlowSkin Inc.'},
    'sample_skincare_analyst':{'ja': 'CRM推進チーム', 'en': 'CRM Team'},

    # ── サイドバー: 異常値処理 ─────────────────────────────────
    'sidebar_outlier':          {'ja': '### 異常値処理', 'en': '### Outlier Handling'},
    'sidebar_outlier_upper':    {'ja': '上位除外 (%)', 'en': 'Upper cutoff (%)'},
    'sidebar_outlier_lower':    {'ja': '下位除外 (%)', 'en': 'Lower cutoff (%)'},
    'sidebar_outlier_upper_help': {'ja': '累計金額の上位○%を除外します。0%で除外なし。', 'en': 'Exclude top X% by cumulative revenue. 0% = no exclusion.'},
    'sidebar_outlier_lower_help': {'ja': '累計金額の下位○%を除外します。0%で除外なし。', 'en': 'Exclude bottom X% by cumulative revenue. 0% = no exclusion.'},
    'sidebar_outlier_caption':  {'ja': '売上分布のヒストグラムとカットラインが分析結果の手前に表示されます。分布を確認してから除外率を調整してください。',
                                 'en': 'A revenue distribution histogram with cut lines will be displayed before the analysis results. Adjust the exclusion rate after reviewing the distribution.'},

    # ── サイドバー: ビジネスタイプ ─────────────────────────────
    'sidebar_biz_type':         {'ja': '### ビジネスタイプ', 'en': '### Business Type'},
    'sidebar_biz_type_label':   {'ja': 'ビジネスタイプ', 'en': 'Business Type'},
    'biz_subscription':         {'ja': 'サブスク・継続課金型', 'en': 'Subscription'},
    'biz_spot':                 {'ja': '都度購入型', 'en': 'Spot Purchase'},

    # ── サイドバー: サブスク設定 ───────────────────────────────
    'sidebar_sub_caption':      {'ja': '解約日（end_date）をベースに離脱を判定します。end_dateが空欄の顧客は継続中として扱われます。',
                                 'en': 'Churn is determined based on end_date. Customers with blank end_date are treated as active.'},
    'sidebar_billing_period':   {'ja': '契約期間', 'en': 'Billing Period'},
    'billing_monthly_calendar': {'ja': '月額（カレンダーベース）', 'en': 'Monthly (calendar-based)'},
    'billing_annual_365':       {'ja': '年額（365日固定）', 'en': 'Annual (365 days)'},
    'billing_custom_days':      {'ja': 'カスタム入力（日数固定）', 'en': 'Custom (fixed days)'},
    'sidebar_custom_cycle_days':{'ja': '契約日数', 'en': 'Billing cycle (days)'},
    'sidebar_billing_caption':  {'ja': '月額：毎月同じ日に更新（例：5/15契約 → 6/15・7/15…）。年額：365日固定。カスタム：隔月・四半期など任意の日数。',
                                 'en': 'Monthly: renews on the same date each month (e.g., 5/15 → 6/15, 7/15…). Annual: 365-day fixed. Custom: any number of days.'},
    'sidebar_prorate_label':    {'ja': '解約時の日割り計算あり', 'en': 'Prorate on cancellation'},
    'sidebar_prorate_caption':  {'ja': 'OFFの場合、解約日を契約更新日に丸めます（一般的なサブスク）。ONの場合、実際の解約日をそのまま使用します。',
                                 'en': 'OFF: rounds cancellation date to the next renewal date (typical for subscriptions). ON: uses the actual cancellation date.'},

    # ── サイドバー: 都度購入設定 ───────────────────────────────
    'sidebar_spot_caption':     {'ja': '最終購買日（last_purchase_date）をベースに休眠判定します。CSVに last_purchase_date 列が必要です。',
                                 'en': 'Dormancy is determined based on last_purchase_date. CSV must include a last_purchase_date column.'},
    'sidebar_dormancy_period':  {'ja': '休眠判定期間', 'en': 'Dormancy Period'},
    'dormancy_180d':            {'ja': '180日', 'en': '180 days'},
    'dormancy_365d':            {'ja': '365日', 'en': '365 days'},
    'dormancy_730d':            {'ja': '730日', 'en': '730 days'},
    'dormancy_custom':          {'ja': 'カスタム入力', 'en': 'Custom'},
    'sidebar_dormancy_caption': {'ja': 'あなたのビジネスに合った休眠顧客の認定期間を設定してください。判断が難しい場合は、自社データで最終購買から再購買が発生しなくなる日数を確認することをお勧めします。',
                                 'en': 'Set the dormancy period that fits your business. If unsure, check when repurchases stop occurring after the last purchase in your data.'},
    'sidebar_dormancy_days_label': {'ja': '休眠判定日数', 'en': 'Dormancy days'},

    # ── サイドバー: GPM ───────────────────────────────────────
    'sidebar_gpm_label':        {'ja': '粗利率：売上に占める（売上－変動費）の割合', 'en': 'Gross profit as a percentage of revenue'},
    'sidebar_gpm_caption':      {'ja': 'LTV∞の表示は売上ベース。CAC上限の算出には粗利ベース（売上×{gpm}）を使用します。',
                                 'en': 'LTV∞ is shown on a revenue basis. CAC cap is calculated on a gross profit basis (revenue × {gpm}).'},

    # ── サイドバー: CAC上限 ───────────────────────────────────
    'sidebar_cac':              {'ja': '### CAC 上限', 'en': '### CAC Cap'},
    'sidebar_cac_slider':       {'ja': 'N（LTV:CAC = N:1）', 'en': 'N (LTV:CAC = N:1)'},
    'sidebar_cac_caption':      {'ja': '例：LTV:CAC = 3:1 の場合、CAC上限 = LTV（粗利）÷ 3',
                                 'en': 'e.g., If LTV:CAC = 3:1, CAC cap = LTV (gross profit) ÷ 3'},

    # ── サイドバー: セグメント分析 ─────────────────────────────
    'sidebar_segment':          {'ja': '### セグメント分析', 'en': '### Segment Analysis'},
    'sidebar_seg_input_label':  {'ja': 'セグメント列名（カンマ区切りで複数指定可）', 'en': 'Segment column names (comma-separated)'},
    'sidebar_seg_placeholder':  {'ja': '例：plan, channel, age_group（最大5列）', 'en': 'e.g., plan, channel, age_group (max 5)'},
    'sidebar_seg_caption':      {'ja': 'CSVの列名をカンマ区切りで入力してください。セグメント別のLTV∞を自動比較し、優先獲得セグメントを特定します。\n1列あたり最大50種類・最大5列。代表的な軸：プラン・チャネル・年齢層・性別・地域など。',
                                 'en': 'Enter CSV column names separated by commas. LTV∞ is compared across segments to identify priority acquisition targets.\nMax 50 unique values per column, up to 5 columns. Common axes: plan, channel, age group, gender, region, etc.'},

    # ── サイドバー: 表示件数 ──────────────────────────────────
    'sidebar_display_limit':    {'ja': '### 表示件数', 'en': '### Display Limit'},
    'sidebar_display_slider':   {'ja': '詳細表示（暫定LTV・生存曲線）の上位N件', 'en': 'Top N segments for detailed view (interim LTV & survival curve)'},
    'sidebar_display_caption':  {'ja': 'セグメント（例：都道府県）の項目数（例：47）が多いほどブラウザの描画に時間がかかります。表示する上位N項目を絞ることで速度が大幅に改善されます。\nエクスポートされる各ファイルには全項目出力されます。\nまず上位5項目で傾向を確認し、必要に応じて増やすことをお勧めします。',
                                 'en': 'The more segment values (e.g., 47 prefectures), the longer the browser takes to render. Limiting to the top N items significantly improves speed.\nAll items are included in exported files.\nStart with the top 5 to see trends, then increase as needed.'},

    # ── サイドバー: レポート情報 ──────────────────────────────
    'sidebar_report_info':      {'ja': '### レポート情報', 'en': '### Report Information'},
    'sidebar_report_title':     {'ja': 'レポートタイトル', 'en': 'Report Title'},
    'sidebar_report_title_ph':  {'ja': '月額SaaS顧客LTV分析など', 'en': 'e.g., Monthly SaaS Customer LTV Analysis'},
    'sidebar_client_name':      {'ja': 'クライアント名', 'en': 'Client Name'},
    'sidebar_client_name_ph':   {'ja': '会社・ブランド・商品/サービスなど', 'en': 'Company, brand, product/service, etc.'},
    'sidebar_analyst_name':     {'ja': '作成者', 'en': 'Analyst'},
    'sidebar_analyst_name_ph':  {'ja': '氏名・チーム・部署・組織など', 'en': 'Name, team, department, etc.'},

    # ── メイン: ファイル未選択時 ──────────────────────────────
    'main_no_file_info':        {'ja': 'サイドバーからCSVをアップロードするか、サンプルデータを選択してください。',
                                 'en': 'Upload a CSV from the sidebar or select sample data.'},
    'main_csv_format_title':    {'ja': 'CSV フォーマット', 'en': 'CSV Format'},
    'main_analysis_flow_title': {'ja': '分析の流れ', 'en': 'Analysis Flow'},
    'main_step1_title':         {'ja': '① KM法', 'en': '① KM Method'},
    'main_step1_desc':          {'ja': '実測データから生存曲線を作成', 'en': 'Create survival curve from observed data'},
    'main_step2_title':         {'ja': '② Weibull', 'en': '② Weibull'},
    'main_step2_desc':          {'ja': '連続曲線にフィッティング', 'en': 'Fit to a continuous curve'},
    'main_step3_title':         {'ja': '③ LTV∞', 'en': '③ LTV∞'},
    'main_step3_desc':          {'ja': '生存積分 × ARPU で算出', 'en': 'Survival integral × ARPU'},
    'main_step4_title':         {'ja': '④ CAC上限', 'en': '④ CAC Cap'},
    'main_step4_desc':          {'ja': 'LTV比率で逆算', 'en': 'Reverse-calculate from LTV ratio'},

    # ── メイン: CSVテーブル列ヘッダー説明 ─────────────────────
    'csv_col_customer_id':      {'ja': '顧客ID', 'en': 'Customer ID'},
    'csv_col_customer_id_desc': {'ja': '任意の文字列', 'en': 'Any string'},
    'csv_col_start_date':       {'ja': '契約開始日 / 初回購入日', 'en': 'Start date / first purchase date'},
    'csv_col_end_date':         {'ja': '解約日（サブスク向け・継続中は**空欄**）', 'en': 'End date (subscription: blank if active)'},
    'csv_col_last_purchase':    {'ja': '最終購買日（都度購入向け・任意）', 'en': 'Last purchase date (spot purchase: optional)'},
    'csv_col_revenue':          {'ja': '**累計売上**', 'en': '**Cumulative revenue**'},
    'csv_col_segment':          {'ja': '**Advanced機能**：プラン・チャネル・年齢層など', 'en': '**Advanced**: plan, channel, age group, etc.'},
    'csv_note_segment':         {'ja': 'Advanced版では必ずセグメント列を追加してください。', 'en': 'Always include segment columns for the Advanced version.'},
    'csv_note_auto_detect':     {'ja': '列名は完全一致でなくてもOKです。`start`・`end`・`last`・`revenue`を含む列名は自動認識します。',
                                 'en': 'Column names don\'t need to match exactly. Columns containing `start`, `end`, `last`, or `revenue` are auto-detected.'},
    'csv_note_arpu':            {'ja': 'ARPU daily はビジネスタイプに応じて自動計算されます。', 'en': 'Daily ARPU is automatically calculated based on business type.'},
    'csv_note_segment_limit':   {'ja': 'セグメント列は1列あたり最大50種類のユニーク値まで対応しています（都道府県47個も対応）。',
                                 'en': 'Each segment column supports up to 50 unique values (e.g., 47 prefectures).'},

    # ── メイン: データ読み込みメッセージ ───────────────────────
    'main_col_missing':         {'ja': '列が見つかりません: {missing}\n\n列名に `start`・`end`・`revenue` を含む列が必要です。サイドバーからサンプルCSVをダウンロードして形式を確認してください。',
                                 'en': 'Columns not found: {missing}\n\nColumns containing `start`, `end`, and `revenue` are required. Download a sample CSV from the sidebar to check the format.'},
    'main_bad_dates':           {'ja': '{n}行で `start_date` が読み取れませんでした。該当行は除外します。',
                                 'en': 'Could not parse `start_date` in {n} rows. Those rows will be excluded.'},
    'main_data_loaded':         {'ja': '全{n}件のデータを正常に読み込みました。',
                                 'en': 'Successfully loaded {n} records.'},
    'main_data_too_few':        {'ja': '有効なデータが10件未満です。分析には最低10件の顧客データが必要です。',
                                 'en': 'Fewer than 10 valid records. At least 10 customer records are required for analysis.'},
    'main_data_error':          {'ja': 'データ読み込みエラー: {e}\n\nCSVの形式を確認してください。サンプルCSVをダウンロードして参照してください。',
                                 'en': 'Data loading error: {e}\n\nPlease check the CSV format. Download a sample CSV for reference.'},
    'main_dormant_converted':   {'ja': '{n}件を休眠顧客（最終購買から{days}日超）として実質離脱に変換しました。',
                                 'en': 'Converted {n} records to dormant (>{days} days since last purchase).'},
    'main_same_day_corrected':  {'ja': '{n}件：{label}と終端日が同日のため1日に補正しました。',
                                 'en': '{n} records: {label} and end date were the same day; corrected to 1 day.'},
    'main_future_excluded':     {'ja': '{n}件：start_dateが未来の日付のため除外しました（入力ミスの可能性）。',
                                 'en': '{n} records: start_date is in the future; excluded (possible data entry error).'},
    'main_outlier_removed':     {'ja': '{n}件を異常値として除外しました（{parts}）。除外率 {pct:.1f}%、残り {remaining}件で分析します。',
                                 'en': '{n} records excluded as outliers ({parts}). Exclusion rate {pct:.1f}%, analyzing {remaining} remaining records.'},
    'main_weibull_fail':        {'ja': 'Weibullフィッティングに失敗しました。解約済み顧客が少なすぎる可能性があります（最低10件の解約データが必要）。',
                                 'en': 'Weibull fitting failed. There may be too few churned customers (at least 10 churn events required).'},

    # ── メイン: 分析結果サマリー ──────────────────────────────
    'main_summary_title':       {'ja': '分析結果サマリー', 'en': 'Analysis Summary'},
    'summary_k_early':          {'ja': '初期離脱大・投資回収が比較的長期', 'en': 'High early churn — longer payback period'},
    'summary_k_late':           {'ja': '継続後に解約増・投資回収が比較的短期', 'en': 'Churn increases over time — shorter payback'},
    'summary_rev_basis':        {'ja': '売上ベース', 'en': 'Revenue basis'},
    'summary_cac_gp_basis':     {'ja': '（粗利ベース）', 'en': '(GP basis)'},
    'summary_k_desc_long':      {'ja': '値が大きいほどLTV∞到達が長期化', 'en': 'Higher = longer time to reach LTV∞'},
    'summary_r2_note':          {'ja': '0.9以上が理想 / 1.0が最高精度', 'en': '≥0.9 ideal / 1.0 = perfect fit'},
    'summary_r2_warning':       {'ja': 'R²={r2:.3f} — フィット精度がやや低めです。データ点数を増やすか、観測期間を見直してください。',
                                 'en': 'R²={r2:.3f} — Fit quality is somewhat low. Consider increasing data volume or reviewing the observation period.'},
    'summary_conclusion':       {'ja': '結論', 'en': 'Conclusion'},

    # ── メイン: チャート ──────────────────────────────────────
    'chart_reliability_title':  {'ja': '分析モデルの信頼性', 'en': 'Model Reliability'},
    'chart_survival_caption':   {'ja': 'Survival Curve：実測のKM曲線（実線）にWeibullモデルをフィット（破線）。右に伸びるほど顧客が長く継続している。',
                                 'en': 'Survival Curve: Observed KM curve (solid) fitted with Weibull model (dashed). Extending right = longer customer retention.'},
    'chart_linearization_caption': {'ja': 'Weibull Linearization Plot：生存率を対数変換して直線化したもの。R²が1.0に近いほどWeibullモデルのフィット精度が高い。',
                                    'en': 'Weibull Linearization Plot: Log-transformed survival rate linearized. R² closer to 1.0 = better Weibull fit.'},
    'chart_interim_ltv_title':  {'ja': '暫定 LTV — 観測期間別', 'en': 'Interim LTV by Observation Period'},
    'chart_revenue_dist':       {'ja': '累計売上の分布', 'en': 'Revenue Distribution'},
    'chart_cumulative_rev':     {'ja': '累計売上', 'en': 'Cumulative Revenue'},
    'chart_customer_count':     {'ja': '顧客数', 'en': 'Customer Count'},
    'chart_amount':             {'ja': '金額', 'en': 'Amount'},

    # ── セクションタイトル ────────────────────────────────────
    'section_export':           {'ja': 'Export', 'en': 'Export'},
    'section_ai_prompt':        {'ja': 'AI Prompt Generator', 'en': 'AI Prompt Generator'},
    'section_segment':          {'ja': 'セグメント別 LTV 分析', 'en': 'Segment LTV Analysis'},

    # ── AI Prompt Generator ───────────────────────────────────
    'prompt_help_box':          {'ja': 'この結果の読み方や戦略への活用方法がわからない場合は、以下のプロンプトをClaude・ChatGPT・Geminiにコピペしてください。テキストボックス右上の <b>コピーアイコン</b> をクリックすると全文コピーできます。',
                                 'en': 'If you\'re unsure how to interpret these results or apply them strategically, copy the prompts below into Claude, ChatGPT, or Gemini. Click the <b>copy icon</b> at the top right of each text box to copy the full text.'},
    'prompt_tab1':              {'ja': '分析結果の解釈', 'en': 'Interpreting Results'},
    'prompt_tab2':              {'ja': 'マーケティング意思決定', 'en': 'Marketing Decisions'},
    'prompt_tab3':              {'ja': '分析の限界と改善', 'en': 'Limitations & Improvements'},

    # ── セグメント分析メッセージ ──────────────────────────────
    'seg_col_not_found':        {'ja': '以下の列が見つかりませんでした: {cols}', 'en': 'The following columns were not found: {cols}'},
    'seg_too_many':             {'ja': '`{col}` のユニーク値が{n}個あります。50個以下にしてください。', 'en': '`{col}` has {n} unique values. Please limit to 50 or fewer.'},
    'seg_many_values':          {'ja': '`{col}` のユニーク値が{n}個あります。計算に少し時間がかかります。', 'en': '`{col}` has {n} unique values. Calculation may take a moment.'},
    'seg_insufficient_data':    {'ja': '分析に十分なデータがありませんでした。', 'en': 'Insufficient data for analysis.'},
    'seg_detail_limit':         {'ja': '詳細表示 — 上位 {limit} 件（全 {total} 件）', 'en': 'Details — top {limit} of {total} segments'},
    'seg_weighted_avg':         {'ja': '加重平均', 'en': 'Weighted Avg'},
    'seg_top_pick':             {'ja': 'TOP PICK', 'en': 'TOP PICK'},

    # ── 結論文テンプレート（動的生成用） ──────────────────────
    'conclusion_spot_early_churn': {
        'ja': 'k={k:.3f}の初期離脱型です。初回購入後{period}以内に再購入しなかった顧客（単発購入）は{rate:.0f}%です。リピートした顧客の多くは初回購入からλ={lam:.0f}日（約{lam_y:.1f}年）以上購買を継続する傾向があります。LTV∞は{ltv}でCAC上限は{cac}ですが、投資回収は比較的長期になるため、暫定LTVテーブルで現実的な回収期間を確認してCACを設計してください。',
        'en': 'With k={k:.3f}, this is an early-churn pattern. {rate:.0f}% of customers did not repurchase within {period} after their first purchase (one-time buyers). Repeat customers tend to continue purchasing for λ={lam:.0f} days (~{lam_y:.1f} years) or more. LTV∞ is {ltv} with a CAC cap of {cac}, but payback takes relatively long — check the interim LTV table for realistic payback periods when designing your CAC.',
    },
    'conclusion_spot_late_churn': {
        'ja': 'k={k:.3f}の逓増離脱型です。初回購入後{period}以内に再購入しなかった顧客（単発購入）は{rate:.0f}%です。リピートした顧客の多くは初回購入からλ={lam:.0f}日（約{lam_y:.1f}年）以上購買を継続する傾向があります。LTV∞は{ltv}でCAC上限は{cac}、比較的短期での投資回収が見込めます。',
        'en': 'With k={k:.3f}, this is an increasing-churn pattern. {rate:.0f}% of customers did not repurchase within {period} after their first purchase (one-time buyers). Repeat customers tend to continue purchasing for λ={lam:.0f} days (~{lam_y:.1f} years) or more. LTV∞ is {ltv} with a CAC cap of {cac} — relatively short payback is expected.',
    },
    'conclusion_sub_early_churn': {
        'ja': 'k={k:.3f}の初期離脱型です。最初の契約期間のみで解約した顧客は{rate:.0f}%です。初期を乗り越えた顧客の多くはλ={lam:.0f}日（約{lam_y:.1f}年）以上継続する傾向があります。LTV∞は{ltv}でCAC上限は{cac}ですが、投資回収は比較的長期になるため、暫定LTVテーブルで現実的な回収期間を確認してCACを設計してください。',
        'en': 'With k={k:.3f}, this is an early-churn pattern. {rate:.0f}% of customers churned within their first billing period. Those who survive the initial period tend to stay for λ={lam:.0f} days (~{lam_y:.1f} years) or more. LTV∞ is {ltv} with a CAC cap of {cac}, but payback takes relatively long — check the interim LTV table for realistic payback periods when designing your CAC.',
    },
    'conclusion_sub_late_churn': {
        'ja': 'k={k:.3f}の逓増離脱型です。最初の契約期間のみで解約した顧客は{rate:.0f}%です。初期を乗り越えた顧客の多くはλ={lam:.0f}日（約{lam_y:.1f}年）以上継続する傾向があります。LTV∞は{ltv}でCAC上限は{cac}、比較的短期での投資回収が見込めます。',
        'en': 'With k={k:.3f}, this is an increasing-churn pattern. {rate:.0f}% of customers churned within their first billing period. Those who survive the initial period tend to stay for λ={lam:.0f} days (~{lam_y:.1f} years) or more. LTV∞ is {ltv} with a CAC cap of {cac} — relatively short payback is expected.',
    },
    'conclusion_r2_high':       {'ja': 'R²={r2:.3f}はモデル精度が非常に高く、この推定値は意思決定に十分活用できます。',
                                 'en': 'R²={r2:.3f} indicates very high model accuracy — these estimates are reliable for decision-making.'},
    'conclusion_r2_mid':        {'ja': 'R²={r2:.3f}は許容範囲内の精度です。推定値に±15%程度の幅を見込んでください。',
                                 'en': 'R²={r2:.3f} is within acceptable range. Allow for approximately ±15% margin in estimates.'},
    'conclusion_r2_low':        {'ja': 'R²={r2:.3f}はやや低めです。推定値の信頼性に注意してください。',
                                 'en': 'R²={r2:.3f} is somewhat low. Exercise caution with the reliability of estimates.'},

    # ── Excel エクスポート ─────────────────────────────────────
    'excel_summary_title':      {'ja': 'LTV分析 サマリー', 'en': 'LTV Analysis Summary'},
    'excel_client':             {'ja': 'クライアント', 'en': 'Client'},
    'excel_analyst':            {'ja': '作成者', 'en': 'Analyst'},
    'excel_none':               {'ja': '除外なし', 'en': 'No exclusion'},

    # ── PPTXエクスポート ──────────────────────────────────────
    'pptx_table_howto':         {'ja': 'このテーブルの読み方', 'en': 'How to read this table'},
    'pptx_conclusion':          {'ja': '結論', 'en': 'Conclusion'},

}
