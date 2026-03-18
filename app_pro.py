import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import gamma, gammainc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import io
import warnings
import calendar
import plotly.graph_objects as go
from scipy.optimize import brentq
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="LTV Analyzer PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0d0d0d; color: #e8e4dc; }
.metric-card {
    background: #0d1f2d;
    border: 1px solid #1a3a4a;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    color: #56b4d3;
    line-height: 1.1;
    word-break: break-all;
}
.metric-label {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: #56b4d3;
    border-bottom: 1px solid #2a4a5a;
    padding-bottom: 6px;
    margin: 28px 0 16px 0;
}
.prompt-box {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #56b4d3;
    border-radius: 8px;
    padding: 16px;
    font-family: 'DM Sans', monospace;
    font-size: 0.85rem;
    color: #ccc;
    white-space: pre-wrap;
    word-break: break-word;
}
.help-box {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.82rem;
    color: #777;
    margin-top: 8px;
}
.tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    color: #888;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────
plt.style.use('dark_background')
rcParams['font.family'] = 'DejaVu Sans'
for _k, _v in {
    'figure.facecolor': '#111820', 'axes.facecolor': '#111820',
    'axes.edgecolor': '#1a3040', 'axes.labelcolor': '#7ab8cc',
    'xtick.color': '#4a8a9a', 'ytick.color': '#4a8a9a',
    'grid.color': '#1a3040', 'grid.linewidth': 0.5,
    'axes.unicode_minus': False,
}.items():
    rcParams[_k] = _v


ACCENT  = '#56b4d3'
ACCENT2 = '#a8dadc'
ACCENT3 = '#1d6fa4'

# ══════════════════════════════════════════════════════════════
# Analysis functions
# ══════════════════════════════════════════════════════════════

def compute_km(df):
    df = df.sort_values('duration').reset_index(drop=True)
    S, records = 1.0, []
    for t in np.sort(df['duration'].unique()):
        n_risk   = (df['duration'] >= t).sum()
        n_events = ((df['duration'] == t) & (df['event'] == 1)).sum()
        if n_risk > 0:
            S *= (1 - n_events / n_risk)
        records.append({'t': t, 'S': S, 'n_events': n_events, 'n_risk': n_risk})
    return pd.DataFrame(records)

def fit_weibull(km_df):
    fd = km_df[(km_df['t'] > 0) & (km_df['S'] > 0) & (km_df['S'] < 1)].copy()
    fd['ln_t']         = np.log(fd['t'])
    fd['ln_neg_ln_S']  = np.log(-np.log(fd['S']))
    fd = fd.replace([np.inf, -np.inf], np.nan).dropna()
    if len(fd) < 3:
        return None, None, None, None
    k, b, r, *_ = stats.linregress(fd['ln_t'], fd['ln_neg_ln_S'])
    lam = np.exp(-b / k)
    return k, lam, r**2, fd

def ltv_inf(k, lam, arpu):
    surv_int = lam * gamma(1 + 1/k)
    return surv_int * arpu, surv_int

def ltv_horizon(k, lam, arpu, h):
    x = (h / lam) ** k
    return lam * gamma(1 + 1/k) * gammainc(1 + 1/k, x) * arpu

def weibull_s(t, k, lam):
    return np.exp(-(t / lam) ** k)

# ══════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📥 データ入力")

    # サンプルデータ生成
    np.random.seed(42)
    n_sample = 10000
    today_ts = pd.Timestamp.today()
    start_dates = pd.date_range('2022-01-01', '2024-06-30', periods=n_sample)
    survival_days = np.random.weibull(1.2, n_sample) * 400
    churned = np.random.random(n_sample) < 0.65

    # ── サブスク用サンプル ──
    # end_date: 解約済みは解約日、継続中は空欄
    # revenue_total: 月額×リニューアル回数の累計売上
    monthly_fee = np.random.choice([300, 500, 980], n_sample, p=[0.5, 0.35, 0.15])
    end_dates_sub = []
    revenues_sub  = []
    for i in range(n_sample):
        sd = start_dates[i]
        if churned[i]:
            ed = sd + pd.Timedelta(days=int(survival_days[i]))
            if ed < today_ts:
                end_dates_sub.append(ed.strftime('%Y-%m-%d'))
                months = max(1, round((ed - sd).days / 30))
            else:
                end_dates_sub.append('')
                months = max(1, round((today_ts - sd).days / 30))
        else:
            end_dates_sub.append('')
            months = max(1, round((today_ts - sd).days / 30))
        revenues_sub.append(monthly_fee[i] * months)

    plans    = np.random.choice(['月額300', '月額500', '月額980'], n_sample, p=[0.5, 0.35, 0.15])
    channels = np.random.choice(['SNS', '検索広告', '紹介', 'オーガニック'], n_sample, p=[0.3, 0.3, 0.2, 0.2])
    ages     = np.random.choice(['10代以下', '20代', '30代', '40代', '50代以上'], n_sample, p=[0.10, 0.25, 0.35, 0.20, 0.10])

    regions = np.random.choice(
        ['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄'],
        n_sample, p=[0.05, 0.07, 0.35, 0.15, 0.18, 0.07, 0.04, 0.09]
    )
    prefs = np.random.choice(
        ['東京', '神奈川', '大阪', '愛知', '埼玉', '千葉', '福岡', '北海道',
         '兵庫', '静岡', '茨城', '広島', '京都', '宮城', '新潟', '長野',
         '栃木', '岐阜', '群馬', '岡山', '三重', '熊本', '鹿児島', '山口',
         '愛媛', '長崎', '奈良', '青森', '岩手', '大分', '石川', '山形',
         '富山', '秋田', '香川', '和歌山', '佐賀', '福井', '徳島', '高知',
         '島根', '宮崎', '鳥取', '沖縄', '滋賀', '山梨', '福島'],
        n_sample
    )
    sample_sub = pd.DataFrame({
        'customer_id':  [f'S{i:05d}' for i in range(1, n_sample+1)],
        'start_date':   [d.strftime('%Y-%m-%d') for d in start_dates],
        'end_date':     end_dates_sub,
        'revenue_total': revenues_sub,
        'plan':         plans,
        'channel':      channels,
        'age_group':    ages,
        'region':       regions,
        'prefecture':   prefs,
    })

    # ── 都度課金用サンプル ──
    # end_dateは基本空欄、last_purchase_dateで休眠判定
    # revenue_total: 実際の累計購買額
    unit_price = np.random.choice([3000, 5000, 10000], n_sample, p=[0.5, 0.35, 0.15])
    last_purchase_dates = []
    revenues_spot = []
    for i in range(n_sample):
        sd = start_dates[i]
        if churned[i]:
            lp = sd + pd.Timedelta(days=int(survival_days[i]))
            lp = min(lp, today_ts - pd.Timedelta(days=1))
        else:
            # アクティブ：直近180日以内にランダムに購買
            if np.random.random() < 0.2:
                lp = today_ts - pd.Timedelta(days=np.random.randint(200, 600))
            else:
                lp = today_ts - pd.Timedelta(days=np.random.randint(1, 180))
        last_purchase_dates.append(lp.strftime('%Y-%m-%d'))
        purchases = max(1, round((lp - sd).days / 60))
        revenues_spot.append(unit_price[i] * purchases)

    spot_plans    = np.random.choice(['ベーシック', 'スタンダード', 'プレミアム'], n_sample, p=[0.5, 0.35, 0.15])
    spot_channels = np.random.choice(['SNS', '検索広告', '紹介', 'オーガニック'], n_sample, p=[0.3, 0.3, 0.2, 0.2])
    spot_ages     = np.random.choice(['10代以下', '20代', '30代', '40代', '50代以上'], n_sample, p=[0.10, 0.25, 0.35, 0.20, 0.10])
    spot_plan_mult = np.where(spot_plans == 'プレミアム', 2.5, np.where(spot_plans == 'スタンダード', 1.5, 1.0))
    spot_age_mult  = np.where(spot_ages == '50代以上', 1.6, np.where(spot_ages == '40代', 1.3,
                     np.where(spot_ages == '30代', 1.1, np.where(spot_ages == '20代', 0.9, 0.7))))
    revenues_spot_adj = np.array(revenues_spot) * spot_plan_mult * spot_age_mult
    sample_spot = pd.DataFrame({
        'customer_id':        [f'D{i:05d}' for i in range(1, n_sample+1)],
        'start_date':         [d.strftime('%Y-%m-%d') for d in start_dates],
        'end_date':           '',
        'last_purchase_date': last_purchase_dates,
        'revenue_total':      revenues_spot_adj.astype(int),
        'plan':               spot_plans,
        'channel':            spot_channels,
        'age_group':          spot_ages,
        'region':             np.random.choice(['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄'], n_sample, p=[0.05, 0.07, 0.35, 0.15, 0.18, 0.07, 0.04, 0.09]),
        'prefecture':         np.random.choice(['東京', '神奈川', '大阪', '愛知', '埼玉', '千葉', '福岡', '北海道', '兵庫', '静岡', '茨城', '広島', '京都', '宮城', '新潟', '長野', '栃木', '岐阜', '群馬', '岡山', '三重', '熊本', '鹿児島', '山口', '愛媛', '長崎', '奈良', '青森', '岩手', '大分', '石川', '山形', '富山', '秋田', '香川', '和歌山', '佐賀', '福井', '徳島', '高知', '島根', '宮崎', '鳥取', '沖縄', '滋賀', '山梨', '福島'], n_sample),
    })

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 サブスク用サンプル（1万件）",
            sample_sub.to_csv(index=False).encode('utf-8-sig'),
            "sample_subscription.csv", "text/csv",
            use_container_width=True
        )
        st.caption("`end_date`: 解約日（継続中は空欄）")
    with col_dl2:
        st.download_button(
            "📥 都度課金用サンプル（1万件）",
            sample_spot.to_csv(index=False).encode('utf-8-sig'),
            "sample_spot_purchase.csv", "text/csv",
            use_container_width=True
        )
        st.caption("`last_purchase_date`: 最終購買日")

    uploaded = st.file_uploader("CSVをアップロード", type=['csv'])

    st.markdown("---")
    st.markdown("### 🏷️ ビジネスタイプ")
    business_type = st.radio(
        "ビジネスタイプを選択してください",
        [
            "サブスク・継続課金型",
            "都度課金型",
        ],
        index=0,
    )

    if business_type == "サブスク・継続課金型":
        st.caption(
            "解約日（end_date）をベースに離脱を判定します。"
            "end_dateが空欄の顧客は継続中として扱われます。"
        )
        dormancy_days = None  # 休眠判定なし
        st.markdown("**契約期間**")
        billing_cycle = st.radio(
            "契約期間",
            [
                "カレンダーベース（月またぎ）← 月額サブスク推奨",
                "30日固定 ← 30日プラン",
                "365日固定 ← 年額サブスク",
                "カスタム入力（日数固定）",
            ],
            index=0,
        )
        if billing_cycle == "カスタム入力（日数固定）":
            custom_cycle_days = st.number_input("契約日数", min_value=1, max_value=365, value=30)
        else:
            custom_cycle_days = None
        st.caption(
            "**カレンダーベース**：実際の月の日数（28〜31日）で計算。日本の月額サブスクの大半はこちら。"
            "　**30日固定**：1ヶ月を常に30日として計算。"
            "　**年額**：1年を365日として計算。"
        )

    else:  # 都度課金型
        st.caption(
            "最終購買日（last_purchase_date）をベースに休眠判定します。"
            "CSVに `last_purchase_date` 列が必要です。"
            "ARPU = 累計売上 ÷ 継続日数で計算します。"
        )
        billing_cycle = "日次（都度課金）"
        custom_cycle_days = None
        st.markdown("**休眠判定期間**")
        dormancy_option = st.radio(
            "休眠判定期間",
            [
                "180日",
                "365日",
                "730日",
                "カスタム入力",
            ],
            index=0,
        )
        st.caption(
            "あなたのビジネスに合った休眠顧客の認定期間を設定してください。"
            "判断が難しい場合は、自社データで最終購買から再購買が発生しなくなる日数を確認することをお勧めします。"
        )
        if dormancy_option == "カスタム入力":
            dormancy_days = st.number_input("休眠判定日数", min_value=30, max_value=3650, value=180)
        else:
            dormancy_days = int(dormancy_option.split("日")[0])

    horizon_days = 730  # 内部計算用デフォルト

    st.markdown("---")
    st.markdown("### 📊 粗利率（GPM）")
    gpm = st.slider("Gross Profit Margin (%)", 0, 100, 54, 1) / 100
    st.caption(f"LTV∞の表示は売上ベース。CAC上限の算出には粗利ベース（売上×{gpm:.0%}）を使用します")

    st.markdown("---")
    st.markdown("### 💰 CAC上限の算出")
    cac_mode = st.radio("算出方法", ['LTV : CAC = N : 1', '回収期間（月）'])
    if cac_mode == 'LTV : CAC = N : 1':
        cac_n = st.slider("N（LTV:CAC = N:1）", 1.0, 10.0, 3.0, 0.5)
        cac_label = f"LTV:CAC = {cac_n}:1"
        st.caption(f"例：LTV:CAC = 3:1 の場合、CAC上限 = LTV ÷ 3")
    else:
        cac_n = st.slider("回収期間（月）", 1, 36, 12)
        cac_label = f"{cac_n}ヶ月回収"

    st.markdown("---")
    st.markdown("### 🔬 セグメント分析設定")
    st.caption(
        "**セグメント列とは？**\n\n"
        "顧客を分類するための列です。CSVに追加しておくことで、"
        "セグメント別のLTV∞を自動比較し、最も収益性の高い"
        "優先獲得セグメントを特定します。\n\n"
        "**指定方法：** CSVの列名をカンマ区切りで入力\n"
        "例：`plan, channel, age_group`\n\n"
        "**1列あたりのユニーク値：** 最大50種類（都道府県47個も対応）\n"
        "**指定できる列数：** 最大5列（5列超は分割して入力）\n\n"
        "**代表的なセグメント軸：**\n"
        "・プラン別（月額・年額・プランA/B）\n"
        "・獲得チャネル別（SNS・検索・紹介・自然流入）\n"
        "・デモグラフィック（年齢層・性別・地域）\n"
        "・行動（初回購買金額・利用頻度・登録経路）"
    )
    segment_cols_input = st.text_input(
        "セグメント列名（カンマ区切りで複数指定可）",
        placeholder="例：plan, channel, age_group（最大5列）",
    )
    st.markdown("---")
    st.markdown("### 🔢 ブラウザ表示件数")
    seg_display_limit = st.slider(
        "詳細表示（暫定LTV・生存曲線）の上位N件",
        min_value=1, max_value=20, value=5,
    )
    st.caption(
        "セグメント（例：都道府県）の項目数（例：47）が多いほどブラウザの描画に時間がかかります。"
        "表示する上位N項目を絞ることで速度が大幅に改善されます。\n"
        "**パワポ・PDFは設定に関わらず全項目出力されます。**\n"
        "まず上位5項目で傾向を確認し、必要に応じて増やすことをお勧めします。"
    )

    st.markdown("---")
    st.markdown("### 💰 CAC（顧客獲得コスト）")
    st.caption(
        "平均CACを入力すると、セグメント別の獲得効率スコア（LTV:CAC比率）も算出されます。"
        "不明な場合は空欄のままでOKです。LTV∞ベースの優先スコアのみ表示されます。"
    )
    cac_input = st.number_input(
        "平均CAC（任意・円）",
        min_value=0, value=0, step=1000,
    )
    cac_known = cac_input > 0

    st.markdown("---")
    st.markdown("### 🏢 レポート情報")
    client_name  = st.text_input("クライアント名", "")
    analyst_name = st.text_input("分析者名", "")

# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 8px 0 24px 0;'>
  <span style='font-family:Syne,sans-serif; font-size:2rem; color:#56b4d3;'>LTV Analyzer PRO</span>
  <span style='font-size:0.8rem; color:#444; margin-left:12px;'>Segment × LTV Maximization</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# No file → instructions
# ══════════════════════════════════════════════════════════════

if uploaded is None:
    st.info("← サイドバーからCSVをアップロードしてください。サンプルCSVでまずお試しいただけます。")

    st.markdown("<div class='section-title'>入力CSVの形式</div>", unsafe_allow_html=True)
    st.markdown("""
| 列名 | 内容 | 形式 | 例 |
|------|------|------|----|
| `customer_id` | 顧客ID | 任意の文字列 | C0001 |
| `start_date` | 契約開始日 | YYYY-MM-DD | 2023-01-01 |
| `end_date` | 解約日（サブスク向け・継続中は**空欄**） | YYYY-MM-DD | 2024-03-15 |
| `last_purchase_date` | 最終購買日（都度課金向け・任意） | YYYY-MM-DD | 2024-06-01 |
| `revenue` | **累計売上**（円） | 数値 | 48000 |
| `セグメント列`（任意の列名） | **PRO機能**：プラン・チャネル・年齢層など | 文字列 | 月額300 |

> **PRO版では必ずセグメント列を追加してください。**複数列追加可能です。\n
> 列名は完全一致でなくてもOKです。`start`・`end`・`last`・`revenue`を含む列名は自動認識します。\n
> ARPU_daily は「累計売上 ÷ 顧客継続日数」で自動計算されます。\n
> セグメント列は1列あたり最大50種類のユニーク値まで対応しています（都道府県47個も対応）。
    """)

    st.markdown("<div class='section-title'>分析の流れ</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    steps = [
        ("① KM法", "実測データから生存曲線を作成"),
        ("② Weibull", "連続曲線にフィッティング"),
        ("③ LTV∞", "生存積分 × ARPU で算出"),
        ("④ CAC上限", "LTV比率で逆算"),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-value' style='font-size:1.1rem'>{title}</div><div class='metric-label' style='font-size:0.75rem; color:#666; margin-top:8px;'>{desc}</div></div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
# Load & validate data
# ══════════════════════════════════════════════════════════════

try:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    col_map = {}
    for c in df_raw.columns:
        if any(x in c for x in ['id','customer','顧客']):      col_map.setdefault('customer_id', c)
        if any(x in c for x in ['start','開始','契約']):       col_map.setdefault('start_date', c)
        if any(x in c for x in ['end','解約','終了','cancel']): col_map.setdefault('end_date', c)
        if any(x in c for x in ['revenue','売上','arpu','rev','price','amount']):
            col_map.setdefault('revenue', c)

    missing = [k for k in ['start_date','end_date','revenue'] if k not in col_map]
    if missing:
        st.error(f"❌ 列が見つかりません: {missing}\n\n列名に `start`・`end`・`revenue` を含む列が必要です。サイドバーからサンプルCSVをダウンロードして形式を確認してください。")
        st.stop()

    # last_purchase_date 列の自動認識
    for c in df_raw.columns:
        if any(x in c for x in ['last','最終','purchase','購買','購入']):
            col_map.setdefault('last_purchase_date', c)

    df = df_raw.rename(columns={v: k for k, v in col_map.items()})
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date']   = pd.to_datetime(df['end_date'], errors='coerce')
    if 'last_purchase_date' in df.columns:
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    else:
        df['last_purchase_date'] = pd.NaT

    bad_dates = df['start_date'].isna().sum()
    if bad_dates > 0:
        st.warning(f"⚠️ {bad_dates}行で `start_date` が読み取れませんでした。該当行は除外します。")
    df = df.dropna(subset=['start_date'])

    today = pd.Timestamp.today()
    n_input = len(df)

    # 休眠判定：end_date がなく last_purchase_date が休眠期間を超えている場合は離脱とみなす
    n_dormant = 0
    if dormancy_days is not None and df['last_purchase_date'].notna().any():
        dormant_mask = (
            df['end_date'].isna() &
            df['last_purchase_date'].notna() &
            ((today - df['last_purchase_date']).dt.days > dormancy_days)
        )
        n_dormant = dormant_mask.sum()
        df.loc[dormant_mask, 'end_date'] = df.loc[dormant_mask, 'last_purchase_date']

    # ── duration・event の確定 ──
    # サブスク（解約日のみ）：end_date あり→解約、なし→継続（今日まで）
    # 都度課金（休眠判定あり）：end_date あり→解約、
    #   end_date なし + last_purchase_date から dormancy_days 超→実質離脱（last_purchase_dateまで）
    #   end_date なし + dormancy_days 未満→継続（今日まで）

    # 期間の終端日を決定
    def get_end(row):
        if pd.notna(row['end_date']):
            return row['end_date'], 1  # 解約済み
        if dormancy_days is not None and pd.notna(row['last_purchase_date']):
            days_since = (today - row['last_purchase_date']).days
            if days_since > dormancy_days:
                return row['last_purchase_date'], 1  # 実質離脱
        return today, 0  # 継続中

    result = df.apply(get_end, axis=1, result_type='expand')
    df['end_resolved'] = result[0]
    df['event']        = result[1]
    df['duration']     = (df['end_resolved'] - df['start_date']).dt.days

    # duration=0 を1日に自動補正
    n_corrected = (df['duration'] == 0).sum()
    df['duration'] = df['duration'].clip(lower=1)

    # duration<0（開始日が未来）は除外
    n_excluded = (df['duration'] < 0).sum()
    df = df[df['duration'] > 0]

    # ── ARPU_daily の計算 ──
    df['revenue_total'] = pd.to_numeric(df['revenue'], errors='coerce')

    def calc_arpu_daily(row):
        rev      = row['revenue_total']
        dur      = row['duration']
        start    = row['start_date']
        end_r    = row['end_resolved']
        if pd.isna(rev) or dur <= 0:
            return np.nan

        if billing_cycle == "日次（都度課金）":
            # 都度課金：累計売上 ÷ 継続日数
            return rev / dur

        elif billing_cycle == "カレンダーベース（月またぎ）← 月額サブスク推奨":
            # 契約開始日の「日」を基準に何ヶ月分更新されたかを数える
            s, e = start, end_r
            renewals = 0
            cur = s
            while True:
                # 翌月の同日を計算
                month = cur.month + 1 if cur.month < 12 else 1
                year  = cur.year if cur.month < 12 else cur.year + 1
                max_day = calendar.monthrange(year, month)[1]
                day   = min(cur.day, max_day)
                nxt   = pd.Timestamp(year, month, day)
                if nxt > e:
                    break
                renewals += 1
                cur = nxt
            renewals = max(renewals, 1)
            # 各月の実日数の平均で日次換算
            total_days = (pd.Timestamp(e.year, e.month,
                          calendar.monthrange(e.year, e.month)[1]) -
                          pd.Timestamp(s.year, s.month, 1)).days / renewals
            avg_days = max(total_days, 1)
            return (rev / renewals) / avg_days

        elif billing_cycle == "30日固定 ← 30日プラン":
            import math
            renewals = max(math.ceil(dur / 30), 1)
            return (rev / renewals) / 30

        elif billing_cycle == "365日固定 ← 年額サブスク":
            import math
            renewals = max(math.ceil(dur / 365), 1)
            return (rev / renewals) / 365

        else:  # カスタム入力
            import math
            cycle = custom_cycle_days or 30
            renewals = max(math.ceil(dur / cycle), 1)
            return (rev / renewals) / cycle

    df['arpu_daily'] = df.apply(calc_arpu_daily, axis=1)
    df['gp_daily']   = df['arpu_daily'] * gpm
    df = df.dropna(subset=['arpu_daily'])
    df = df[df['arpu_daily'] > 0]
    arpu_daily = df['arpu_daily'].mean()
    gp_daily   = df['gp_daily'].mean()

    # ── 通知メッセージ ──
    if n_dormant > 0:
        st.info(f"💤 {n_dormant:,}件を休眠顧客（最終購買から{dormancy_days}日超）として実質離脱に変換しました。")
    if n_corrected > 0:
        st.info(f"ℹ️ {n_corrected}件：契約開始日と終端日が同日のため1日に補正しました。")
    if n_excluded > 0:
        st.warning(f"⚠️ {n_excluded}件：契約開始日が未来のため除外しました。")
    if n_dormant == 0 and n_corrected == 0 and n_excluded == 0:
        st.success(f"✅ 全{n_input:,}件のデータを正常に読み込みました。")

    if len(df) < 10:
        st.error("❌ 有効なデータが10件未満です。分析には最低10件の顧客データが必要です。")
        st.stop()

except Exception as e:
    st.error(f"❌ データ読み込みエラー: {e}\n\nCSVの形式を確認してください。サンプルCSVをダウンロードして参照してください。")
    st.stop()

# ══════════════════════════════════════════════════════════════
# Run analysis
# ══════════════════════════════════════════════════════════════

km_df = compute_km(df)
k, lam, r2, fit_df = fit_weibull(km_df)

if k is None:
    st.error("❌ Weibullフィッティングに失敗しました。解約済み顧客が少なすぎる可能性があります（最低10件の解約データが必要）。")
    st.stop()

ltv_rev, surv_int = ltv_inf(k, lam, arpu_daily)   # 売上ベース
ltv_val, _        = ltv_inf(k, lam, gp_daily)      # 粗利ベース（CACに使う）
cac_upper = ltv_val / cac_n

# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>分析結果サマリー</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)

k_desc  = "k<1: 初期離脱が多い" if k < 1 else "k>1: 時間とともに離脱増"
metrics = [
    (f"¥{ltv_rev:,.0f}", "LTV∞",       "売上ベース"),
    (f"¥{cac_upper:,.0f}", f"CAC上限",  f"{cac_label}（粗利ベース）"),
    (f"{k:.3f}",           "Weibull k", f"形状パラメータ / {k_desc}"),
    (f"{lam:.1f}日",       "Weibull λ", "尺度 / 典型的な継続期間の目安"),
    (f"{r2:.3f}",          "R²",        "0.9以上が理想 / 1.0が最高精度"),
]
for col, (val, title, desc) in zip([m1,m2,m3,m4,m5], metrics):
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label' style='font-size:0.72rem; color:#56b4d3; margin-top:6px; letter-spacing:0.05em;'>{title}</div>"
            f"<div class='metric-label' style='font-size:0.65rem; color:#444; margin-top:3px; letter-spacing:0.03em; line-height:1.4;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# R² warning
if r2 < 0.85:
    st.warning(f"⚠️ R²={r2:.3f} — フィット精度がやや低めです。データ点数を増やすか、観測期間を見直してください。")

# ══════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>分析グラフ</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

t_smooth = np.linspace(1, km_df['t'].max() * 1.3, 600)
S_wei    = weibull_s(t_smooth, k, lam)

with c1:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.step(km_df['t'], km_df['S'], where='post', color=ACCENT, lw=1.8, label='KM Curve (Observed)')
    ax.plot(t_smooth, S_wei, color=ACCENT2, lw=1.5, ls='--', label='Weibull Fit')
    ax.fill_between(t_smooth, S_wei, alpha=0.06, color=ACCENT2)
    ax.set(xlabel='Days', ylabel='Survival Rate S(t)', ylim=(0,1.05))
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(True, alpha=0.25)
    ax.set_title('Survival Curve', color='#ccc', fontsize=10, pad=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with c1:
    st.caption("📌 生存曲線（Survival Curve）：実測のKM曲線（実線）にWeibullモデルをフィット（破線）。右に伸びるほど顧客が長く継続している。")

with c2:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    if fit_df is not None:
        ax.scatter(fit_df['ln_t'], fit_df['ln_neg_ln_S'], color=ACCENT, s=18, alpha=0.7, label='Observed')
        x_r = np.linspace(fit_df['ln_t'].min(), fit_df['ln_t'].max(), 200)
        b   = np.log(-np.log(weibull_s(1, k, lam) + 1e-15))
        ax.plot(x_r, k*x_r + b, color=ACCENT2, lw=1.5, ls='--', label=f'Regression Line (R²={r2:.3f})')
        ax.annotate(f'y = {k:.4f}x + {b:.4f}',
                    xy=(0.05,0.93), xycoords='axes fraction', color='#777', fontsize=8)
    ax.set(xlabel='ln(t)', ylabel='ln(−ln(S(t)))')
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(True, alpha=0.25)
    ax.set_title('Weibull Linearization Plot', color='#ccc', fontsize=10, pad=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with c2:
    st.caption("📌 Weibull直線化プロット：生存率を対数変換して直線化したもの。R²が1.0に近いほどWeibullモデルのフィット精度が高い。")

# Save chart images for export
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.step(km_df['t'], km_df['S'], where='post', color=ACCENT, lw=2, label='KM Curve')
ax1.plot(t_smooth, S_wei, color=ACCENT2, lw=1.8, ls='--', label='Weibull Fit')
ax1.fill_between(t_smooth, S_wei, alpha=0.07, color=ACCENT2)
ax1.set(xlabel='Days', ylabel='Survival Rate S(t)', ylim=(0,1.05))
ax1.legend(fontsize=9, framealpha=0.15)
ax1.grid(True, alpha=0.25)
ax1.set_title('Survival Curve (KM × Weibull)', color='#ccc', fontsize=11, pad=10)
fig1.tight_layout()
buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight'); buf1.seek(0)
plt.close()

fig2, ax2 = plt.subplots(figsize=(7, 4))
if fit_df is not None:
    ax2.scatter(fit_df['ln_t'], fit_df['ln_neg_ln_S'], color=ACCENT, s=22, alpha=0.75)
    x_r = np.linspace(fit_df['ln_t'].min(), fit_df['ln_t'].max(), 200)
    b   = np.log(-np.log(weibull_s(1, k, lam) + 1e-15))
    ax2.plot(x_r, k*x_r+b, color=ACCENT2, lw=1.8, ls='--', label=f'R²={r2:.3f}')
    ax2.annotate(f'y = {k:.4f}x + {b:.4f}', xy=(0.05,0.93), xycoords='axes fraction', color='#777', fontsize=9)
ax2.set(xlabel='ln(t)', ylabel='ln(−ln(S(t)))')
ax2.legend(fontsize=9, framealpha=0.15)
ax2.grid(True, alpha=0.25)
ax2.set_title('Weibull Linearization Plot', color='#ccc', fontsize=11, pad=10)
fig2.tight_layout()
buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight'); buf2.seek(0)
plt.close()

# ══════════════════════════════════════════════════════════════
# Horizon table
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>暫定LTV（観測期間別）</div>", unsafe_allow_html=True)

horizons = [30, 90, 180, 365, 730, 1095, 1825]  # 30日・90日・180日・1年・2年・3年・5年
rows = []
for h in horizons:
    lh = ltv_horizon(k, lam, arpu_daily, h)
    rows.append({
        'ホライズン': f'{h//365}年' if h >= 365 and h % 365 == 0 else f'{h}日',
        '暫定LTV（売上ベース）': f'¥{lh:,.0f}',
        'LTV∞比（売上）': f'{lh/ltv_rev*100:.1f}%',
        f'CAC上限（粗利ベース）({cac_label})': f'¥{lh/cac_n:,.0f}',
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# 解釈ガイドを自動生成
ltv_1y  = ltv_horizon(k, lam, arpu_daily, 365)
ltv_2y  = ltv_horizon(k, lam, arpu_daily, 730)
ltv_3y  = ltv_horizon(k, lam, arpu_daily, 1095)
pct_1y  = ltv_1y  / ltv_rev * 100
pct_2y  = ltv_2y  / ltv_rev * 100
pct_3y  = ltv_3y  / ltv_rev * 100

# CAC回収期間を逆算（何日でCAC上限を回収できるか）
try:
    cac_recover_days = brentq(
        lambda h: ltv_horizon(k, lam, arpu_daily, h) - cac_upper,
        1, 9999
    )
    cac_recover_str = (
        f"{cac_recover_days/365:.1f}年（{int(cac_recover_days)}日）"
        if cac_recover_days >= 365
        else f"{int(cac_recover_days)}日"
    )
except Exception:
    cac_recover_str = "算出不可（LTV∞がCAC上限を下回っています）"

# λの解釈
if lam < 180:
    lam_desc = f"λ={lam:.0f}日は非常に短く、顧客の大半が半年以内に離脱するビジネスです。"
elif lam < 365:
    lam_desc = f"λ={lam:.0f}日は比較的短く、多くの顧客が1年以内に離脱するビジネスです。"
elif lam < 730:
    lam_desc = f"λ={lam:.0f}日（約{lam/365:.1f}年）は中程度の継続期間で、1〜2年継続する顧客が多いビジネスです。"
else:
    lam_desc = f"λ={lam:.0f}日（約{lam/365:.1f}年）は長く、顧客が数年にわたって継続するビジネスです。"

# k の解釈
if k < 0.8:
    k_desc = f"k={k:.3f}は1より大きく小さいため、契約直後の離脱が特に多いパターンです。初期のオンボーディング改善が最優先です。"
elif k < 1.0:
    k_desc = f"k={k:.3f}は1に近いため、離脱率がほぼ一定（指数分布に近い）パターンです。"
else:
    k_desc = f"k={k:.3f}は1より大きいため、時間とともに離脱率が上がるパターンです。長期継続顧客のフォローが重要です。"

insight_html = f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-radius:10px; padding:18px 20px; margin-top:12px; line-height:1.9; font-size:0.85rem; color:#ccc;'>
  <div style='color:#56b4d3; font-size:0.82rem; font-weight:500; margin-bottom:10px; letter-spacing:0.05em;'>💡 このテーブルの読み方</div>
  <div>・{lam_desc}</div>
  <div>・{k_desc}</div>
  <div>・LTV∞（¥{ltv_rev:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。</div>
  <div style='margin-top:8px;'>
    <span style='color:#56b4d3;'>1年時点</span>でLTV∞の<b style='color:#a8dadc;'>{pct_1y:.1f}%</b>（¥{ltv_1y:,.0f}）、
    <span style='color:#56b4d3;'>2年時点</span>で<b style='color:#a8dadc;'>{pct_2y:.1f}%</b>（¥{ltv_2y:,.0f}）、
    <span style='color:#56b4d3;'>3年時点</span>で<b style='color:#a8dadc;'>{pct_3y:.1f}%</b>（¥{ltv_3y:,.0f}）を回収できます。
  </div>
  <div style='margin-top:8px;'>・CAC上限（¥{cac_upper:,.0f}）を回収できるのは契約から約 <b style='color:#a8dadc;'>{cac_recover_str}</b> 後です。</div>
</div>
"""
st.markdown(insight_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# AI Prompt Generator
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>🤖 AIに質問するプロンプト</div>", unsafe_allow_html=True)
st.markdown("<div class='help-box'>この結果の読み方や戦略への活用方法がわからない場合は、以下のプロンプトをClaude・ChatGPT・Geminiにコピペしてください。</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 分析結果の解釈", "📈 マーケティング意思決定", "⚠️ 分析の限界と改善"])

dormancy_label = "なし（解約日ベース）" if dormancy_days is None else f"{dormancy_days}日"
churned_count  = int(df['event'].sum())
active_count   = int((df['event']==0).sum())
churn_rate     = churned_count / len(df) * 100
k_pattern      = "初期集中型（契約直後の離脱が多い）" if k < 1 else "逓増型（時間とともに離脱が増える）"

prompt_base = f"""私はLTV分析ツール（Kaplan-Meier法 × Weibullモデル）を使い、以下の結果を得ました。

【分析結果】
・ビジネスタイプ: {business_type} / 休眠判定: {dormancy_label}
・顧客数: {len(df):,}件（解約済み: {churned_count:,}件 / 継続中: {active_count:,}件 / 解約率: {churn_rate:.1f}%）
・平均日次ARPU（売上）: ¥{arpu_daily:,.2f} / 平均日次GP（粗利）: ¥{gp_daily:,.2f} / GPM: {gpm:.1%}
・LTV∞（売上ベース）: ¥{ltv_rev:,.0f} / LTV∞（粗利ベース）: ¥{ltv_val:,.0f}
・CAC上限（{cac_label}）: ¥{cac_upper:,.0f}
・Weibull k（形状）: {k:.4f} → {k_pattern}
・Weibull λ（尺度）: {lam:.1f}日 / R²（フィット精度）: {r2:.4f}
・分析手法: Kaplan-Meier法 + Weibullモデルによる生存分析"""

copy_html = """<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#56b4d3; margin-top:4px;'>
📋 上のテキストボックス右上の <b>コピーアイコン</b> をクリック → Claude / ChatGPT / Gemini に貼り付けてください
</div>"""

with tab1:
    p1 = prompt_base + f"""

【質問】―― 以下の数値を具体的に使って答えてください ――
1. k={k:.4f}・λ={lam:.1f}日という値は、このビジネスの顧客離脱パターンをどう示していますか？k<1・k>1それぞれの意味と今回の値の解釈を教えてください。
2. 解約率{churn_rate:.1f}%（解約{churned_count:,}件・継続{active_count:,}件）という比率から、このビジネスの健全性をどう評価しますか？
3. LTV∞（売上）¥{ltv_rev:,.0f}に対してCAC上限¥{cac_upper:,.0f}という比率は適切ですか？業界水準と比較して教えてください。
4. R²={r2:.4f}のフィット精度はWeibull分析として許容範囲ですか？この値が示す信頼性の限界を教えてください。"""
    st.code(p1, language=None)
    st.markdown(copy_html, unsafe_allow_html=True)

with tab2:
    p2 = prompt_base + f"""

【質問】―― 上記の数値から直接導ける意思決定を具体的に答えてください ――
1. CAC上限¥{cac_upper:,.0f}（粗利ベースLTV÷{cac_n}）をもとに、CPAとROASの目標値をどう設定すべきですか？計算式も示してください。
2. λ={lam:.1f}日（典型的な継続期間）を踏まえると、契約後何日目にリテンション施策を打つのが最も効果的ですか？推奨タイミングと施策内容を教えてください。
3. ARPU_daily¥{arpu_daily:,.2f}・LTV∞¥{ltv_rev:,.0f}の水準で費用対効果的に合いやすい広告チャネルはどれですか？CPAとの関係で説明してください。
4. {k_pattern}に対して最も効果的なリテンション施策のタイミングと種類を教えてください。"""
    st.code(p2, language=None)
    st.markdown(copy_html, unsafe_allow_html=True)

with tab3:
    p3 = prompt_base + f"""

【質問】―― モデルの信頼性・限界・改善策を具体的に答えてください ――
1. {len(df):,}件・解約{churned_count:,}件のサンプルサイズでWeibull推定の信頼区間はどの程度ですか？十分なサンプル数の目安を教えてください。
2. k={k:.4f}はWeibull分布の単調ハザード率の仮定を満たしていますか？仮定が崩れる典型例とチェック方法を教えてください。
3. R²={r2:.4f}を改善するにはどうすればよいですか？データ量・期間・外れ値処理の観点から具体的に教えてください。
{"4. 解約日ベースで分析していますが、解約データの欠損や遅延がある場合にLTV推定にどんな影響が出ますか？" if dormancy_days is None else f"4. 休眠判定{dormancy_label}の設定はこのビジネスに適切ですか？最適な判定日数を決める感度分析の手順を教えてください。"}
"""
    st.code(p3, language=None)
    st.markdown(copy_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Export buttons
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>📤 エクスポート</div>", unsafe_allow_html=True)

exp1, exp2, exp3 = st.columns(3)

# ── Excel export ─────────────────────────────────────────────
with exp1:
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.chart import LineChart, Reference, Series

        wb = openpyxl.Workbook()

        # Summary sheet
        ws = wb.active
        ws.title = 'Summary'
        hdr_fill = PatternFill('solid', start_color='1a1a1a', end_color='1a1a1a')
        hdr_font = Font(name='Calibri', bold=True, color='56B4D3', size=11)
        val_font = Font(name='Calibri', size=11, color='111111')

        ws['A1'] = 'LTV分析 サマリー'
        ws['A1'].font = Font(name='Calibri', bold=True, size=14, color='56B4D3')
        if client_name:
            ws['A2'] = f'クライアント: {client_name}'
        if analyst_name:
            ws['A3'] = f'分析者: {analyst_name}'

        summary_data = [
            ('', ''),
            ('【分析結果】', ''),
            ('顧客数（総数）', len(df)),
            ('うち解約済み', int(df['event'].sum())),
            ('平均日次ARPU（売上ベース・¥）', round(arpu_daily, 2)),
            ('粗利率（GPM）', f'{gpm:.1%}'),
            ('平均日次GP（粗利ベース・¥）', round(gp_daily, 2)),
            ('LTV∞ 売上ベース（¥）', round(ltv_rev, 0)),
            ('LTV∞ 粗利ベース（CAC算出用）（¥）', round(ltv_val, 0)),
            (f'CAC上限（{cac_label}）（¥）', round(cac_upper, 0)),
            ('', ''),
            ('【Weibullパラメータ】', ''),
            ('k（形状パラメータ）', round(k, 4)),
            ('λ（尺度パラメータ・日）', round(lam, 2)),
            ('R²', round(r2, 4)),
        ]
        for i, (label, val) in enumerate(summary_data, start=5):
            ws.cell(i, 1, label).font = Font(name='Calibri', bold=('【' in str(label)), color='333333', size=10)
            ws.cell(i, 2, val).font   = Font(name='Calibri', size=10, color='111111')
        ws.column_dimensions['A'].width = 32
        ws.column_dimensions['B'].width = 20

        # KM sheet
        ws2 = wb.create_sheet('KM_生存曲線')
        ws2.append(['t (days)', 'S(t) KM Observed', 'S(t) Weibull Fit'])
        for _, row in km_df.iterrows():
            t = row['t']
            ws2.append([int(t), round(row['S'], 6), round(float(weibull_s(t, k, lam)), 6)])

        # Horizon sheet
        ws3 = wb.create_sheet('暫定LTV')
        ws3.append(['ホライズン（日）', '暫定LTV（¥）', 'LTV∞比（%）', f'CAC上限（¥）'])
        for h in horizons:
            lh = ltv_horizon(k, lam, arpu_daily, h)
            ws3.append([h, round(lh, 0), round(lh/ltv_val*100, 1), round(lh/cac_n, 0)])

        # セグメント別シート追加
        if segment_cols_input.strip():
            seg_cols_xl = [c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
            for sc in seg_cols_xl:
                ws_seg = wb.create_sheet(f'SEG_{sc}'[:31])
                ws_seg.append(['セグメント', '顧客数', 'LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）', '総ポテンシャル', 'k', 'λ（日）', 'R²', '獲得効率'])
                seg_vals = df[sc].dropna().unique()
                seg_rows = []
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = compute_km(df_s)
                        k_s, lam_s, r2_s, _ = fit_weibull(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['arpu_daily'].mean()
                        gp_s   = arpu_s * gpm
                        ltv_r, _ = ltv_inf(k_s, lam_s, arpu_s)
                        ltv_g, _ = ltv_inf(k_s, lam_s, gp_s)
                        eff = round(ltv_g / cac_input, 2) if cac_known else '-'
                        seg_rows.append([sv, len(df_s), round(ltv_r,0), round(ltv_g,0), round(ltv_g/cac_n,0), round(ltv_r*len(df_s),0), round(k_s,4), round(lam_s,1), round(r2_s,4), eff])
                    except Exception:
                        continue
                seg_rows.sort(key=lambda x: x[2], reverse=True)
                for row in seg_rows:
                    ws_seg.append(row)
                ws_seg.column_dimensions['A'].width = 20

        xl_buf = io.BytesIO()
        wb.save(xl_buf)
        xl_buf.seek(0)
        st.download_button("📊 Excelダウンロード", xl_buf,
                           file_name=f"LTV分析_{client_name or 'report'}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.caption(f"Excel出力エラー: {e}")

# ── PowerPoint export ─────────────────────────────────────────
with exp2:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN

        BG    = RGBColor(0x0d, 0x0d, 0x0d)
        GOLD  = RGBColor(0x56, 0xb4, 0xd3)
        TEAL  = RGBColor(0xa8, 0xda, 0xdc)
        WHITE = RGBColor(0xe8, 0xe4, 0xdc)
        GRAY  = RGBColor(0x44, 0x44, 0x44)

        def add_bg(slide, prs):
            bg = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid(); bg.fill.fore_color.rgb = BG
            bg.line.fill.background(); bg.zorder = 0

        def txbox(slide, text, l, t, w, h, size=12, bold=False, color=WHITE, align=PP_ALIGN.LEFT):
            tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
            tf = tb.text_frame; tf.word_wrap = True
            p  = tf.paragraphs[0]; p.alignment = align
            run = p.add_run(); run.text = text
            run.font.size  = Pt(size)
            run.font.bold  = bold
            run.font.color.rgb = color
            return tb

        prs = Presentation()
        prs.slide_width  = Inches(13.33)
        prs.slide_height = Inches(7.5)
        blank = prs.slide_layouts[6]

        # ── Slide 1: Title ──
        s1 = prs.slides.add_slide(blank)
        add_bg(s1, prs)
        # Accent line
        line = s1.shapes.add_shape(1, Inches(0.8), Inches(3.2), Inches(0.04), Inches(1.4))
        line.fill.solid(); line.fill.fore_color.rgb = GOLD; line.line.fill.background()
        txbox(s1, 'LTV Analysis Report', 1.0, 2.8, 8, 0.8, size=36, bold=True, color=WHITE)
        txbox(s1, 'Kaplan–Meier × Weibull Model', 1.0, 3.7, 8, 0.5, size=14, color=GOLD)
        if client_name:
            txbox(s1, client_name, 1.0, 4.4, 8, 0.4, size=13, color=RGBColor(0x88,0x88,0x88))
        from datetime import date
        txbox(s1, date.today().strftime('%Y年%m月%d日'), 1.0, 5.0, 6, 0.4, size=11, color=GRAY)
        if analyst_name:
            txbox(s1, analyst_name, 1.0, 5.4, 6, 0.4, size=11, color=GRAY)

        # ── Slide 2: Metrics ──
        s2 = prs.slides.add_slide(blank)
        add_bg(s2, prs)
        txbox(s2, '分析結果サマリー', 0.5, 0.3, 10, 0.6, size=22, bold=True, color=WHITE)

        cards = [
            ('LTV∞（売上）', f'¥{ltv_rev:,.0f}', 0.5),
            (f'CAC上限\n({cac_label})', f'¥{cac_upper:,.0f}', 3.6),
            ('Weibull k', f'{k:.3f}', 6.7),
            ('R²', f'{r2:.3f}', 9.8),
        ]
        for label, val, x in cards:
            card = s2.shapes.add_shape(1, Inches(x), Inches(1.3), Inches(2.9), Inches(1.8))
            card.fill.solid(); card.fill.fore_color.rgb = RGBColor(0x16,0x16,0x16)
            card.line.color.rgb = GRAY
            txbox(s2, val, x+0.15, 1.45, 2.6, 0.8, size=24, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
            txbox(s2, label, x+0.15, 2.2, 2.6, 0.5, size=10, color=GRAY, align=PP_ALIGN.CENTER)

        # Data stats
        stats_text = (
            f"顧客数: {len(df):,}件  ／  解約済み: {df['event'].sum():,}件  ／  "
            f"継続中: {(df['event']==0).sum():,}件  ／  "
            f"平均日次ARPU: ¥{arpu_daily:,.2f}  ／  GPM: {gpm:.0%}  ／  LTV∞(売上): ¥{ltv_rev:,.0f}  ／  LTV∞(粗利): ¥{ltv_val:,.0f}  ／  λ: {lam:.1f}日"
        )
        txbox(s2, stats_text, 0.5, 3.3, 12.3, 0.5, size=10, color=GRAY)

        # Horizon table header
        txbox(s2, '観測期間別 暫定LTV', 0.5, 3.9, 12, 0.4, size=13, bold=True, color=WHITE)
        cols_h = ['ホライズン', '暫定LTV', 'LTV∞比', 'CAC上限']
        col_x  = [0.5, 3.3, 6.5, 9.3]
        for cx, ch in zip(col_x, cols_h):
            txbox(s2, ch, cx, 4.4, 2.6, 0.35, size=9, bold=True, color=GOLD)
        row_y = 4.8
        for h in horizons[:5]:
            lh = ltv_horizon(k, lam, arpu_daily, h)
            row_vals = [
                f'{h}日' if h<365 else f'{h//365}年',
                f'¥{lh:,.0f}', f'{lh/ltv_val*100:.1f}%', f'¥{lh/cac_n:,.0f}'
            ]
            for cx, rv in zip(col_x, row_vals):
                txbox(s2, rv, cx, row_y, 2.6, 0.32, size=9, color=WHITE)
            row_y += 0.34

        # ── Slide 3: Charts ──
        s3 = prs.slides.add_slide(blank)
        add_bg(s3, prs)
        txbox(s3, 'Survival Curve  /  Weibull Linearization Plot', 0.5, 0.3, 12, 0.6, size=22, bold=True, color=WHITE)
        buf1.seek(0); s3.shapes.add_picture(buf1, Inches(0.4), Inches(1.1), Inches(6.1), Inches(3.8))
        buf2.seek(0); s3.shapes.add_picture(buf2, Inches(6.8), Inches(1.1), Inches(6.1), Inches(3.8))

        # Weibull parameter explanation box
        param_box = s3.shapes.add_shape(1, Inches(0.4), Inches(5.0), Inches(12.5), Inches(2.1))
        param_box.fill.solid(); param_box.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
        param_box.line.color.rgb = GOLD

        txbox(s3, 'Weibullパラメータの読み方', 0.55, 5.05, 12, 0.3, size=9, bold=True, color=GOLD)

        k_text = (
            f"k（形状パラメータ） = {k:.3f}\n"
            f"→ 左グラフ（Survival Curve）の曲線の急峻さを決める値\n"
            f"→ k<1: 初期に離脱が集中  k>1: 時間とともに離脱が増加\n"
            f"→ 今回は k={k:.3f} のため: {'初期に離脱が集中するパターン' if k < 1 else '継続するほど離脱が増えるパターン'}"
        )
        txbox(s3, k_text, 0.55, 5.4, 6.0, 1.5, size=8, color=WHITE)

        lam_text = (
            f"λ（尺度パラメータ） = {lam:.1f}日\n"
            f"→ 右グラフ（Weibull Plot）の直線の切片から算出\n"
            f"→ 顧客の典型的な継続期間の目安\n"
            f"→ R²={r2:.3f}: フィット精度（1.0が理想、0.9以上を推奨）"
        )
        txbox(s3, lam_text, 6.8, 5.4, 6.0, 1.5, size=8, color=WHITE)

        # 共通データ部分
        pdata = (
            f"【分析結果】\n"
            f"顧客数: {len(df):,}件（解約済み: {df['event'].sum():,}件）\n"
            f"平均日次ARPU（売上）: ¥{arpu_daily:,.2f} / GPM: {gpm:.0%}\n"
            f"LTV∞ 売上ベース: ¥{ltv_rev:,.0f} / 粗利ベース: ¥{ltv_val:,.0f}\n"
            f"CAC上限 ({cac_label}): ¥{cac_upper:,.0f}\n"
            f"Weibull k={k:.4f} / λ={lam:.1f}日 / R²={r2:.4f}\n"
            f"分析手法: Kaplan-Meier法 + Weibullモデル"
        )

        # ── Slide 4: AI Prompt 1 - 結果の読み方 ──
        s4 = prs.slides.add_slide(blank)
        add_bg(s4, prs)
        txbox(s4, 'AIプロンプト ①  結果の読み方', 0.5, 0.25, 12, 0.5, size=18, bold=True, color=WHITE)
        txbox(s4, 'Claude / ChatGPT / Gemini にコピペしてください', 0.5, 0.8, 12, 0.3, size=9, color=GRAY)
        p1_text = (
            f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata}\n\n【質問】\n"
            f"1. Weibullのkとλの値は何を意味していますか？顧客離脱パターンはどう解釈すればよいですか？\n"
            f"2. LTV∞（売上ベース¥{ltv_rev:,.0f}）の値は適切な水準ですか？\n"
            f"3. R²={r2:.4f}からフィット精度はどう評価できますか？\n"
            f"4. この結果で特に注意すべき点があれば教えてください。"
        )
        pb1 = s4.shapes.add_shape(1, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8))
        pb1.fill.solid(); pb1.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
        pb1.line.color.rgb = GOLD
        txbox(s4, p1_text, 0.65, 1.35, 12.0, 5.6, size=9, color=WHITE)

        # ── Slide 5: AI Prompt 2 - マーケ戦略 ──
        s5 = prs.slides.add_slide(blank)
        add_bg(s5, prs)
        txbox(s5, 'AIプロンプト ②  マーケ戦略への活用', 0.5, 0.25, 12, 0.5, size=18, bold=True, color=WHITE)
        txbox(s5, 'Claude / ChatGPT / Gemini にコピペしてください', 0.5, 0.8, 12, 0.3, size=9, color=GRAY)
        p2_text = (
            f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata}\n\n【質問】\n"
            f"1. このLTV∞とCAC上限をもとに、広告予算の上限をどう設定すべきですか？\n"
            f"2. 顧客獲得チャネル別にROIを評価するには何が必要ですか？\n"
            f"3. LTVを高めるために優先すべき施策は何ですか？\n"
            f"4. このビジネスに最適なLTV:CAC比率の目安を教えてください。"
        )
        pb2 = s5.shapes.add_shape(1, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8))
        pb2.fill.solid(); pb2.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
        pb2.line.color.rgb = GOLD
        txbox(s5, p2_text, 0.65, 1.35, 12.0, 5.6, size=9, color=WHITE)

        # ── Slide 6: AI Prompt 3 - 精度の検証 ──
        s6 = prs.slides.add_slide(blank)
        add_bg(s6, prs)
        txbox(s6, 'AIプロンプト ③  精度の検証', 0.5, 0.25, 12, 0.5, size=18, bold=True, color=WHITE)
        txbox(s6, 'Claude / ChatGPT / Gemini にコピペしてください', 0.5, 0.8, 12, 0.3, size=9, color=GRAY)
        dormancy_q = "4. 解約日ベースで分析していますが、解約データの欠損や遅延がある場合にLTV推定にどんな影響が出ますか？" if dormancy_days is None else f"4. 休眠判定{dormancy_label}の設定はこのビジネスに適切ですか？最適な判定日数を決める感度分析の手順を教えてください。"
        p3_text = (
            f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata}\n\n【質問】\n"
            f"1. このデータ件数と解約件数でWeibullフィッティングの信頼性はどう評価できますか？\n"
            f"2. R²={r2:.4f}は十分ですか？改善するにはどうすればよいですか？\n"
            f"3. Weibullモデルの仮定が成立していない可能性はありますか？どうチェックすればよいですか？\n"
            f"{dormancy_q}"
        )
        pb3 = s6.shapes.add_shape(1, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8))
        pb3.fill.solid(); pb3.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
        pb3.line.color.rgb = GOLD
        txbox(s6, p3_text, 0.65, 1.35, 12.0, 5.6, size=9, color=WHITE)

        # ── セグメント別スライド追加 ──
        if segment_cols_input.strip():
            seg_cols_pp = [c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
            for sc in seg_cols_pp:
                # セグメント別データ計算
                seg_vals = df[sc].dropna().unique()
                pp_rows = []
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = compute_km(df_s)
                        k_s, lam_s, r2_s, _ = fit_weibull(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['arpu_daily'].mean()
                        gp_s   = arpu_s * gpm
                        ltv_r, _ = ltv_inf(k_s, lam_s, arpu_s)
                        ltv_g, _ = ltv_inf(k_s, lam_s, gp_s)
                        pp_rows.append({'seg': str(sv), 'n': len(df_s), 'ltv_r': ltv_r, 'ltv_g': ltv_g, 'cac': ltv_g/cac_n, 'total': ltv_r*len(df_s)})
                    except Exception:
                        continue
                if not pp_rows:
                    continue
                pp_rows.sort(key=lambda x: x['ltv_r'], reverse=True)
                top10 = pp_rows[:10]
                best_pp = top10[0]
                avg_ltv_pp = sum(r['ltv_r'] for r in pp_rows) / len(pp_rows)
                premium_pp = (best_pp['ltv_r'] - avg_ltv_pp) / avg_ltv_pp * 100

                # スライド追加
                s_seg = prs.slides.add_slide(blank)
                add_bg(s_seg, prs)
                txbox(s_seg, f'セグメント別 LTV∞ 比較：{sc}', 0.5, 0.2, 12.3, 0.5, size=18, bold=True, color=WHITE)
                txbox(s_seg, f'上位{len(top10)}セグメント（LTV∞降順）', 0.5, 0.75, 12.3, 0.3, size=9, color=GRAY)

                # テーブル
                col_x = [0.5, 2.8, 4.8, 6.6, 8.4, 10.2]
                headers = ['セグメント', '顧客数', 'LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）', '総ポテンシャル']
                for cx, hd in zip(col_x, headers):
                    txbox(s_seg, hd, cx, 1.1, 1.9, 0.3, size=8, bold=True, color=GOLD)
                row_y = 1.45
                for r in top10:
                    vals = [r['seg'], f"{r['n']:,}", f"¥{r['ltv_r']:,.0f}", f"¥{r['ltv_g']:,.0f}", f"¥{r['cac']:,.0f}", f"¥{r['total']:,.0f}"]
                    for cx, v in zip(col_x, vals):
                        txbox(s_seg, v, cx, row_y, 1.9, 0.28, size=8, color=WHITE)
                    row_y += 0.3

                # 推奨ボックス
                rec_box = s_seg.shapes.add_shape(1, Inches(0.5), Inches(5.0), Inches(12.3), Inches(1.8))
                rec_box.fill.solid(); rec_box.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
                rec_box.line.color.rgb = GOLD
                rec_text = (
                    f"\U0001f3af 優先獲得推奨：{best_pp['seg']}\n"
                    f"LTV∞（売上）¥{best_pp['ltv_r']:,.0f}（全平均比 +{premium_pp:.1f}%）　"
                    f"LTV∞（粗利）¥{best_pp['ltv_g']:,.0f}　"
                    f"CAC上限（粗利）¥{best_pp['cac']:,.0f}\n"
                    f"→ このセグメントに広告予算を集中させることで、競合より高いCPAで入札しながら収益性を維持できます。"
                )
                if cac_known:
                    ratio_pp = best_pp['ltv_g'] / cac_input
                    judge_pp = "✓ 健全" if ratio_pp >= 3.0 else "⚠️ 要改善"
                    rec_text += f"\nLTV:CAC比率（粗利）= {ratio_pp:.1f}:1　{judge_pp}"
                # 優先セグメントの暫定LTV専用スライド
                try:
                    km_best = compute_km(df[df[sc] == best_pp['seg']])
                    k_b, lam_b, r2_b, _ = fit_weibull(km_best)
                    if k_b is not None:
                        arpu_b = df[df[sc] == best_pp['seg']]['arpu_daily'].mean()
                        s_hor = prs.slides.add_slide(blank)
                        add_bg(s_hor, prs)
                        txbox(s_hor, f'優先獲得推奨セグメント：{best_pp["seg"]}　暫定LTV（観測期間別）', 0.5, 0.2, 12.3, 0.5, size=16, bold=True, color=WHITE)
                        txbox(s_hor, f'セグメント軸：{sc}　顧客数：{best_pp["n"]:,}件　k={k_b:.3f}　λ={lam_b:.1f}日　R²={r2_b:.3f}', 0.5, 0.75, 12.3, 0.3, size=9, color=GRAY)

                        cols_hor = ['ホライズン', '暫定LTV（売上）', 'LTV∞比', 'CAC上限（粗利）']
                        col_xh = [0.5, 3.3, 7.0, 9.8]
                        for cx, ch in zip(col_xh, cols_hor):
                            txbox(s_hor, ch, cx, 1.2, 3.0, 0.35, size=9, bold=True, color=GOLD)
                        hl2 = s_hor.shapes.add_shape(1, Inches(0.5), Inches(1.57), Inches(12.3), Inches(0.02))
                        hl2.fill.solid(); hl2.fill.fore_color.rgb = GOLD; hl2.line.fill.background()

                        row_yh = 1.65
                        for i, h in enumerate(horizons):
                            lh_b = ltv_horizon(k_b, lam_b, arpu_b, h)
                            ltv_inf_b = lam_b * __import__('scipy').special.gamma(1 + 1/k_b) * arpu_b
                            label = f'{h}日' if h < 365 else f'{h//365}年'
                            row_vals_h = [label, f'¥{lh_b:,.0f}', f'{lh_b/ltv_inf_b*100:.1f}%', f'¥{lh_b/cac_n:,.0f}']
                            bg_h = s_hor.shapes.add_shape(1, Inches(0.5), Inches(row_yh - 0.02), Inches(12.3), Inches(0.38))
                            bg_h.fill.solid()
                            bg_h.fill.fore_color.rgb = RGBColor(0x1a,0x3a,0x4a) if i%2==0 else RGBColor(0x0d,0x1f,0x2d)
                            bg_h.line.fill.background()
                            for cx, rv in zip(col_xh, row_vals_h):
                                txbox(s_hor, rv, cx, row_yh, 3.0, 0.35, size=10, color=WHITE)
                            row_yh += 0.4

                        # 生存曲線グラフ
                        import matplotlib.pyplot as plt_pp
                        fig_pp, ax_pp = plt_pp.subplots(figsize=(5, 3))
                        fig_pp.patch.set_facecolor('#111820')
                        ax_pp.set_facecolor('#111820')
                        km_b_df = km_best
                        ax_pp.step(km_b_df['t'], km_b_df['S'], color='#56b4d3', lw=1.5, label='KM observed')
                        t_range = km_b_df['t'].values
                        ax_pp.plot(t_range, [float(weibull_s(t, k_b, lam_b)) for t in t_range], '--', color='#a8dadc', lw=1.2, label=f'Weibull fit (R²={r2_b:.3f})')
                        ax_pp.set_xlabel('Days', color='#888', fontsize=8)
                        ax_pp.set_ylabel('Survival Rate', color='#888', fontsize=8)
                        ax_pp.tick_params(colors='#666', labelsize=7)
                        ax_pp.legend(fontsize=7, framealpha=0.2)
                        ax_pp.grid(True, alpha=0.2)
                        ax_pp.set_title(f'Survival Curve: {best_pp["seg"]}', color='#ccc', fontsize=9)
                        fig_pp.tight_layout()
                        buf_pp = io.BytesIO()
                        fig_pp.savefig(buf_pp, format='png', dpi=120, bbox_inches='tight', facecolor='#111820')
                        buf_pp.seek(0)
                        plt_pp.close()
                        s_hor.shapes.add_picture(buf_pp, Inches(0.5), Inches(4.55), Inches(12.3), Inches(2.7))
                except Exception:
                    pass

        # 全セグメント詳細スライド（4分割レイアウト）
        if segment_cols_input.strip():
            seg_cols_all = [c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
            for sc_all in seg_cols_all:
                seg_vals_all = df[sc_all].dropna().unique()
                for sv_all in sorted(seg_vals_all):
                    df_sv_all = df[df[sc_all] == sv_all]
                    if len(df_sv_all) < 10 or df_sv_all['event'].sum() < 5:
                        continue
                    try:
                        km_sv_all = compute_km(df_sv_all)
                        k_sv, lam_sv, r2_sv, _ = fit_weibull(km_sv_all)
                        if k_sv is None: continue
                        arpu_sv = df_sv_all['arpu_daily'].mean()
                        ltv_inf_sv = lam_sv * __import__('scipy').special.gamma(1 + 1/k_sv) * arpu_sv

                        s_all = prs.slides.add_slide(blank)
                        add_bg(s_all, prs)

                        # タイトル
                        txbox(s_all, f'{sc_all}：{str(sv_all)}　詳細分析', 0.5, 0.1, 12.3, 0.45, size=15, bold=True, color=WHITE)
                        txbox(s_all, f'顧客数 {len(df_sv_all):,}件　LTV∞（売上）¥{ltv_inf_sv:,.0f}　k={k_sv:.3f}　λ={lam_sv:.1f}日　R²={r2_sv:.3f}', 0.5, 0.58, 12.3, 0.28, size=8, color=GRAY)

                        # ── 上段左：生存曲線 ──
                        import matplotlib.pyplot as plt_sv1
                        fig_sv1, ax_sv1 = plt_sv1.subplots(figsize=(5.5, 3.2))
                        fig_sv1.patch.set_facecolor('#111820')
                        ax_sv1.set_facecolor('#111820')
                        ax_sv1.step(km_sv_all['t'], km_sv_all['S'], color='#56b4d3', lw=1.5, label='KM Curve (Observed)')
                        t_ra = km_sv_all['t'].values
                        ax_sv1.plot(t_ra, [float(weibull_s(t, k_sv, lam_sv)) for t in t_ra],
                                   '--', color='#a8dadc', lw=1.2, label='Weibull Fit')
                        ax_sv1.set_xlabel('Days', color='#888', fontsize=8)
                        ax_sv1.set_ylabel('Survival Rate S(t)', color='#888', fontsize=8)
                        ax_sv1.tick_params(colors='#666', labelsize=7)
                        ax_sv1.legend(fontsize=7, framealpha=0.2)
                        ax_sv1.grid(True, alpha=0.2)
                        ax_sv1.set_title('Survival Curve', color='#ccc', fontsize=9)
                        fig_sv1.tight_layout()
                        buf_sv1 = io.BytesIO()
                        fig_sv1.savefig(buf_sv1, format='png', dpi=120, bbox_inches='tight', facecolor='#111820')
                        buf_sv1.seek(0)
                        plt_sv1.close()
                        s_all.shapes.add_picture(buf_sv1, Inches(0.3), Inches(0.9), Inches(6.0), Inches(3.5))

                        # ── 上段右：Weibull直線化プロット ──
                        import numpy as np_sv2
                        km_fit2 = km_sv_all[km_sv_all['S'] > 0]
                        ln_t2 = np_sv2.log(km_fit2['t'].values.astype(float) + 1e-10)
                        ln_neg2 = np_sv2.log(-np_sv2.log(km_fit2['S'].values.astype(float) + 1e-15))
                        valid2 = np_sv2.isfinite(ln_t2) & np_sv2.isfinite(ln_neg2)
                        ln_t2, ln_neg2 = ln_t2[valid2], ln_neg2[valid2]
                        slope2, int2, _, _, _ = __import__('scipy').stats.linregress(ln_t2, ln_neg2)
                        x_l2 = np_sv2.linspace(ln_t2.min(), ln_t2.max(), 100)

                        fig_sv2, ax_sv2 = plt_sv1.subplots(figsize=(5.5, 3.2))
                        fig_sv2.patch.set_facecolor('#111820')
                        ax_sv2.set_facecolor('#111820')
                        ax_sv2.scatter(ln_t2, ln_neg2, color='#56b4d3', s=18, alpha=0.75, label='Observed')
                        ax_sv2.plot(x_l2, slope2 * x_l2 + int2, '--', color='#a8dadc', lw=1.5, label=f'R²={r2_sv:.3f}')
                        ax_sv2.annotate(f'y = {slope2:.4f}x + {int2:.4f}', xy=(0.05, 0.93), xycoords='axes fraction', color='#777', fontsize=8)
                        ax_sv2.set_xlabel('ln(t)', color='#888', fontsize=8)
                        ax_sv2.set_ylabel('ln(−ln(S(t)))', color='#888', fontsize=8)
                        ax_sv2.tick_params(colors='#666', labelsize=7)
                        ax_sv2.legend(fontsize=7, framealpha=0.2)
                        ax_sv2.grid(True, alpha=0.2)
                        ax_sv2.set_title('Weibull Linearization Plot', color='#ccc', fontsize=9)
                        fig_sv2.tight_layout()
                        buf_sv2 = io.BytesIO()
                        fig_sv2.savefig(buf_sv2, format='png', dpi=120, bbox_inches='tight', facecolor='#111820')
                        buf_sv2.seek(0)
                        plt_sv1.close()
                        s_all.shapes.add_picture(buf_sv2, Inches(6.6), Inches(0.9), Inches(6.0), Inches(3.5))

                        # ── 下段左：暫定LTVテーブル ──
                        cols_ha = ['ホライズン', '暫定LTV（売上）', 'LTV∞比', 'CAC上限（粗利）']
                        col_xa = [0.3, 1.9, 4.1, 5.4]
                        for cx, ch in zip(col_xa, cols_ha):
                            txbox(s_all, ch, cx, 4.55, 1.5, 0.3, size=7, bold=True, color=GOLD)
                        hl_a = s_all.shapes.add_shape(1, Inches(0.3), Inches(4.85), Inches(6.0), Inches(0.02))
                        hl_a.fill.solid(); hl_a.fill.fore_color.rgb = GOLD; hl_a.line.fill.background()
                        row_ya = 4.92
                        for i_h, h in enumerate(horizons):
                            lh_a = ltv_horizon(k_sv, lam_sv, arpu_sv, h)
                            label_a = f'{h}日' if h < 365 else f'{h//365}年'
                            row_vals_a = [label_a, f'¥{lh_a:,.0f}', f'{lh_a/ltv_inf_sv*100:.1f}%', f'¥{lh_a*gpm/cac_n:,.0f}']
                            bg_a = s_all.shapes.add_shape(1, Inches(0.3), Inches(row_ya - 0.02), Inches(6.0), Inches(0.32))
                            bg_a.fill.solid()
                            bg_a.fill.fore_color.rgb = RGBColor(0x1a,0x3a,0x4a) if i_h%2==0 else RGBColor(0x0d,0x1f,0x2d)
                            bg_a.line.fill.background()
                            for cx, rv in zip(col_xa, row_vals_a):
                                txbox(s_all, rv, cx, row_ya, 1.5, 0.3, size=8, color=WHITE)
                            row_ya += 0.33

                        # ── 下段右：コメント欄 ──
                        comment_box = s_all.shapes.add_shape(1, Inches(6.6), Inches(4.55), Inches(6.0), Inches(2.85))
                        comment_box.fill.solid(); comment_box.fill.fore_color.rgb = RGBColor(0x0d,0x1f,0x2d)
                        comment_box.line.color.rgb = GOLD
                        k_desc_c = "初期離脱型（契約直後の離脱が多い）" if k_sv < 1 else "逓増離脱型（時間とともに離脱増）"
                        ltv_1y_c = ltv_horizon(k_sv, lam_sv, arpu_sv, 365)
                        ltv_2y_c = ltv_horizon(k_sv, lam_sv, arpu_sv, 730)
                        pct_1y_c = ltv_1y_c / ltv_inf_sv * 100
                        pct_2y_c = ltv_2y_c / ltv_inf_sv * 100
                        k_desc_c = "初期離脱型" if k_sv < 1 else "逓増離脱型"
                        comment_text = (
                            "【分析コメント】\n"
                            f"k={k_sv:.3f}（{k_desc_c}）\n"
                            f"λ={lam_sv:.1f}日（約{lam_sv/365:.1f}年）\n\n"
                            f"LTV∞（売上）：¥{ltv_inf_sv:,.0f}\n"
                            f"CAC上限（粗利）：¥{ltv_inf_sv*gpm/cac_n:,.0f}\n\n"
                            f"1年時点：LTV∞の{pct_1y_c:.1f}%（¥{ltv_1y_c:,.0f}）\n"
                            f"2年時点：LTV∞の{pct_2y_c:.1f}%（¥{ltv_2y_c:,.0f}）"
                        )
                        txbox(s_all, comment_text, 6.65, 4.62, 5.85, 2.7, size=8, color=WHITE)

                    except Exception:
                        continue

        pptx_buf = io.BytesIO()
        prs.save(pptx_buf)
        pptx_buf.seek(0)
        st.download_button("📑 PowerPointダウンロード", pptx_buf,
                           file_name=f"LTV分析_{client_name or 'report'}.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    except ImportError:
        st.caption("PowerPoint出力には `pip install python-pptx` が必要です")
    except Exception as e:
        st.caption(f"PowerPoint出力エラー: {e}")

# ── PDF export ────────────────────────────────────────────────
with exp3:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))

        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('T', fontName='HeiseiMin-W3', fontSize=18, spaceAfter=6,
                                     textColor=colors.HexColor('#56b4d3'))
        h2_style    = ParagraphStyle('H2', fontName='HeiseiMin-W3', fontSize=13, spaceAfter=4,
                                     textColor=colors.HexColor('#111111'), spaceBefore=14)
        body_style  = ParagraphStyle('B', fontName='HeiseiMin-W3', fontSize=9,
                                     textColor=colors.HexColor('#333333'), spaceAfter=3)

        story = []
        story.append(Paragraph('LTV Analysis Report', title_style))
        if client_name: story.append(Paragraph(f'クライアント: {client_name}', body_style))
        story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph('分析結果サマリー', h2_style))
        tdata = [
            ['指標', '値'],
            ['LTV∞（売上ベース）', f'¥{ltv_rev:,.0f}'],
            ['LTV∞（粗利ベース・CAC算出用）', f'¥{ltv_val:,.0f}'],
            [f'CAC上限（粗利ベース・{cac_label}）', f'¥{cac_upper:,.0f}'],
            ['Weibull k（形状パラメータ）', f'{k:.4f}  →  {"k<1: 初期離脱型" if k < 1 else "k>1: 逓増離脱型"}'],
            ['Weibull λ（尺度パラメータ）', f'{lam:.1f}日（約{lam/365:.1f}年）'],
            ['R²（フィット精度）', f'{r2:.4f}  →  {"✓ 良好（0.9以上）" if r2 >= 0.9 else "△ やや低め（0.9未満）"}'],
            ['顧客数', f'{len(df):,}件'],
            ['解約済み / 継続中', f'{int(df["event"].sum()):,}件 / {int((df["event"]==0).sum()):,}件'],
            ['平均日次ARPU（売上）', f'¥{arpu_daily:,.2f}'],
            ['GPM（粗利率）', f'{gpm:.1%}'],
            ['ビジネスタイプ', business_type],
            ['休眠判定', dormancy_label],
        ]
        t = Table(tdata, colWidths=[9*cm, 6*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#c8b89a')),
            ('TEXTCOLOR',  (0,1), (-1,-1), colors.HexColor('#cccccc')),
            ('FONTNAME',   (0,0), (-1,-1), 'HeiseiMin-W3'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#111111'), colors.HexColor('#161616')]),
            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#333333')),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)

        story.append(Paragraph('Survival Curve', h2_style))
        buf1.seek(0)
        story.append(Image(buf1, width=13*cm, height=7.5*cm))

        story.append(Paragraph('Weibull Linearization Plot', h2_style))
        buf2.seek(0)
        story.append(Image(buf2, width=13*cm, height=7.5*cm))

        # AI Prompts
        pdata_pdf = (
            f"顧客数: {len(df):,}件（解約済み: {df['event'].sum():,}件）\n"
            f"平均日次ARPU（売上）: ¥{arpu_daily:,.2f} / GPM: {gpm:.0%}\n"
            f"LTV∞ 売上ベース: ¥{ltv_rev:,.0f} / 粗利ベース: ¥{ltv_val:,.0f}\n"
            f"CAC上限 ({cac_label}): ¥{cac_upper:,.0f}\n"
            f"Weibull k={k:.4f} / λ={lam:.1f}日 / R²={r2:.4f}"
        )
        prompt_style = ParagraphStyle('P', fontName='HeiseiMin-W3', fontSize=8,
                                      textColor=colors.HexColor('#111111'),
                                      spaceAfter=2, leading=12,
                                      leftIndent=8, rightIndent=8)
        label_style = ParagraphStyle('L', fontName='HeiseiMin-W3', fontSize=8,
                                     textColor=colors.HexColor('#1d6fa4'), spaceAfter=2, spaceBefore=10)

        prompts_pdf = [
            ('AIプロンプト① 結果の読み方',
             f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata_pdf}\n\n【質問】\n"
             f"1. Weibullのkとλの値は何を意味していますか？顧客離脱パターンはどう解釈すればよいですか？\n"
             f"2. LTV∞（売上ベース¥{ltv_rev:,.0f}）の値は適切な水準ですか？\n"
             f"3. R²={r2:.4f}からフィット精度はどう評価できますか？\n"
             f"4. この結果で特に注意すべき点があれば教えてください。"),
            ('AIプロンプト② マーケ戦略への活用',
             f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata_pdf}\n\n【質問】\n"
             f"1. このLTV∞とCAC上限をもとに、広告予算の上限をどう設定すべきですか？\n"
             f"2. 顧客獲得チャネル別にROIを評価するには何が必要ですか？\n"
             f"3. LTVを高めるために優先すべき施策は何ですか？\n"
             f"4. このビジネスに最適なLTV:CAC比率の目安を教えてください。"),
            ('AIプロンプト③ 精度の検証',
             f"私はLTV分析ツールを使い、以下の結果を得ました。\n{pdata_pdf}\n\n【質問】\n"
             f"1. このデータ件数と解約件数でWeibullフィッティングの信頼性はどう評価できますか？\n"
             f"2. R²={r2:.4f}は十分ですか？改善するにはどうすればよいですか？\n"
             f"3. Weibullモデルの仮定が成立していない可能性はありますか？どうチェックすればよいですか？\n"
             f"4. {"解約日ベースで分析していますが、解約データの欠損や遅延がある場合にLTV推定にどんな影響が出ますか？" if dormancy_days is None else f"休眠判定{dormancy_label}の設定はこのビジネスに適切ですか？最適な判定日数を決める感度分析の手順を教えてください。"}"),
        ]

        # セグメント別セクション
        if segment_cols_input.strip():
            seg_cols_pdf = [c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
            for sc in seg_cols_pdf:
                story.append(Paragraph(f'セグメント別 LTV∞ 分析：{sc}', h2_style))
                seg_vals = df[sc].dropna().unique()
                pdf_rows = [['セグメント', '顧客数', 'LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）', 'k', 'R²']]
                best_pdf = None
                avg_ltv_pdf = []
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = compute_km(df_s)
                        k_s, lam_s, r2_s, _ = fit_weibull(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['arpu_daily'].mean()
                        gp_s   = arpu_s * gpm
                        ltv_r, _ = ltv_inf(k_s, lam_s, arpu_s)
                        ltv_g, _ = ltv_inf(k_s, lam_s, gp_s)
                        pdf_rows.append([str(sv), f'{len(df_s):,}', f'¥{ltv_r:,.0f}', f'¥{ltv_g:,.0f}', f'¥{ltv_g/cac_n:,.0f}', f'{k_s:.3f}', f'{r2_s:.3f}'])
                        avg_ltv_pdf.append(ltv_r)
                        if best_pdf is None or ltv_r > best_pdf['ltv_r']:
                            best_pdf = {'seg': str(sv), 'ltv_r': ltv_r, 'ltv_g': ltv_g, 'cac': ltv_g/cac_n}
                    except Exception:
                        continue
                # テーブル（上位10件）
                pdf_rows_show = [pdf_rows[0]] + sorted(pdf_rows[1:], key=lambda x: float(x[2].replace('¥','').replace(',','')), reverse=True)[:10]
                t_seg = Table(pdf_rows_show, colWidths=[3*cm, 1.8*cm, 2.5*cm, 2.5*cm, 2.5*cm, 1.5*cm, 1.5*cm])
                t_seg.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a1a')),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#56b4d3')),
                    ('TEXTCOLOR',  (0,1), (-1,-1), colors.HexColor('#111111')),
                    ('FONTNAME',   (0,0), (-1,-1), 'HeiseiMin-W3'),
                    ('FONTSIZE',   (0,0), (-1,-1), 8),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f5f5f5'), colors.HexColor('#ffffff')]),
                    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(t_seg)
                # 推奨セグメント + 暫定LTV + 生存曲線
                if best_pdf and avg_ltv_pdf:
                    avg_pp = sum(avg_ltv_pdf) / len(avg_ltv_pdf)
                    prem = (best_pdf['ltv_r'] - avg_pp) / avg_pp * 100
                    rec_style = ParagraphStyle('R', fontName='HeiseiMin-W3', fontSize=9,
                                               textColor=colors.HexColor('#111111'),
                                               backColor=colors.HexColor('#e8f4fb'),
                                               borderPadding=8, spaceAfter=4, spaceBefore=6,
                                               leftIndent=8, rightIndent=8)
                    story.append(Paragraph(
                        f"🎯 優先獲得推奨：{best_pdf['seg']}　"
                        f"LTV∞（売上）¥{best_pdf['ltv_r']:,.0f}（全平均比+{prem:.1f}%）　"
                        f"CAC上限（粗利）¥{best_pdf['cac']:,.0f}",
                        rec_style
                    ))

                    # 優先セグメントの暫定LTVテーブル
                    try:
                        df_best = df[df[sc] == best_pdf['seg']]
                        km_best_pdf = compute_km(df_best)
                        k_bp, lam_bp, r2_bp, _ = fit_weibull(km_best_pdf)
                        if k_bp is not None:
                            arpu_bp = df_best['arpu_daily'].mean()
                            ltv_inf_bp = lam_bp * __import__('scipy').special.gamma(1 + 1/k_bp) * arpu_bp

                            story.append(Spacer(1, 0.2*cm))
                            story.append(Paragraph(
                                f'優先獲得推奨セグメント「{best_pdf["seg"]}」の暫定LTV（k={k_bp:.3f}　λ={lam_bp:.1f}日　R²={r2_bp:.3f}）',
                                h2_style
                            ))
                            hor_data = [['ホライズン', '暫定LTV（売上）', 'LTV∞比', 'CAC上限（粗利）']]
                            for h in horizons:
                                lh_bp = ltv_horizon(k_bp, lam_bp, arpu_bp, h)
                                label = f'{h}日' if h < 365 else f'{h//365}年'
                                hor_data.append([label, f'¥{lh_bp:,.0f}', f'{lh_bp/ltv_inf_bp*100:.1f}%', f'¥{lh_bp/cac_n:,.0f}'])
                            t_hor = Table(hor_data, colWidths=[3*cm, 4*cm, 3*cm, 4*cm])
                            t_hor.setStyle(TableStyle([
                                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a1a')),
                                ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#56b4d3')),
                                ('TEXTCOLOR',  (0,1), (-1,-1), colors.HexColor('#111111')),
                                ('FONTNAME',   (0,0), (-1,-1), 'HeiseiMin-W3'),
                                ('FONTSIZE',   (0,0), (-1,-1), 8),
                                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f5f5f5'), colors.HexColor('#ffffff')]),
                                ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
                                ('LEFTPADDING', (0,0), (-1,-1), 6),
                            ]))
                            story.append(t_hor)

                            # 生存曲線グラフ
                            import matplotlib.pyplot as plt_pdf2
                            fig_bp, ax_bp = plt_pdf2.subplots(figsize=(7, 3.5))
                            fig_bp.patch.set_facecolor('white')
                            ax_bp.step(km_best_pdf['t'], km_best_pdf['S'], color='#1d6fa4', lw=1.5, label='KM observed')
                            t_range_bp = km_best_pdf['t'].values
                            ax_bp.plot(t_range_bp, [float(weibull_s(t, k_bp, lam_bp)) for t in t_range_bp],
                                      '--', color='#56b4d3', lw=1.2, label=f'Weibull fit (R²={r2_bp:.3f})')
                            ax_bp.set_xlabel('Days', fontsize=9)
                            ax_bp.set_ylabel('Survival Rate', fontsize=9)
                            ax_bp.legend(fontsize=8)
                            ax_bp.grid(True, alpha=0.3)
                            ax_bp.set_title(f'Survival Curve: {best_pdf["seg"]}', fontsize=10)
                            fig_bp.tight_layout()
                            buf_bp = io.BytesIO()
                            fig_bp.savefig(buf_bp, format='png', dpi=120, bbox_inches='tight')
                            buf_bp.seek(0)
                            plt_pdf2.close()
                            story.append(Image(buf_bp, width=13*cm, height=6.5*cm))
                    except Exception:
                        pass

                # 全セグメントの詳細分析（2グラフ横並び）
                story.append(Paragraph(f'全セグメント詳細（{sc}）', h2_style))
                for sv in sorted(seg_vals):
                    df_sv2 = df[df[sc] == sv]
                    if len(df_sv2) < 10 or df_sv2['event'].sum() < 5:
                        continue
                    try:
                        km_sv2 = compute_km(df_sv2)
                        k_sv2, lam_sv2, r2_sv2, _ = fit_weibull(km_sv2)
                        if k_sv2 is None: continue
                        arpu_sv2 = df_sv2['arpu_daily'].mean()
                        ltv_inf_sv2 = lam_sv2 * __import__('scipy').special.gamma(1 + 1/k_sv2) * arpu_sv2

                        story.append(Paragraph(
                            f'{str(sv)}　（{len(df_sv2):,}件 / LTV∞ ¥{ltv_inf_sv2:,.0f} / k={k_sv2:.3f} / λ={lam_sv2:.1f}日 / R²={r2_sv2:.3f}）',
                            label_style
                        ))

                        # 生存曲線＋Weibull直線化プロット（横並び）
                        import matplotlib.pyplot as plt_pdf_all
                        import numpy as np_pdf_all
                        fig_2g, (ax_g1, ax_g2) = plt_pdf_all.subplots(1, 2, figsize=(14, 4.5))
                        fig_2g.patch.set_facecolor('white')

                        # 左：生存曲線
                        ax_g1.step(km_sv2['t'], km_sv2['S'], color='#1d6fa4', lw=1.5, label='KM Curve (Observed)')
                        t_r2 = km_sv2['t'].values
                        ax_g1.plot(t_r2, [float(weibull_s(t, k_sv2, lam_sv2)) for t in t_r2],
                                  '--', color='#56b4d3', lw=1.2, label=f'Weibull Fit')
                        ax_g1.set_xlabel('Days', fontsize=9)
                        ax_g1.set_ylabel('Survival Rate S(t)', fontsize=9)
                        ax_g1.tick_params(labelsize=8)
                        ax_g1.legend(fontsize=8)
                        ax_g1.grid(True, alpha=0.3)
                        ax_g1.set_title(f'Survival Curve: {str(sv)}', fontsize=10)

                        # 右：Weibull直線化プロット
                        km_fit_p = km_sv2[km_sv2['S'] > 0]
                        ln_t_p = np_pdf_all.log(km_fit_p['t'].values.astype(float) + 1e-10)
                        ln_neg_p = np_pdf_all.log(-np_pdf_all.log(km_fit_p['S'].values.astype(float) + 1e-15))
                        valid_p = np_pdf_all.isfinite(ln_t_p) & np_pdf_all.isfinite(ln_neg_p)
                        ln_t_p, ln_neg_p = ln_t_p[valid_p], ln_neg_p[valid_p]
                        slope_p, int_p, _, _, _ = __import__('scipy').stats.linregress(ln_t_p, ln_neg_p)
                        x_lp = np_pdf_all.linspace(ln_t_p.min(), ln_t_p.max(), 100)
                        ax_g2.scatter(ln_t_p, ln_neg_p, color='#1d6fa4', s=18, alpha=0.75, label='Observed')
                        ax_g2.plot(x_lp, slope_p * x_lp + int_p, '--', color='#56b4d3', lw=1.5, label=f'R²={r2_sv2:.3f}')
                        ax_g2.annotate(f'y = {slope_p:.4f}x + {int_p:.4f}', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=8)
                        ax_g2.set_xlabel('ln(t)', fontsize=9)
                        ax_g2.set_ylabel('ln(−ln(S(t)))', fontsize=9)
                        ax_g2.tick_params(labelsize=8)
                        ax_g2.legend(fontsize=8)
                        ax_g2.grid(True, alpha=0.3)
                        ax_g2.set_title(f'Weibull Linearization Plot: {str(sv)}', fontsize=10)

                        fig_2g.tight_layout()
                        buf_2g = io.BytesIO()
                        fig_2g.savefig(buf_2g, format='png', dpi=100, bbox_inches='tight')
                        buf_2g.seek(0)
                        plt_pdf_all.close()
                        story.append(Image(buf_2g, width=16*cm, height=5.5*cm))

                        # 暫定LTVテーブル
                        hor_data2 = [['ホライズン', '暫定LTV（売上）', 'LTV∞比', 'CAC上限（粗利）']]
                        for h in horizons:
                            lh_sv2 = ltv_horizon(k_sv2, lam_sv2, arpu_sv2, h)
                            label_h = f'{h}日' if h < 365 else f'{h//365}年'
                            hor_data2.append([label_h, f'¥{lh_sv2:,.0f}', f'{lh_sv2/ltv_inf_sv2*100:.1f}%', f'¥{lh_sv2*gpm/cac_n:,.0f}'])
                        t_sv2 = Table(hor_data2, colWidths=[3*cm, 4*cm, 3*cm, 4*cm])
                        t_sv2.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a1a')),
                            ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#56b4d3')),
                            ('TEXTCOLOR',  (0,1), (-1,-1), colors.HexColor('#111111')),
                            ('FONTNAME',   (0,0), (-1,-1), 'HeiseiMin-W3'),
                            ('FONTSIZE',   (0,0), (-1,-1), 7),
                            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f5f5f5'), colors.HexColor('#ffffff')]),
                            ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#cccccc')),
                            ('LEFTPADDING', (0,0), (-1,-1), 4),
                        ]))
                        story.append(t_sv2)
                        story.append(Spacer(1, 0.4*cm))
                    except Exception:
                        continue

                story.append(Spacer(1, 0.3*cm))


        story.append(Paragraph('AIへの質問プロンプト', h2_style))
        story.append(Paragraph('以下のプロンプトをClaude / ChatGPT / Gemini にコピペしてご活用ください。', body_style))
        for label, prompt_text in prompts_pdf:
            story.append(Paragraph(label, label_style))
            for line in prompt_text.split('\n'):
                story.append(Paragraph(line if line else ' ', prompt_style))
            story.append(Spacer(1, 0.2*cm))

        doc.build(story)
        pdf_buf.seek(0)
        st.download_button("📄 PDFダウンロード", pdf_buf,
                           file_name=f"LTV分析_{client_name or 'report'}.pdf",
                           mime="application/pdf")
    except ImportError:
        st.caption("PDF出力には `pip install reportlab` が必要です")
    except Exception as e:
        st.caption(f"PDF出力エラー: {e}")

# ══════════════════════════════════════════════════════════════
# Segment Analysis (PRO)
# ══════════════════════════════════════════════════════════════

# セグメント結果を保存（エクスポート用）
all_seg_results = {}  # {seg_col: seg_df}

if segment_cols_input.strip():
    seg_cols = [c.strip() for c in segment_cols_input.split(',') if c.strip()]
    valid_seg_cols = [c for c in seg_cols if c in df.columns]
    invalid_seg_cols = [c for c in seg_cols if c not in df.columns]

    if invalid_seg_cols:
        st.warning(f"⚠️ 以下の列が見つかりませんでした: {invalid_seg_cols}")

    MAX_SEG_COLS = 5
    if len(valid_seg_cols) > MAX_SEG_COLS:
        st.warning(
            f"⚠️ セグメント軸は最大{MAX_SEG_COLS}列まで指定できます（処理速度の確保のため）。"
            f"現在{len(valid_seg_cols)}列指定されています。"
            f"先頭{MAX_SEG_COLS}列のみ分析します。残りの列は別途入力してください。"
        )
        valid_seg_cols = valid_seg_cols[:MAX_SEG_COLS]

    if valid_seg_cols:
        st.markdown("<div class='section-title'>🔬 セグメント別LTV分析（PRO）</div>", unsafe_allow_html=True)

        for seg_col in valid_seg_cols:
            st.markdown(f"#### 📊 セグメント：`{seg_col}`")

            seg_values = df[seg_col].dropna().unique()
            if len(seg_values) > 50:
                st.warning(f"⚠️ `{seg_col}` のユニーク値が{len(seg_values)}個あります。50個以下にしてください。")
                continue
            elif len(seg_values) > 20:
                st.info(f"ℹ️ `{seg_col}` のユニーク値が{len(seg_values)}個あります。計算に少し時間がかかります。")

            seg_results = []
            seg_figs = []

            progress_bar = st.progress(0, text=f"`{seg_col}` を分析中...")
            for seg_idx, seg_val in enumerate(sorted(seg_values)):
                progress_bar.progress(
                    int((seg_idx + 1) / len(seg_values) * 100),
                    text=f"`{seg_col}` 分析中... {seg_val} ({seg_idx+1}/{len(seg_values)})"
                )
                df_seg = df[df[seg_col] == seg_val].copy()
                if len(df_seg) < 10 or df_seg['event'].sum() < 5:
                    continue

                try:
                    km_seg   = compute_km(df_seg)
                    k_s, lam_s, r2_s, _ = fit_weibull(km_seg)
                    if k_s is None:
                        continue
                    arpu_s   = df_seg['arpu_daily'].mean()
                    gp_s     = arpu_s * gpm
                    ltv_rev_s, _ = ltv_inf(k_s, lam_s, arpu_s)
                    ltv_gp_s, _  = ltv_inf(k_s, lam_s, gp_s)
                    cac_s    = ltv_gp_s / cac_n
                    total_ltv_s = ltv_rev_s * len(df_seg)
                    # 獲得効率スコア（CAC既知の場合）
                    efficiency = (ltv_gp_s / cac_input) if cac_known else None
                    # 優先スコア = LTV∞ × 顧客数の正規化
                    priority_score = ltv_rev_s * len(df_seg)

                    seg_results.append({
                        'セグメント': seg_val,
                        '顧客数': len(df_seg),
                        'LTV∞（売上）': ltv_rev_s,
                        'LTV∞（粗利）': ltv_gp_s,
                        'CAC上限（粗利）': cac_s,
                        '総ポテンシャル': total_ltv_s,
                        'k': k_s,
                        'λ（日）': lam_s,
                        'R²': r2_s,
                        '獲得効率': efficiency,
                        '優先スコア': priority_score,
                    })
                except Exception:
                    continue

            if not seg_results:
                st.caption("分析に十分なデータがありませんでした。")
                continue

            seg_df = pd.DataFrame(seg_results).sort_values('LTV∞（売上）', ascending=False).reset_index(drop=True)
            all_seg_results[seg_col] = seg_df  # エクスポート用に保存

            progress_bar.empty()

            # Plotlyで棒グラフ（日本語フォント問題を完全回避）
            n_seg = len(seg_df)
            bar_colors = [ACCENT if i == 0 else ACCENT2 for i in range(n_seg)]
            fig_height = 400 if n_seg <= 10 else 450 if n_seg <= 20 else 520
            tick_angle = 0 if n_seg <= 8 else -45 if n_seg <= 20 else -90
            tick_size  = 12 if n_seg <= 10 else 10 if n_seg <= 20 else 8

            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Bar(
                x=seg_df['セグメント'].astype(str).tolist(),
                y=seg_df['LTV∞（売上）'].tolist(),
                marker=dict(color=bar_colors),
                text=[f'¥{v:,.0f}' for v in seg_df['LTV∞（売上）']],
                textposition='outside' if n_seg <= 15 else 'none',
            ))
            fig_plotly.update_xaxes(
                tickfont=dict(color='#aaa', size=tick_size),
                tickangle=tick_angle,
                gridcolor='#1a3040',
                linecolor='#1a3040',
            )
            fig_plotly.update_yaxes(
                title_text='LTV∞（¥）',
                tickfont=dict(color='#888'),
                gridcolor='#1a3040',
                tickprefix='¥',
                tickformat=',',
            )
            fig_plotly.update_layout(
                title_text=f'セグメント別 LTV∞ 比較（{seg_col}）',
                paper_bgcolor='#111820',
                plot_bgcolor='#111820',
                height=fig_height,
                showlegend=False,
                margin=dict(t=50, b=120 if tick_angle != 0 else 60, l=60, r=20),
                font=dict(color='#ccc', size=12),
            )
            st.plotly_chart(fig_plotly, use_container_width=True)

            # 結果テーブル
            display_df = seg_df.copy()
            display_df['LTV∞（売上）'] = display_df['LTV∞（売上）'].apply(lambda x: f'¥{x:,.0f}')
            display_df['LTV∞（粗利）'] = display_df['LTV∞（粗利）'].apply(lambda x: f'¥{x:,.0f}')
            display_df['CAC上限（粗利）'] = display_df['CAC上限（粗利）'].apply(lambda x: f'¥{x:,.0f}')
            display_df['総ポテンシャル'] = display_df['総ポテンシャル'].apply(lambda x: f'¥{x:,.0f}')
            display_df['k'] = display_df['k'].apply(lambda x: f'{x:.3f}')
            display_df['λ（日）'] = display_df['λ（日）'].apply(lambda x: f'{x:.1f}')
            display_df['R²'] = display_df['R²'].apply(lambda x: f'{x:.3f}')
            if cac_known:
                display_df['獲得効率'] = display_df['獲得効率'].apply(lambda x: f'{x:.2f}x' if x else '-')
            else:
                display_df = display_df.drop(columns=['獲得効率'])
            display_df = display_df.drop(columns=['優先スコア'])

            # セグメント加重平均LTV∞を計算してテーブル上に表示
            total_n = seg_df['顧客数'].sum()
            weighted_ltv_rev = (seg_df['LTV∞（売上）'] * seg_df['顧客数']).sum() / total_n
            weighted_ltv_gp  = (seg_df['LTV∞（粗利）'] * seg_df['顧客数']).sum() / total_n
            diff_pct = (weighted_ltv_rev - ltv_rev) / ltv_rev * 100
            diff_str = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"
            diff_color = "#a8dadc" if diff_pct >= 0 else "#e8a0a0"

            st.markdown(f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-radius:8px; padding:12px 18px; margin-bottom:8px; font-size:0.85rem; color:#ccc;'>
  <span style='color:#56b4d3; font-weight:500;'>セグメント加重平均 LTV∞</span>　
  <span style='font-size:1.1rem; color:#a8dadc; font-weight:700;'>¥{weighted_ltv_rev:,.0f}</span>（売上）　
  <span style='color:#888;'>¥{weighted_ltv_gp:,.0f}（粗利）</span>　
  <span style='color:{diff_color}; font-size:0.8rem;'>全体値（¥{ltv_rev:,.0f}）との差：{diff_str}</span>
</div>""", unsafe_allow_html=True)

            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # テーブル下の説明
            st.markdown(f"""
<div style='background:#0a1520; border:1px solid #1a3040; border-radius:8px; padding:12px 18px; margin-top:6px; font-size:0.78rem; color:#888; line-height:1.7;'>
  💡 <b style='color:#56b4d3;'>セグメント加重平均と全体LTV∞が異なる理由</b><br>
  全体LTV∞（¥{ltv_rev:,.0f}）はすべての顧客を1つのWeibullモデルでフィットした値です。セグメント加重平均（¥{weighted_ltv_rev:,.0f}）は各セグメントを個別にフィットした後、顧客数で重み付け平均した値です。<br>
  セグメントを切ることで顧客の<b>異質性（heterogeneity）が分離</b>されるため、切り口によって値が変わります。これはモデルの誤りではなく統計的に正しい現象です。<br>
  <b>意思決定の基準：</b>広告投資・予算配分にはセグメント別LTV∞を、ビジネス全体の健全性評価には全体LTV∞を参照することを推奨します。
</div>""", unsafe_allow_html=True)

            # 優先獲得推奨
            best = seg_results[0] if seg_results else None
            if best:
                best_seg = seg_df.iloc[0]
                avg_ltv  = seg_df['LTV∞（売上）'].mean()
                premium  = (best_seg['LTV∞（売上）'] - avg_ltv) / avg_ltv * 100
                cac_best = best_seg['CAC上限（粗利）']
                cac_avg  = ltv_val / cac_n

                if cac_known:
                    ltv_cac_ratio = best_seg['LTV∞（粗利）'] / cac_input
                    ratio_judge = "✓ 健全" if ltv_cac_ratio >= 3.0 else "⚠️ 要改善（推奨3:1以上）"
                    efficiency_str = f"LTV:CAC比率（粗利ベース）= {ltv_cac_ratio:.1f}:1　{ratio_judge}"
                    cac_uplift = ((cac_best - cac_avg) / cac_avg * 100)
                    cac_str = f"許容CAC上限 ¥{cac_best:,.0f}（平均より{cac_uplift:+.1f}%高く設定可能）"
                else:
                    efficiency_str = None
                    cac_str = f"許容CAC上限 ¥{cac_best:,.0f}（平均より¥{cac_best - cac_avg:,.0f}高く設定可能）"

                insight_pro = f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-left:3px solid #56b4d3; border-radius:10px; padding:18px 20px; margin-top:8px; line-height:1.9; font-size:0.85rem; color:#ccc;'>
  <div style='color:#56b4d3; font-size:0.9rem; font-weight:500; margin-bottom:10px;'>🎯 優先獲得推奨セグメント：<b style='color:#a8dadc;'>{best_seg['セグメント']}</b></div>
  <div>・LTV∞（売上）: <b style='color:#a8dadc;'>¥{best_seg['LTV∞（売上）']:,.0f}</b>（全セグメント平均比 <b style='color:#a8dadc;'>+{premium:.1f}%</b>）</div>
  <div>・{cac_str}</div>
  {"<div>・" + efficiency_str + "</div>" if efficiency_str else ""}
  <div style='margin-top:10px; color:#56b4d3;'>💡 このセグメントに顧客獲得投資を集中させることで、CAC上限¥{cac_best:,.0f}の範囲内で収益性を維持しながら積極的な顧客獲得が可能です。</div>
</div>
"""
                st.markdown(insight_pro, unsafe_allow_html=True)

            # ── 全セグメントの詳細分析（上位N件のみブラウザ表示）──
            seg_results_display = seg_results[:seg_display_limit]
            remaining = len(seg_results) - len(seg_results_display)
            st.markdown(f"#### 📅 セグメント別 詳細分析（上位{seg_display_limit}件／全{len(seg_results)}件）")
            if remaining > 0:
                st.caption(f"ℹ️ 残り{remaining}件はパワポ・PDFに出力されます。サイドバーの「ブラウザ表示件数」を増やすと追加表示できます。")
            for sr in seg_results_display:
                sv    = sr['セグメント']
                k_s   = sr['k']
                lam_s = sr['λ（日）']
                r2_s  = sr['R²']
                n_s   = sr['顧客数']
                df_sv  = df[df[seg_col] == sv]
                arpu_s = df_sv['arpu_daily'].mean()
                gp_s   = arpu_s * gpm
                ltv_inf_s = lam_s * __import__('scipy').special.gamma(1 + 1/k_s) * arpu_s
                is_best = (sv == seg_df.iloc[0]['セグメント'])

                with st.expander(
                    f"{'🥇' if is_best else '📊'} {sv}（顧客数 {n_s:,}件 / LTV∞ ¥{sr['LTV∞（売上）']:,.0f} / k={k_s:.3f} / λ={lam_s:.1f}日 / R²={r2_s:.3f}）",
                    expanded=is_best
                ):
                    try:
                        km_sv = compute_km(df_sv)

                        # ── グラフ2枚（全体と同じ）──
                        col_g1, col_g2 = st.columns(2)

                        with col_g1:
                            fig_sv1 = go.Figure()
                            fig_sv1.add_trace(go.Scatter(
                                x=km_sv['t'].tolist(), y=km_sv['S'].tolist(),
                                mode='lines', name='KM Curve (Observed)',
                                line=dict(color=ACCENT, width=2, shape='hv')
                            ))
                            t_max_s = int(km_sv['t'].max())
                            t_range_s = list(range(0, t_max_s + 30, max(1, t_max_s // 200)))
                            fig_sv1.add_trace(go.Scatter(
                                x=t_range_s,
                                y=[float(weibull_s(t, k_s, lam_s)) for t in t_range_s],
                                mode='lines', name=f'Weibull Fit',
                                line=dict(color=ACCENT2, width=1.5, dash='dash')
                            ))
                            fig_sv1.update_layout(
                                title=dict(text='Survival Curve', font=dict(color='#ccc', size=12)),
                                paper_bgcolor='#111820', plot_bgcolor='#111820',
                                height=300, margin=dict(t=40, b=50, l=50, r=10),
                                font=dict(color='#ccc', size=10),
                                legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
                                xaxis=dict(title='Days', gridcolor='#1a3040', tickfont=dict(color='#888')),
                                yaxis=dict(title='Survival Rate S(t)', gridcolor='#1a3040', tickfont=dict(color='#888'), range=[0, 1.05]),
                            )
                            st.plotly_chart(fig_sv1, use_container_width=True)
                            st.caption(f"📌 生存曲線：実測KM曲線（実線）にWeibullフィット（破線）。k={k_s:.3f}・λ={lam_s:.1f}日")

                        with col_g2:
                            # Weibull直線化プロット
                            import numpy as np_sv
                            km_fit = km_sv[km_sv['S'] > 0].copy()
                            ln_t_s = np_sv.log(km_fit['t'].values.astype(float) + 1e-10)
                            ln_neg_ln_S_s = np_sv.log(-np_sv.log(km_fit['S'].values.astype(float) + 1e-15))
                            valid_s = np_sv.isfinite(ln_t_s) & np_sv.isfinite(ln_neg_ln_S_s)
                            ln_t_s = ln_t_s[valid_s]
                            ln_neg_ln_S_s = ln_neg_ln_S_s[valid_s]
                            slope_s, intercept_s, _, _, _ = __import__('scipy').stats.linregress(ln_t_s, ln_neg_ln_S_s)
                            x_line_s = [float(ln_t_s.min()), float(ln_t_s.max())]
                            y_line_s = [slope_s * x + intercept_s for x in x_line_s]

                            fig_sv2 = go.Figure()
                            fig_sv2.add_trace(go.Scatter(
                                x=ln_t_s.tolist(), y=ln_neg_ln_S_s.tolist(),
                                mode='markers', name='Observed',
                                marker=dict(color=ACCENT, size=5, opacity=0.7)
                            ))
                            fig_sv2.add_trace(go.Scatter(
                                x=x_line_s, y=y_line_s,
                                mode='lines', name=f'Regression Line (R²={r2_s:.3f})',
                                line=dict(color=ACCENT2, width=1.5, dash='dash')
                            ))
                            fig_sv2.add_annotation(
                                x=0.05, y=0.93, xref='paper', yref='paper',
                                text=f'y = {slope_s:.4f}x + {intercept_s:.4f}',
                                showarrow=False, font=dict(color='#777', size=9)
                            )
                            fig_sv2.update_layout(
                                title=dict(text='Weibull Linearization Plot', font=dict(color='#ccc', size=12)),
                                paper_bgcolor='#111820', plot_bgcolor='#111820',
                                height=300, margin=dict(t=40, b=50, l=50, r=10),
                                font=dict(color='#ccc', size=10),
                                legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
                                xaxis=dict(title='ln(t)', gridcolor='#1a3040', tickfont=dict(color='#888')),
                                yaxis=dict(title='ln(−ln(S(t)))', gridcolor='#1a3040', tickfont=dict(color='#888')),
                            )
                            st.plotly_chart(fig_sv2, use_container_width=True)
                            st.caption(f"📌 Weibull直線化プロット：R²={r2_s:.3f}（1.0に近いほど精度高い）")

                        # ── 暫定LTVテーブル ──
                        st.markdown("**暫定LTV（観測期間別）**")
                        hor_rows = []
                        for h in horizons:
                            lh_s = ltv_horizon(k_s, lam_s, arpu_s, h)
                            label = f'{h}日' if h < 365 else f'{h//365}年'
                            hor_rows.append({
                                'ホライズン': label,
                                '暫定LTV（売上）': f'¥{lh_s:,.0f}',
                                'LTV∞比': f'{lh_s/ltv_inf_s*100:.1f}%',
                                'CAC上限（粗利）': f'¥{lh_s*gpm/cac_n:,.0f}',
                            })
                        st.dataframe(pd.DataFrame(hor_rows), hide_index=True, use_container_width=True)

                        # ── 解釈ガイド ──
                        k_desc_s = "k<1: 初期離脱型" if k_s < 1 else "k>1: 逓増離脱型"
                        lam_yr_s = lam_s / 365
                        ltv_1y_s = ltv_horizon(k_s, lam_s, arpu_s, 365)
                        ltv_2y_s = ltv_horizon(k_s, lam_s, arpu_s, 730)
                        pct_1y_s = ltv_1y_s / ltv_inf_s * 100
                        pct_2y_s = ltv_2y_s / ltv_inf_s * 100
                        st.markdown(f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-radius:8px; padding:14px 18px; font-size:0.82rem; color:#ccc; line-height:1.8; margin-top:8px;'>
  <div style='color:#56b4d3; margin-bottom:6px;'>💡 解釈ガイド</div>
  <div>・k={k_s:.3f}（{k_desc_s}）　λ={lam_s:.0f}日（約{lam_yr_s:.1f}年）</div>
  <div>・LTV∞（売上）¥{ltv_inf_s:,.0f}　CAC上限（粗利）¥{sr['CAC上限（粗利）']:,.0f}</div>
  <div>・1年時点でLTV∞の{pct_1y_s:.1f}%（¥{ltv_1y_s:,.0f}）、2年時点で{pct_2y_s:.1f}%（¥{ltv_2y_s:,.0f}）回収</div>
</div>""", unsafe_allow_html=True)

                    except Exception as e_sv:
                        st.caption(f"グラフ生成エラー: {e_sv}")

else:
    st.markdown("<div class='section-title'>🔬 セグメント別LTV分析（PRO）</div>", unsafe_allow_html=True)
    st.info("← サイドバーの「セグメント列名」にCSVの列名を入力してください。例：`plan, channel`")
    st.markdown("""
**使い方：**
1. CSVにセグメント列を追加（例：`plan`列に「月額」「年額」など）
2. サイドバーに列名を入力
3. セグメント別LTV∞・優先獲得推奨が自動で出力されます
    """)

# ══════════════════════════════════════════════════════════════
# Data preview
# ══════════════════════════════════════════════════════════════

with st.expander("📋 読み込んだデータを確認"):
    st.write(f"有効データ: {len(df):,}件 ／ 解約: {df['event'].sum():,}件 ／ 継続中: {(df['event']==0).sum():,}件 ／ 平均日次ARPU: ¥{arpu_daily:,.2f}")
    st.dataframe(
        df[['customer_id','start_date','end_date','duration','event','arpu_daily']].head(30),
        hide_index=True
    )

st.markdown("---")
st.markdown("<p style='color:#333; font-size:0.72rem; text-align:center;'>LTV Analyzer — KM × Weibull Model — Built for marketing analytics professionals</p>", unsafe_allow_html=True)
