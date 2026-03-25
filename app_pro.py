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
    page_title="LTV Analyzer Advanced",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base font ── */
html {
    font-size: 13px;
}
.stApp *:not(code):not(pre):not(.js-plotly-plot):not(.js-plotly-plot *):not([class*="material"]):not([class*="icon"]):not([data-testid="stIconMaterial"]) {
    font-family: 'BIZ UDPGothic', sans-serif !important;
}
.stApp { background-color: #0a0e14; color: #c8d0d8; }

/* ── Sidebar divider lines ── */
[data-testid="stSidebar"] hr,
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] hr,
[data-testid="stSidebarContent"] hr { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1c2430;
    color: #c8d0d8;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8ab4c4;
    margin: 24px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1c2430;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1520;
    border: 1px solid #1a2a3a;
    border-radius: 8px;
    padding: 22px 18px;
    text-align: center;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #2a4a5a; }
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #56b4d3;
    line-height: 1.1;
    word-break: break-all;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 500;
    color: #7a9aaa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
}
.metric-desc {
    font-size: 0.82rem;
    color: #8aaabb;
    margin-top: 3px;
    line-height: 1.3;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.68rem;
    font-weight: 600;
    color: #7ab4c4;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid #1a2a3a;
    padding-bottom: 8px;
    margin: 36px 0 18px 0;
}

/* ── Prompt box ── */
.prompt-box {
    background: #0d1117;
    border: 1px solid #1c2430;
    border-left: 2px solid #56b4d3;
    border-radius: 6px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8a9ab0;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.7;
}

/* ── Help / info box ── */
.help-box {
    background: #0d1117;
    border: 1px solid #1c2430;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 0.82rem;
    color: #5a7a8a;
    margin-top: 8px;
    line-height: 1.6;
}

/* ── Tag ── */
.tag {
    display: inline-block;
    background: #0d1520;
    border: 1px solid #1a2a3a;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.68rem;
    font-weight: 500;
    color: #4a7a8a;
    letter-spacing: 0.04em;
    margin: 2px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #1a2a3a;
    border-radius: 6px;
    margin-bottom: 6px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
/* ── Text / Number input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: #0d1a28 !important;
    color: #c8d0d8 !important;
    border: 1px solid #1c3a4a !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #56b4d3 !important;
    box-shadow: 0 0 0 1px #56b4d3 !important;
}
[data-testid="stNumberInput"] button {
    background-color: #0d1a28 !important;
    border-color: #1c3a4a !important;
    color: #6a9aaa !important;
}

/* ── Radio: 選択時の赤を青に ── */
div[data-baseweb="radio"] [data-checked="true"] { border-color: #56b4d3 !important; }
div[data-baseweb="radio"] [data-checked="true"] div { background-color: #56b4d3 !important; }
div[data-baseweb="radio"] input:checked ~ * { border-color: #56b4d3 !important; }
/* フォーカスリング */
div[data-baseweb="radio"] [data-focused="true"] { box-shadow: 0 0 0 3px rgba(86,180,211,0.3) !important; }



/* ── Download buttons ── */
div.stDownloadButton > button {
    width: 72px !important;
    height: 30px !important;
    padding: 0 !important;
    background: #0d1f2d !important;
    color: #a8dadc !important;
    border: 1.5px solid #56b4d3 !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: background 0.2s, color 0.2s !important;
}
div.stDownloadButton > button:hover {
    background: #56b4d3 !important;
    color: #0d1f2d !important;
}
div.stDownloadButton {
    display: inline-block !important;
    margin-right: 0.2rem !important;
}

/* ── Sample selectbox font ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] select,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    font-size: 0.82rem !important;
}

/* ── Radio & Slider accent color (override Streamlit red) ── */
/* ── Radio label ── */
[data-testid="stRadio"] > label {
    color: #56b4d3 !important;
    font-size: 0.78rem !important;
}
[data-testid="stRadio"] label div p {
    color: #c8d0d8 !important;
    font-size: 0.82rem !important;
}

/* Radio: 選択済みの塗り・ボーダー */
div[data-baseweb="radio"] div { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }
div[data-baseweb="radio"] [data-checked="true"] div { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }
div[data-baseweb="radio"] input:checked + div { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }
[data-testid="stRadio"] [aria-checked="true"] > div { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }

/* Slider: トラック・ハンドル・塗り済みトラック */
div[data-baseweb="slider"] [role="slider"] { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }
div[data-baseweb="slider"] [data-testid="stSlider"] div { background-color: #56b4d3 !important; }
div[data-baseweb="slider"] div[class*="Track"] > div { background-color: #56b4d3 !important; }
div[data-baseweb="slider"] div[class*="InnerTrack"] { background-color: #56b4d3 !important; }
/* Streamlit 1.x のスライダー塗り */
.stSlider [data-baseweb="slider"] > div > div > div:nth-child(2) { background: #56b4d3 !important; }
.stSlider [data-baseweb="slider"] > div > div > div:last-child { background-color: #56b4d3 !important; border-color: #56b4d3 !important; }

/* ── Sidebar text colors ── */
[data-testid="stSidebar"] label { color: #c8d0d8 !important; }
[data-testid="stSidebar"] p { color: #c8d0d8 !important; }
[data-testid="stSidebar"] .stCaption p { color: #3a6a7a !important; font-size: 0.78rem !important; }
[data-testid="stSidebar"] .stRadio label p { color: #c8d0d8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #56b4d3 !important; }

/* ── Toggle font size ── */
[data-testid="stSidebar"] [data-testid="stToggle"] label p,
[data-testid="stSidebar"] [data-testid="stToggle"] > label,
[data-testid="stSidebar"] [data-testid="stToggle"] span:not([data-testid]),
[data-testid="stSidebar"] .stToggle label {
    font-size: 0.82rem !important;
    color: #c8d0d8 !important;
}

/* ── Main area background ── */
.stApp { background-color: #0a0e14 !important; }

/* ── Header / toolbar / top decoration ── */
[data-testid="stHeader"] {
    background-color: #0a0e14 !important;
    border-bottom: 1px solid #1a2430 !important;
}
[data-testid="stToolbar"] { background-color: #0a0e14 !important; }
[data-testid="stDecoration"] { background-color: #0a0e14 !important; display: none; }
.stAppHeader, header[data-testid="stHeader"] { background: #0a0e14 !important; }

/* ── Download buttons: sidebar (sample CSV) ── */
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
    background-color: #0d1a28 !important;
    color: #a8c8d8 !important;
    border: 1px solid #1c3a4a !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 0.78rem !important;
    line-height: 1.4 !important;
    padding: 6px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button:hover {
    background-color: #112030 !important;
    border-color: #56b4d3 !important;
    color: #56b4d3 !important;
}

/* ── Download buttons: main area (export) ── */
.main [data-testid="stDownloadButton"] > button,
.main [data-testid="stDownloadButton"] button {
    background-color: #0d1a28 !important;
    color: #a8c8d8 !important;
    border: 1px solid #1c3a4a !important;
    border-radius: 8px !important;
}
.main [data-testid="stDownloadButton"] > button:hover,
.main [data-testid="stDownloadButton"] button:hover {
    background-color: #112030 !important;
    border-color: #56b4d3 !important;
    color: #56b4d3 !important;
}

/* ── File uploader ── */
section[data-testid="stFileUploader"],
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploader"] > section {
    background-color: #0d1520 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"] {
    background-color: #0d1520 !important;
    border: 1px dashed #1c3a4a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small { color: #6a9aaa !important; }
[data-testid="stFileUploadDropzone"] svg { fill: #3a6a7a !important; }
[data-testid="stFileUploaderFile"] {
    background-color: #0d1a28 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderFileName"] { color: #a8c8d8 !important; }

/* ── Inline code badges (end_date, last_purchase_date etc.) ── */
code, .stMarkdown code {
    background-color: #0d1f2d !important;
    color: #56b4d3 !important;
    border: 1px solid #1a3a4a !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.82em !important;
}

/* ── Browse files button ── */
[data-testid="stFileUploadDropzone"] button {
    background-color: #0d2030 !important;
    color: #56b4d3 !important;
    border: 1px solid #1c3a4a !important;
    border-radius: 6px !important;
}

/* ── Plotly chart top padding reset ── */
[data-testid="stPlotlyChart"] {
    margin-top: -12px !important;
}

/* ── section-title直後のPlotlyグラフ用 ── */
.section-title-tight {
    font-size: 0.68rem;
    font-weight: 600;
    color: #7ab4c4;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid #1a2a3a;
    padding-bottom: 8px;
    margin: 36px 0 0 0;
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

@st.cache_data(show_spinner=False)
def compute_km(duration_arr, event_arr):
    """Kaplan-Meier推定（配列を受け取りキャッシュ対応）"""
    S, records = 1.0, []
    for t in np.sort(np.unique(duration_arr)):
        n_risk   = (duration_arr >= t).sum()
        n_events = ((duration_arr == t) & (event_arr == 1)).sum()
        if n_risk > 0:
            S *= (1 - n_events / n_risk)
        records.append({'t': t, 'S': S, 'n_events': n_events, 'n_risk': n_risk})
    return pd.DataFrame(records)

@st.cache_data(show_spinner=False)
def fit_weibull(t_arr, S_arr):
    """Weibullフィッティング（配列を受け取りキャッシュ対応）"""
    mask = (t_arr > 0) & (S_arr > 0) & (S_arr < 1)
    t_f, S_f = t_arr[mask], S_arr[mask]
    ln_t        = np.log(t_f)
    ln_neg_ln_S = np.log(-np.log(S_f))
    valid = np.isfinite(ln_t) & np.isfinite(ln_neg_ln_S)
    ln_t, ln_neg_ln_S = ln_t[valid], ln_neg_ln_S[valid]
    if len(ln_t) < 3:
        return None, None, None, None
    k, b, r, *_ = stats.linregress(ln_t, ln_neg_ln_S)
    lam = np.exp(-b / k)
    fd = pd.DataFrame({'ln_t': ln_t, 'ln_neg_ln_S': ln_neg_ln_S})
    return k, lam, r**2, fd

def _compute_km_df(df):
    """DataFrameからKM計算（内部ヘルパー）"""
    return compute_km(df['duration'].values, df['event'].values)

def _fit_weibull_df(km_df):
    """KM DataFrameからWeibullフィット（内部ヘルパー）"""
    return fit_weibull(km_df['t'].values.astype(float), km_df['S'].values.astype(float))

@st.cache_data(show_spinner=False)
def load_and_preprocess_csv(file_bytes, dormancy_days, billing_cycle, business_type):
    """CSVの読み込みと前処理をキャッシュ化（同じファイル・設定なら再計算しない）"""
    import io as _io
    df_raw = pd.read_csv(_io.BytesIO(file_bytes))
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    col_map = {}
    for c in df_raw.columns:
        if any(x in c for x in ['id','customer','顧客']):       col_map.setdefault('customer_id', c)
        if any(x in c for x in ['start','開始','契約']):        col_map.setdefault('start_date', c)
        if any(x in c for x in ['end','解約','終了','cancel']):  col_map.setdefault('end_date', c)
        if any(x in c for x in ['revenue','売上','arpu','rev','price','amount']):
            col_map.setdefault('revenue', c)

    missing = [k for k in ['start_date','end_date','revenue'] if k not in col_map]
    if missing:
        return None, missing, None, None, None, None

    for c in df_raw.columns:
        if any(x in c for x in ['last','最終','purchase','購買','購入']):
            col_map.setdefault('last_purchase_date', c)

    df = df_raw.rename(columns={v: k for k, v in col_map.items()})
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date']   = pd.to_datetime(df['end_date'],   errors='coerce')
    if 'last_purchase_date' in df.columns:
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
    else:
        df['last_purchase_date'] = pd.NaT

    n_input = len(df)
    bad_dates = df['start_date'].isna().sum()
    df = df.dropna(subset=['start_date'])

    today = pd.Timestamp.today()
    dates_for_today = [df['start_date'].max()]
    if df['last_purchase_date'].notna().any():
        dates_for_today.append(df['last_purchase_date'].max())
    if df['end_date'].notna().any():
        dates_for_today.append(df['end_date'].max())
    ref_date = max(d for d in dates_for_today if pd.notna(d))
    if ref_date <= today:
        today = ref_date
    n_dormant = 0
    if dormancy_days is not None and df['last_purchase_date'].notna().any():
        dormant_mask = (
            df['end_date'].isna() &
            df['last_purchase_date'].notna() &
            ((today - df['last_purchase_date']).dt.days > dormancy_days)
        )
        n_dormant = dormant_mask.sum()
        # 離脱日 = last_purchase_date + dormancy_days（休眠判定日が実質の離脱日）
        df.loc[dormant_mask, 'end_date'] = (
            df.loc[dormant_mask, 'last_purchase_date'] + pd.Timedelta(days=dormancy_days)
        )

    def get_end(row):
        if pd.notna(row['end_date']):
            return row['end_date'], 1
        if dormancy_days is not None and pd.notna(row['last_purchase_date']):
            if (today - row['last_purchase_date']).days > dormancy_days:
                return row['last_purchase_date'] + pd.Timedelta(days=dormancy_days), 1
        return today, 0

    result = df.apply(get_end, axis=1, result_type='expand')
    df['end_resolved'] = result[0]
    df['event']        = result[1]
    df['duration']     = (df['end_resolved'] - df['start_date']).dt.days

    # サブスクの最低契約期間を保証
    if business_type != '都度購入型':
        if billing_cycle == 'カレンダーベース（月またぎ）← 月額サブスク推奨':
            min_dur = 30
        elif billing_cycle == '30日固定 ← 30日プラン':
            min_dur = 30
        elif billing_cycle == '365日固定 ← 年額サブスク':
            min_dur = 365
        elif '日数固定' in billing_cycle:
            try:
                min_dur = int(billing_cycle.split('日数固定')[0].strip().split()[-1])
            except Exception:
                min_dur = 30
        else:
            min_dur = 30
        df['duration'] = df['duration'].clip(lower=min_dur)

    n_corrected = (df['duration'] == 0).sum()
    df['duration'] = df['duration'].clip(lower=1)
    n_excluded = (df['duration'] < 0).sum()
    df = df[df['duration'] > 0]

    df['revenue_total'] = pd.to_numeric(df['revenue'], errors='coerce')

    meta = {
        'n_input': n_input,
        'n_dormant': n_dormant,
        'n_corrected': n_corrected,
        'n_excluded': n_excluded,
        'bad_dates': bad_dates,
    }
    return df, None, meta, billing_cycle, business_type, col_map
    surv_int = lam * gamma(1 + 1/k)
    return surv_int * arpu, surv_int

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
    st.markdown("### データ入力")

    # ══════════════════════════════════════════════════════
    # サンプルデータ生成
    # サブスク：フィットネスジム（月額7,000〜12,000円、k>1・逓増離脱型）
    # 都度購入：ファストファッションEC（1回4,000〜15,000円、購入間隔90日）
    # ══════════════════════════════════════════════════════
    np.random.seed(42)
    n_sample  = 10000
    BASE_DATE = pd.Timestamp('2025-12-31')  # 基準日固定
    OBS_START = pd.Timestamp('2021-01-01')  # 観測期間開始（5年間）
    today_ts  = BASE_DATE

    # start_datesを観測期間内（2023-01-01〜2025-12-31）で均等に生成
    # ※単発顧客の観測完結を保証するため全期間で均等生成
    _all_dates  = pd.date_range(OBS_START, BASE_DATE, periods=n_sample)
    _all_dates  = list(_all_dates)
    np.random.shuffle(_all_dates)
    start_dates = _all_dates[:n_sample]
    # 単発顧客生成カットオフ（基準日-180日）：これ以前のstart_dateなら観測完結保証
    _single_cutoff = BASE_DATE - pd.Timedelta(days=180)

    # ── サブスク：SaaS（月額）──────────────────────────
    # k=0.921（初期離脱型）、λ=273日
    # 6ヶ月：50%生存、1年：28%、2年：8%、3年：3%
    # 月額プラン：5,000 / 9,800 / 19,800円
    # LTV∞イメージ：約8万円、99%到達6年
    # サブスクは3年観測期間で独自のstart_datesを使用
    SUB_START = pd.Timestamp('2023-01-01')
    _sub_dates = list(pd.date_range(SUB_START, BASE_DATE, periods=n_sample))
    np.random.shuffle(_sub_dates)
    start_dates_sub = _sub_dates[:n_sample]

    ec_plans    = np.random.choice([5000, 9800, 19800], n_sample, p=[0.50, 0.35, 0.15])
    ec_survival = np.random.weibull(0.921, n_sample) * 273
    ec_churned  = np.random.random(n_sample) < 0.85

    end_dates_sub = []
    revenues_sub  = []
    for i in range(n_sample):
        sd  = start_dates_sub[i]
        fee = ec_plans[i]
        if ec_churned[i]:
            ed = sd + pd.Timedelta(days=max(1, int(ec_survival[i])))
            if ed <= BASE_DATE:
                end_dates_sub.append(ed.strftime('%Y-%m-%d'))
                months = max(1, int((ed - sd).days // 30) + 1)
            else:
                end_dates_sub.append('')
                months = max(1, int((BASE_DATE - sd).days // 30) + 1)
        else:
            end_dates_sub.append('')
            months = max(1, int((BASE_DATE - sd).days // 30) + 1)
        revenues_sub.append(fee * months)

    ec_plan_label = np.where(ec_plans == 7000,  'レギュラー（¥7,000）',
                    np.where(ec_plans == 9800,  'プレミアム（¥9,800）', 'パーソナル（¥12,000）'))
    channels_sub  = np.random.choice(['SNS広告', '検索広告', '紹介', 'オーガニック'], n_sample, p=[0.35, 0.30, 0.15, 0.20])
    ages_sub      = np.random.choice(['10代', '20代', '30代', '40代', '50代以上'], n_sample, p=[0.05, 0.25, 0.35, 0.25, 0.10])
    regions_sub   = np.random.choice(['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄'],
                        n_sample, p=[0.05, 0.07, 0.35, 0.15, 0.18, 0.07, 0.04, 0.09])
    prefs = ['東京','神奈川','大阪','愛知','埼玉','千葉','福岡','北海道','兵庫','静岡',
             '茨城','広島','京都','宮城','新潟','長野','栃木','岐阜','群馬','岡山',
             '三重','熊本','鹿児島','山口','愛媛','長崎','奈良','青森','岩手','大分',
             '石川','山形','富山','秋田','香川','和歌山','佐賀','福井','徳島','高知',
             '島根','宮崎','鳥取','沖縄','滋賀','山梨','福島']
    prefs_sub = np.random.choice(prefs, n_sample)

    sample_sub = pd.DataFrame({
        'customer_id':   [f'GY{i:05d}' for i in range(1, n_sample+1)],
        'start_date':    [d.strftime('%Y-%m-%d') for d in start_dates_sub],
        'end_date':      end_dates_sub,
        'revenue_total': revenues_sub,
        'plan':          ec_plan_label,
        'channel':       channels_sub,
        'age_group':     ages_sub,
        'region':        regions_sub,
        'prefecture':    prefs_sub,
    })

    # ── 都度購入：ファストファッションEC ────────────────
    # k≈0.7（初期離脱型）、購入間隔90日、休眠判定180日推奨
    # 単発65%、リピート35%（うちアクティブ15%・離脱85%）
    np.random.seed(43)
    ff_unit   = np.random.choice([8000, 15000, 25000, 30000], n_sample, p=[0.35, 0.35, 0.20, 0.10])
    ff_surv   = np.random.weibull(0.75, n_sample) * 300   # k<1：初期離脱型
    ff_single = np.random.random(n_sample) < 0.721         # 単発65%相当（カットオフ補正済）
    ff_active = np.random.random(n_sample) < 0.15          # リピートのうちアクティブ15%

    last_purchase_dates = []
    revenues_spot       = []
    for i in range(n_sample):
        sd    = start_dates[i]
        price = ff_unit[i]
        if ff_single[i] and sd <= _single_cutoff:
            # 単発：first=last、売上=単価1回
            lp        = sd
            purchases = 1
        elif ff_active[i] or (ff_single[i] and sd > _single_cutoff):
            # アクティブ：基準日から180日以内に購入あり
            days_since = np.random.randint(1, 180)
            lp         = BASE_DATE - pd.Timedelta(days=int(days_since))
            lp         = max(lp, sd + pd.Timedelta(days=1))
            purchases  = min(max(2, round((lp - sd).days / 90)), 15)  # 上限15回
        else:
            # 離脱リピート：Weibull生存期間で自然に離脱
            surv_days = max(1, int(ff_surv[i]))
            lp        = sd + pd.Timedelta(days=surv_days)
            lp        = min(lp, BASE_DATE - pd.Timedelta(days=1))
            lp        = max(lp, sd + pd.Timedelta(days=1))
            purchases = min(max(2, round((lp - sd).days / 90)), 15)  # 上限15回
        last_purchase_dates.append(lp.strftime('%Y-%m-%d'))
        revenues_spot.append(price * purchases)

    ff_gender   = np.random.choice(['男性', '女性', '未回答'],
                      n_sample, p=[0.38, 0.55, 0.07])
    ff_channels = np.random.choice(['Instagram広告', '検索広告', 'アプリ通知', 'メルマガ', '口コミ'],
                      n_sample, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    ff_ages     = np.random.choice(['10代', '20代', '30代', '40代', '50代以上'],
                      n_sample, p=[0.15, 0.40, 0.28, 0.12, 0.05])
    ff_regions  = np.random.choice(['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄'],
                      n_sample, p=[0.05, 0.07, 0.35, 0.15, 0.18, 0.07, 0.04, 0.09])
    prefs_ff    = np.random.choice(prefs, n_sample)

    sample_spot = pd.DataFrame({
        'customer_id':        [f'FF{i:05d}' for i in range(1, n_sample+1)],
        'start_date':         [d.strftime('%Y-%m-%d') for d in start_dates],
        'end_date':           '',
        'last_purchase_date': last_purchase_dates,
        'revenue_total':      revenues_spot,
        'gender':             ff_gender,
        'channel':            ff_channels,
        'age_group':          ff_ages,
        'region':             ff_regions,
        'prefecture':         prefs_ff,
    })

    # ── サブスク：ジム（日割りON版）────────────────────
    # 既存データの累計売上を日割り計算に変換
    revenues_sub_on = []
    for i in range(n_sample):
        sd  = start_dates_sub[i]
        fee = ec_plans[i]
        ed_str = end_dates_sub[i]
        if ed_str:
            ed = pd.Timestamp(ed_str)
            days = max(1, (ed - sd).days)
        else:
            days = max(1, (BASE_DATE - sd).days)
        revenues_sub_on.append(round(fee * days / 30, 0))  # 日割り：月額×日数/30

    sample_sub_on = pd.DataFrame({
        'customer_id':   [f'GY{i:05d}' for i in range(1, n_sample+1)],
        'start_date':    [d.strftime('%Y-%m-%d') for d in start_dates_sub],
        'end_date':      end_dates_sub,
        'revenue_total': revenues_sub_on,
        'plan':          ec_plan_label,
        'channel':       channels_sub,
        'age_group':     ages_sub,
        'region':        regions_sub,
        'prefecture':    prefs_sub,
    })

    # ── 都度購入：サプリ・健康食品EC ──────────────────
    # k>1（逓増離脱型）、購入間隔45日、休眠判定180日推奨
    # 単発45%：初回購入後リピートなし、リピート55%：定期的に購入
    np.random.seed(99)
    sp_unit   = np.random.choice([3000, 5000, 8000, 12000], n_sample, p=[0.25, 0.40, 0.25, 0.10])
    sp_surv   = np.random.weibull(1.3, n_sample) * 180   # k>1：逓増離脱型
    sp_single = np.random.random(n_sample) < 0.499        # 単発45%相当（カットオフ補正済）
    sp_active = np.random.random(n_sample) < 0.20         # リピートのうちアクティブ20%

    sp_last_purchase = []
    sp_revenues      = []
    for i in range(n_sample):
        sd    = start_dates[i]
        price = sp_unit[i]
        if sp_single[i] and sd <= _single_cutoff:
            # 単発：first=last、売上=単価1回（観測完結保証）
            lp        = sd
            purchases = 1
        elif sp_active[i] or (sp_single[i] and sd > _single_cutoff):
            # アクティブリピート：基準日から180日以内に購入あり
            days_since = np.random.randint(1, 180)
            lp         = BASE_DATE - pd.Timedelta(days=int(days_since))
            lp         = max(lp, sd + pd.Timedelta(days=1))
            purchases  = min(max(2, round((lp - sd).days / 45)), 20)  # 上限20回
        else:
            # 離脱リピート：生存期間に従い自然に離脱
            surv_days = max(1, int(sp_surv[i]))
            lp        = sd + pd.Timedelta(days=surv_days)
            lp        = min(lp, BASE_DATE - pd.Timedelta(days=1))
            lp        = max(lp, sd + pd.Timedelta(days=1))
            purchases = min(max(2, round((lp - sd).days / 45)), 20)  # 上限20回
        sp_last_purchase.append(lp.strftime('%Y-%m-%d'))
        sp_revenues.append(price * purchases)

    sp_gender   = np.random.choice(['男性', '女性', '未回答'], n_sample, p=[0.35, 0.58, 0.07])
    sp_channels = np.random.choice(['SNS広告', '検索広告', 'アプリ通知', 'メルマガ', '口コミ'],
                      n_sample, p=[0.30, 0.25, 0.15, 0.20, 0.10])
    sp_ages     = np.random.choice(['20代', '30代', '40代', '50代以上', '60代以上'],
                      n_sample, p=[0.15, 0.30, 0.30, 0.18, 0.07])
    sp_regions  = np.random.choice(['北海道', '東北', '関東', '中部', '近畿', '中国', '四国', '九州・沖縄'],
                      n_sample, p=[0.05, 0.07, 0.35, 0.15, 0.18, 0.07, 0.04, 0.09])
    prefs_sp    = np.random.choice(prefs, n_sample)

    sample_supp = pd.DataFrame({
        'customer_id':        [f'SP{i:05d}' for i in range(1, n_sample+1)],
        'start_date':         [d.strftime('%Y-%m-%d') for d in start_dates],
        'end_date':           '',
        'last_purchase_date': sp_last_purchase,
        'revenue_total':      sp_revenues,
        'gender':             sp_gender,
        'channel':            sp_channels,
        'age_group':          sp_ages,
        'region':             sp_regions,
        'prefecture':         prefs_sp,
    })

    import base64
    sub_csv     = sample_sub.to_csv(index=False).encode('utf-8-sig')
    sub_on_csv  = sample_sub_on.to_csv(index=False).encode('utf-8-sig')
    spot_csv    = sample_spot.to_csv(index=False).encode('utf-8-sig')
    supp_csv    = sample_supp.to_csv(index=False).encode('utf-8-sig')
    sub_b64     = base64.b64encode(sub_csv).decode()
    sub_on_b64  = base64.b64encode(sub_on_csv).decode()
    spot_b64    = base64.b64encode(spot_csv).decode()
    supp_b64    = base64.b64encode(supp_csv).decode()

    st.markdown("<span style='color:#c8d0d8; font-size:0.78rem;'>サンプルデータを選択してお試しください。</span>", unsafe_allow_html=True)

    # サンプルデータをセッションステートで管理
    if 'sample_df' not in st.session_state:
        st.session_state.sample_df = None
    if 'sample_label' not in st.session_state:
        st.session_state.sample_label = None

    _sample_options = {
        'サブスク型：月額ジム（日割りOFF）': ('sub', sample_sub),
        'サブスク型：月額ジム（日割りON）':  ('sub_on', sample_sub_on),
        '都度購入型：ファッションEC':        ('spot', sample_spot),
        '都度購入型：サプリEC':              ('supp', sample_supp),
    }
    _btn_s = "display:block; width:100%; text-align:center; text-decoration:none; background:#0d1a28; color:#a8c8d8; border:1px solid #1c3a4a; border-radius:8px; padding:8px 6px; font-size:0.75rem; line-height:1.5; box-sizing:border-box;"
    _selected_sample = st.selectbox(
        'サンプルデータを選択',
        ['（選択してください）'] + list(_sample_options.keys()),
        key='sample_select',
        label_visibility='collapsed'
    )
    if _selected_sample != '（選択してください）' and st.session_state.get('_prev_sample') != _selected_sample:
        st.session_state._prev_sample = _selected_sample
        _key, _df = _sample_options[_selected_sample]
        st.session_state.sample_df = _df
        st.session_state.sample_label = _selected_sample
        # サンプルに応じてデフォルト設定をセッションに保存
        if 'sub' in _key:
            st.session_state['_sample_biz']     = 'サブスク・継続課金型'
            st.session_state['_sample_prorate'] = (_key == 'sub_on')
            st.session_state['_sample_seg']     = 'plan, channel, age_group, region'
        else:
            st.session_state['_sample_biz']     = '都度購入型'
            st.session_state['_sample_prorate'] = False
            st.session_state['_sample_seg']     = 'gender, channel, age_group, region' 
        st.rerun()


    uploaded = st.file_uploader("CSVをアップロード", type=['csv'])

    # サンプルボタン or アップロードでデータを確定
    if uploaded is not None:
        st.session_state.sample_df = None  # アップロード優先
        st.session_state.sample_label = None

    st.markdown("### 異常値処理")
    iqr_multiplier = st.select_slider(
        "外れ値カット強度",
        options=[0.0, 3.0, 2.5, 2.0, 1.5],
        value=0.0,
        format_func=lambda x: "除外なし" if x == 0.0 else {
            3.0: "上位約0.2〜1%除外",
            2.5: "上位約0.5〜2%除外",
            2.0: "上位約1〜3%除外",
            1.5: "上位約3〜5%除外",
        }[x]
    )
    outlier_removal = iqr_multiplier > 0.0
    st.caption(
        "累計金額の上位外れ値をIQR（四分位範囲）の倍率で除外します。"
        "累計金額の下位1%（¥0・極端な低額）は常時除外されます。"
    )

    st.markdown("### ビジネスタイプ")
    _biz_options = ["サブスク・継続課金型", "都度購入型"]
    _biz_default = st.session_state.get('_sample_biz', 'サブスク・継続課金型')
    _biz_idx = _biz_options.index(_biz_default) if _biz_default in _biz_options else 0
    business_type = st.radio(
        "ビジネスタイプ",
        _biz_options,
        index=_biz_idx,
    )

    if business_type == "サブスク・継続課金型":
        st.caption(
            "解約日（end_date）をベースに離脱を判定します。"
            "end_dateが空欄の顧客は継続中として扱われます。"
        )
        dormancy_days = None  # 休眠判定なし
        billing_cycle_display = st.radio(
            "契約期間",
            [
                "月額（カレンダーベース）",
                "年額（365日固定）",
                "カスタム入力（日数固定）",
            ],
            index=0,
        )
        _billing_map = {
            "月額（カレンダーベース）": "カレンダーベース（月またぎ）← 月額サブスク推奨",
            "年額（365日固定）": "365日固定 ← 年額サブスク",
            "カスタム入力（日数固定）": "カスタム入力（日数固定）",
        }
        billing_cycle = _billing_map[billing_cycle_display]

        if billing_cycle_display == "カスタム入力（日数固定）":
            custom_cycle_days = st.number_input("契約日数", min_value=1, max_value=365, value=30)
        else:
            custom_cycle_days = None
        st.caption("月額：毎月同じ日に更新（例：5/15契約 → 6/15・7/15…）。年額：365日固定。カスタム：隔月・四半期など任意の日数。")

        st.markdown("<div style='font-size:0.82rem; color:#c8d0d8; margin-bottom:4px;'>解約時の日割り計算あり</div>", unsafe_allow_html=True)
        prorate_cancel = st.toggle("解約時の日割り計算あり", value=st.session_state.get("_sample_prorate", False), label_visibility="collapsed")
        st.caption("OFFの場合、解約日を契約更新日に丸めます（一般的なサブスク）。ONの場合、実際の解約日をそのまま使用します。")

    else:  # 都度購入型
        st.caption(
            "最終購買日（last_purchase_date）をベースに休眠判定します。"
            "CSVに last_purchase_date 列が必要です。"
        )
        billing_cycle = "日次（都度購入）"
        custom_cycle_days = None
        prorate_cancel = False
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

    st.markdown("### Gross Profit Margin (%)")
    # サンプルCSVのファイル名からデフォルトGPMを設定
    _gpm_default = 50
    _sample_label = st.session_state.get('sample_label', '') or ''
    _fn = (uploaded.name.lower() if uploaded is not None else '') + _sample_label.lower()
    if uploaded is not None or _sample_label:
        if 'saas' in _fn or 'subscription' in _fn:
            _gpm_default = 75
        elif 'gym' in _fn:
            _gpm_default = 50
        elif 'supplement' in _fn or 'supp' in _fn:
            _gpm_default = 60
        elif 'fec' in _fn or 'fashion' in _fn or 'spot' in _fn:
            _gpm_default = 40
    gpm = st.slider("粗利率：売上に占める（売上－変動費）の割合", 0, 100, _gpm_default, 1) / 100
    st.caption(f"LTV∞の表示は売上ベース。CAC上限の算出には粗利ベース（売上×{gpm:.0%}）を使用します。")

    st.markdown("### CAC 上限")
    cac_n = st.slider("N（LTV:CAC = N:1）", 1.0, 10.0, 3.0, 0.5)
    cac_label = f"LTV:CAC = {cac_n}:1"
    cac_mode = 'LTV : CAC = N : 1'
    cac_recover_days = None
    st.caption(f"例：LTV:CAC = 3:1 の場合、CAC上限 = LTV（粗利）÷ 3")

    st.markdown("### セグメント分析")
    segment_cols_input = st.text_input(
        "セグメント列名（カンマ区切りで複数指定可）",
        value=st.session_state.get('_sample_seg', ''),
        placeholder="例：plan, channel, age_group（最大5列）",
    )
    st.caption(
        "CSVの列名をカンマ区切りで入力してください。"
        "セグメント別のLTV∞を自動比較し、優先獲得セグメントを特定します。\n"
        "1列あたり最大50種類・最大5列。代表的な軸：プラン・チャネル・年齢層・性別・地域など。"
    )
    st.markdown("### 表示件数")
    seg_display_limit = st.slider(
        "詳細表示（暫定LTV・生存曲線）の上位N件",
        min_value=1, max_value=20, value=5,
    )
    st.caption(
        "セグメント（例：都道府県）の項目数（例：47）が多いほどブラウザの描画に時間がかかります。"
        "表示する上位N項目を絞ることで速度が大幅に改善されます。\n"
        "エクスポートされる各ファイルには全項目出力されます。\n"
        "まず上位5項目で傾向を確認し、必要に応じて増やすことをお勧めします。"
    )

    cac_input = 0
    cac_known = False

    st.markdown("")
    st.markdown("### レポート情報")
    client_name  = st.text_input("クライアント名", "", placeholder="会社・ブランド・商品/サービスなど")
    analyst_name = st.text_input("作成者", "", placeholder="氏名・チーム・部署・組織など")

# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 16px 0 32px 0; border-bottom: 1px solid #1a2a3a; margin-bottom: 28px;'>
  <div style='font-family: 'BIZ UDPGothic', sans-serif; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: #3a6a7a; margin-bottom: 8px;'>Analytics Tool</div>
  <div style='font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: #c8d0d8; letter-spacing: -0.03em; line-height: 1;'>LTV Analyzer <span style='color: #56b4d3;'>Advanced</span></div>
  <div style='font-size: 0.78rem; color: #3a5a6a; margin-top: 8px; letter-spacing: 0.02em;'>Kaplan–Meier × Weibull — Segment-level LTV Intelligence &nbsp;·&nbsp; v198</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# No file → instructions
# ══════════════════════════════════════════════════════════════

if uploaded is None and st.session_state.get('sample_df') is None:
    st.info("サイドバーからCSVをアップロードするか、サンプルデータを選択してください。")

    st.markdown("<div class='section-title'>CSV フォーマット</div>", unsafe_allow_html=True)
    st.markdown("""
| 列名 | 内容 | 形式 | 例 |
|------|------|------|----|
| `customer_id` | 顧客ID | 任意の文字列 | C0001 |
| `start_date` | 契約開始日 / 初回購入日 | YYYY-MM-DD | 2023-01-01 |
| `end_date` | 解約日（サブスク向け・継続中は**空欄**） | YYYY-MM-DD | 2024-03-15 |
| `last_purchase_date` | 最終購買日（都度購入向け・任意） | YYYY-MM-DD | 2024-06-01 |
| `revenue` | **累計売上**（円） | 数値 | 48000 |
| `セグメント列`（任意の列名） | **Advanced機能**：プラン・チャネル・年齢層など | 文字列 | 月額300 |

> **Advanced版では必ずセグメント列を追加してください。**複数列追加可能です。\n
> 列名は完全一致でなくてもOKです。`start`・`end`・`last`・`revenue`を含む列名は自動認識します。\n
> ARPU daily はビジネスタイプに応じて自動計算されます。\n
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

# query_paramsからサンプル選択を処理
_qp = st.query_params
if 'sample' in _qp and st.session_state.get('sample_df') is None:
    _s = _qp['sample']
    if _s == 'sub':
        st.session_state.sample_df    = sample_sub
        st.session_state.sample_label = 'サブスク型：月額ジム（日割りOFF）'
    elif _s == 'sub_on':
        st.session_state.sample_df    = sample_sub_on
        st.session_state.sample_label = 'サブスク型：月額ジム（日割りON）'
    elif _s == 'spot':
        st.session_state.sample_df    = sample_spot
        st.session_state.sample_label = '都度購入型：ファッションEC'
    elif _s == 'supp':
        st.session_state.sample_df    = sample_supp
        st.session_state.sample_label = '都度購入型：サプリEC'
    st.query_params.clear()

# ══════════════════════════════════════════════════════════════
# Load & validate data
# ══════════════════════════════════════════════════════════════

# デフォルト値（tryブロックが途中終了した場合のフォールバック）
arpu_daily = None
gp_daily   = None

# sample_df優先、なければuploadedから読み込み
_active_df = st.session_state.get('sample_df', None)
if _active_df is not None:
    pass
try:
    if _active_df is not None:
        df_raw = _active_df.copy()
        df_raw.columns = df_raw.columns.str.strip().str.lower()
    else:
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
        st.error(f" 列が見つかりません: {missing}\n\n列名に `start`・`end`・`revenue` を含む列が必要です。サイドバーからサンプルCSVをダウンロードして形式を確認してください。")
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
        st.warning(f" {bad_dates}行で `start_date` が読み取れませんでした。該当行は除外します。")
    df = df.dropna(subset=['start_date'])

    today = pd.Timestamp.today()
    n_input = len(df)

    # 分析基準日：CSVの最新観測日（last_purchase_dateまたはend_dateの最大値）
    # → 同じCSVなら何日に分析しても結果が変わらない（再現性の確保）
    dates_for_today = [df['start_date'].max()]
    if df['last_purchase_date'].notna().any():
        dates_for_today.append(df['last_purchase_date'].max())
    if df['end_date'].notna().any():
        dates_for_today.append(df['end_date'].max())
    ref_date = max(d for d in dates_for_today if pd.notna(d))
    if ref_date <= today:
        today = ref_date

    # 休眠判定：end_date がなく last_purchase_date が休眠期間を超えている場合は離脱とみなす
    n_dormant = 0
    if dormancy_days is not None and df['last_purchase_date'].notna().any():
        dormant_mask = (
            df['end_date'].isna() &
            df['last_purchase_date'].notna() &
            ((today - df['last_purchase_date']).dt.days > dormancy_days)
        )
        n_dormant = dormant_mask.sum()
        # 離脱日 = last_purchase_date + dormancy_days（休眠判定日が実質の離脱日）
        df.loc[dormant_mask, 'end_date'] = (
            df.loc[dormant_mask, 'last_purchase_date'] + pd.Timedelta(days=dormancy_days)
        )

    # ── duration・event の確定 ──
    def get_end(row):
        if pd.notna(row['end_date']):
            return row['end_date'], 1  # 解約済み
        if dormancy_days is not None and pd.notna(row['last_purchase_date']):
            days_since = (today - row['last_purchase_date']).days
            if days_since > dormancy_days:
                # 離脱日 = last_purchase_date + dormancy_days
                return row['last_purchase_date'] + pd.Timedelta(days=dormancy_days), 1
        return today, 0  # 継続中

    result = df.apply(get_end, axis=1, result_type='expand')
    df['end_resolved'] = result[0]
    df['event']        = result[1]

    if business_type == '都度購入型' and 'last_purchase_date' in df.columns:
        # 都度購入型：duration = last_purchase_date - start_date（dormancy_days除外）
        # アクティブ顧客（event=0）：基準日 - start_date
        # 休眠離脱顧客（event=1）：last_purchase_date - start_date + dormancy_days
        def calc_duration_spot(row):
            if row['event'] == 0:
                return (today - row['start_date']).days
            else:
                if pd.notna(row['last_purchase_date']):
                    return (row['last_purchase_date'] - row['start_date']).days + (dormancy_days or 0)
                else:
                    return (row['end_resolved'] - row['start_date']).days
        df['duration'] = df.apply(calc_duration_spot, axis=1)
    else:
        df['duration'] = (df['end_resolved'] - df['start_date']).dt.days

    # 丸め前のdurationを保存（single_churn_rate計算用）
    df['duration_raw'] = df['duration']

    # サブスクの最低契約期間を保証（契約期間未満のdurationを引き上げる）
    if business_type != '都度購入型':
        if '365日固定' in billing_cycle:
            min_contract = 365
        elif custom_cycle_days and 'カスタム' in billing_cycle:
            min_contract = custom_cycle_days
        else:
            min_contract = 30

        if not prorate_cancel:
            # 日割りなし：durationを契約更新日に丸める
            if billing_cycle_display == '月額（カレンダーベース）':
                # start_dateから月単位で次の更新日を計算
                import calendar as _cal
                def round_to_renewal(row):
                    sd = row['start_date']
                    dur = row['duration']
                    ed = sd + pd.Timedelta(days=dur)
                    months = 0
                    cur = sd
                    while True:
                        m = cur.month + 1 if cur.month < 12 else 1
                        y = cur.year if cur.month < 12 else cur.year + 1
                        max_d = _cal.monthrange(y, m)[1]
                        nxt = pd.Timestamp(y, m, min(sd.day, max_d))
                        if nxt > ed:
                            break
                        months += 1
                        cur = nxt
                    m2 = cur.month + 1 if cur.month < 12 else 1
                    y2 = cur.year if cur.month < 12 else cur.year + 1
                    max_d2 = _cal.monthrange(y2, m2)[1]
                    renewal = pd.Timestamp(y2, m2, min(sd.day, max_d2))
                    return max((renewal - sd).days, min_contract)
                df['duration'] = df.apply(round_to_renewal, axis=1)
            else:
                import numpy as _np
                df['duration'] = (_np.ceil(df['duration'] / min_contract) * min_contract).astype(int)

            # 最低契約期間未満で解約した顧客 → 打ち切り扱い（event=0）にしてdurationを引き上げ
            short_mask = df['duration'] < min_contract
            df.loc[short_mask, 'event'] = 0
            df.loc[short_mask, 'duration'] = min_contract

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

        if billing_cycle == "日次（都度購入）":
            # 都度購入：累計売上 ÷ 継続日数
            return rev / dur

        # 日割りONの場合：実際の日数で割る
        if prorate_cancel:
            return rev / max(dur, 1)

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

    # ── 異常値除外（オプション）──
    n_outlier = 0
    if outlier_removal:
        before = len(df)
        # 累計売上：下位1%（¥0・極端な低額）と上位IQR×倍率でカット
        lower_r = df['revenue_total'].quantile(0.01)
        q1_r = df['revenue_total'].quantile(0.25)
        q3_r = df['revenue_total'].quantile(0.75)
        upper_r = q3_r + iqr_multiplier * (q3_r - q1_r)
        df = df[(df['revenue_total'] >= lower_r) & (df['revenue_total'] <= upper_r)]
        n_outlier = before - len(df)

    # ARPU計算
    if billing_cycle == "日次（都度購入）":
        # ── 都度購入型：ARPU_short / ARPU_long / ARPU_0-dormancy の3段階計算 ──
        _dorm = dormancy_days or 180

        # 基準日（today）
        # 単発顧客：first == last かつ 基準日-last >= dormancy_days（観測完結）
        if 'last_purchase_date' in df.columns:
            _gap_first_last = (df['last_purchase_date'] - df['start_date']).dt.days.fillna(-1)
            _days_since_last = (today - df['last_purchase_date']).dt.days.fillna(0)

            # 単発顧客マスク（first=last かつ観測完結）
            _single_mask = (_gap_first_last == 0) & (_days_since_last >= _dorm)
            # リピート顧客マスク（first≠last、基準日-last問わず）
            _long_mask   = _gap_first_last > 0

            # ARPU_short：単発顧客の総売上 ÷ 総顧客数 ÷ dormancy_days
            _single_df = df[_single_mask]
            if len(_single_df) > 0:
                arpu_short = _single_df['revenue_total'].sum() / len(_single_df) / _dorm
            else:
                arpu_short = 0.0

            # ARPU_long：リピート顧客の総売上 ÷ 総duration（加重平均）
            _long_df = df[_long_mask]
            if len(_long_df) > 0:
                arpu_long = _long_df['revenue_total'].sum() / _long_df['duration'].sum()
            else:
                arpu_long = df['revenue_total'].sum() / df['duration'].sum()

            # 比率計算
            _n_single = len(_single_df)
            _n_long   = len(_long_df)
            _n_total  = _n_single + _n_long
            if _n_total > 0:
                _w_short = _n_single / _n_total
                _w_long  = _n_long  / _n_total
            else:
                _w_short, _w_long = 0.5, 0.5

            # ARPU_0-dormancy：単発とロングの加重平均
            arpu_0_dorm = arpu_short * _w_short + arpu_long * _w_long

            # 全体arpu_dailyはARPU_longを採用（t=180以降の積分用）
            arpu_daily  = arpu_long
        else:
            # last_purchase_dateがない場合はフォールバック
            arpu_short  = df['revenue_total'].sum() / df['duration'].sum()
            arpu_long   = arpu_short
            arpu_0_dorm = arpu_short
            arpu_daily  = arpu_short
            _dorm       = dormancy_days or 180

    elif prorate_cancel:
        # 日割りON：算術平均（revenue_totalが既に日割り計算済みのため）
        arpu_daily  = df['arpu_daily'].mean()
        arpu_short  = arpu_daily
        arpu_long   = arpu_daily
        arpu_0_dorm = arpu_daily
    else:
        # サブスク日割りOFF：billing_cycleで月額÷月日数に正規化済みなので算術平均が正確
        arpu_daily  = df['arpu_daily'].mean()
        arpu_short  = arpu_daily
        arpu_long   = arpu_daily
        arpu_0_dorm = arpu_daily
    gp_daily = arpu_daily * gpm

    # ビジネスタイプ依存ラベル（データ読み込みブロック内で使用）
    acq_label  = "初回購入" if business_type == "都度購入型" else "契約"
    date_label = "初回購入日" if business_type == "都度購入型" else "契約開始日"

    # ── 通知メッセージ ──
    if n_dormant > 0:
        st.info(f"{n_dormant:,}件を休眠顧客（最終購買から{dormancy_days}日超）として実質離脱に変換しました。")
    if n_corrected > 0:
        st.info(f"ℹ {n_corrected}件：{date_label}と終端日が同日のため1日に補正しました。")
    if n_excluded > 0:
        st.warning(f" {n_excluded}件：start_dateが未来の日付のため除外しました（入力ミスの可能性）。")
    if n_outlier > 0:
        st.info(f"{n_outlier:,}件を異常値として除外しました（累計金額のIQR×{iqr_multiplier}基準）。")
    if n_dormant == 0 and n_corrected == 0 and n_excluded == 0 and n_outlier == 0:
        st.success(f" 全{n_input:,}件のデータを正常に読み込みました。")

    if len(df) < 10:
        st.error(" 有効なデータが10件未満です。分析には最低10件の顧客データが必要です。")
        st.stop()

except Exception as e:
    st.error(f" データ読み込みエラー: {e}\n\nCSVの形式を確認してください。サンプルCSVをダウンロードして参照してください。")
    st.stop()

# ══════════════════════════════════════════════════════════════
# Run analysis
# ══════════════════════════════════════════════════════════════

# ── オフセット設定 ──
if business_type == "都度購入型":
    # 都度購入型：dormancy_daysをオフセットとして使用
    # t=0〜dormancy_daysはS(t)=1.0で確定なのでWeibullフィットから切り離す
    ltv_offset_days = dormancy_days or 180
elif prorate_cancel:
    ltv_offset_days = 0
elif billing_cycle_display == "月額（カレンダーベース）":
    ltv_offset_days = 30.44
elif billing_cycle_display == "年額（365日固定）":
    ltv_offset_days = 365
elif billing_cycle_display == "カスタム入力（日数固定）":
    ltv_offset_days = custom_cycle_days or 30
else:
    ltv_offset_days = 30.44

# オフセット適用：durationからオフセットを引いてフィッティング
km_df_raw = _compute_km_df(df)  # オフセット前のKM（グラフ・Excel表示用）
if ltv_offset_days > 0:
    df_fit = df.copy()
    df_fit['duration'] = df_fit['duration'] - ltv_offset_days
    # オフセット後にduration≤0になった顧客は打ち切り扱い（最低契約期間内）
    df_fit.loc[df_fit['duration'] <= 0, 'event'] = 0
    df_fit.loc[df_fit['duration'] <= 0, 'duration'] = 1
    df_fit['duration'] = df_fit['duration'].clip(lower=1)
    km_df = _compute_km_df(df_fit)
else:
    km_df = km_df_raw

k, lam, r2, fit_df = _fit_weibull_df(km_df)

if k is None:
    st.error(" Weibullフィッティングに失敗しました。解約済み顧客が少なすぎる可能性があります（最低10件の解約データが必要）。")
    st.stop()

# LTV計算（オフセット分を加算）
def ltv_inf_offset(k, lam, arpu, offset_days):
    surv_int = lam * gamma(1 + 1/k)
    return (surv_int + offset_days) * arpu, surv_int

def ltv_horizon_offset(k, lam, arpu, h, offset_days):
    h_adj = max(h - offset_days, 0)
    if h_adj == 0:
        return offset_days * arpu
    x = (h_adj / lam) ** k
    return (lam * gamma(1 + 1/k) * gammainc(1 + 1/k, x) + offset_days) * arpu

def ltv_horizon_spot(k, lam, arpu_0d, arpu_long, h, dorm):
    """都度購入型専用LTVホライズン計算
    h: 総ホライズン（初回購入からの日数）
    dorm: dormancy_days
    """
    h_short = min(h, dorm)
    h_long  = max(h - dorm, 0)
    ltv_s   = h_short * arpu_0d
    if h_long > 0:
        x     = (h_long / lam) ** k
        ltv_l = lam * gamma(1 + 1/k) * gammainc(1 + 1/k, x) * arpu_long
    else:
        ltv_l = 0
    return ltv_s + ltv_l

if business_type == "都度購入型":
    # LTV = LTV_short（固定部分）+ LTV_long（Weibull積分部分）
    _dorm_off = dormancy_days or 180
    _ltv_short_rev = _dorm_off * arpu_0_dorm   # t=0〜dormancy_daysの固定積分
    _ltv_long_rev, _surv_long = ltv_inf_offset(k, lam, arpu_long, 0)  # t=180以降
    ltv_rev  = _ltv_short_rev + _ltv_long_rev
    surv_int = _dorm_off + _surv_long
else:
    ltv_rev, surv_int = ltv_inf_offset(k, lam, arpu_daily, ltv_offset_days)  # 売上ベース
if business_type == "都度購入型":
    _gp_short  = _dorm_off * (arpu_0_dorm * gpm)
    _gp_long_v, _ = ltv_inf_offset(k, lam, arpu_long * gpm, 0)
    ltv_val = _gp_short + _gp_long_v
else:
    ltv_val, _ = ltv_inf_offset(k, lam, gp_daily, ltv_offset_days)  # 粗利ベース
cac_upper = ltv_val / cac_n

# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>分析結果サマリー</div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)

if k < 1.0:
    k_desc = "初期離脱大・投資回収が比較的長期"
else:
    k_desc = "継続後に解約増・投資回収が比較的短期"
metrics = [
    (f"¥{ltv_rev:,.0f}", "LTV∞",       "売上ベース"),
    (f"¥{cac_upper:,.0f}", f"CAC上限",  f"{cac_label}（粗利ベース）"),
    (f"{k:.3f}",           "Weibull k", f"{k_desc}"),
    (f"{lam + ltv_offset_days:.1f}日", "Weibull λ", "値が大きいほどLTV∞到達が長期化"),
    (f"{r2:.3f}",          "R²",        "0.9以上が理想 / 1.0が最高精度"),
]
for col, (val, title, desc) in zip([m1,m2,m3,m4,m5], metrics):
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label' style='font-size:0.82rem; color:#56b4d3; margin-top:6px; letter-spacing:0.05em;'>{title}</div>"
            f"<div class='metric-label' style='font-size:0.65rem; color:#444; margin-top:3px; letter-spacing:0.03em; line-height:1.4;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# R² warning
if r2 < 0.85:
    st.warning(f" R²={r2:.3f} — フィット精度がやや低めです。データ点数を増やすか、観測期間を見直してください。")

# ── サマリー解説ボックス ──────────────────────────────────────

# 単発離脱率の計算
if business_type == "都度購入型":
    # 初回購入後に1度も再購入せず離脱した顧客の割合
    churn_period = dormancy_days if dormancy_days else 365
    if 'last_purchase_date' in df.columns and df['last_purchase_date'].notna().any():
        # first=last かつ 観測完結（基準日-last >= dormancy_days）= 単発購入・離脱確定
        _gap = (df['last_purchase_date'] - df['start_date']).dt.days.fillna(-1)
        _days_since = (today - df['last_purchase_date']).dt.days.fillna(0)
        single_churn_rate = ((_gap == 0) & (_days_since >= churn_period)).sum() / len(df) * 100
    else:
        single_churn_rate = ((df['event'] == 1) & (df['duration'] <= churn_period)).sum() / len(df) * 100
    period_label = f"{churn_period}日"
else:
    # 最初の契約期間のみで解約した割合（実データから直接計算）
    if '365日固定' in billing_cycle:
        _min_c = 365
    elif custom_cycle_days and 'カスタム' in billing_cycle:
        _min_c = custom_cycle_days
    else:
        _min_c = 30
    churn_period = _min_c
    _dur_col = "duration_raw" if "duration_raw" in df.columns else "duration"
    single_churn_rate = ((df["event"] == 1) & (df[_dur_col] <= churn_period)).sum() / len(df) * 100
    period_label = f"{churn_period}日（1契約期間）"

if business_type == "都度購入型":
    if k < 1.0:
        k_summary = (
            f"k={k:.3f}の初期離脱型です。初回購入後{period_label}以内に再購入しなかった顧客（単発購入）は"
            f"{single_churn_rate:.0f}%です。"
            f"リピートした顧客の多くは初回購入からλ={lam+ltv_offset_days:.0f}日（約{(lam+ltv_offset_days)/365:.1f}年）以上購買を継続する傾向があります。"
            f"LTV∞は¥{ltv_rev:,.0f}でCAC上限は¥{cac_upper:,.0f}ですが、"
            f"投資回収は比較的長期になるため、暫定LTVテーブルで現実的な回収期間を確認してCACを設計してください。"
        )
    else:
        k_summary = (
            f"k={k:.3f}の逓増離脱型です。初回購入後{period_label}以内に再購入しなかった顧客（単発購入）は"
            f"{single_churn_rate:.0f}%です。"
            f"リピートした顧客の多くは初回購入からλ={lam+ltv_offset_days:.0f}日（約{(lam+ltv_offset_days)/365:.1f}年）以上購買を継続する傾向があります。"
            f"LTV∞は¥{ltv_rev:,.0f}でCAC上限は¥{cac_upper:,.0f}、比較的短期での投資回収が見込めます。"
        )
else:  # サブスク
    if k < 1.0:
        k_summary = (
            f"k={k:.3f}の初期離脱型です。最初の契約期間のみで解約した顧客は"
            f"{single_churn_rate:.0f}%です。"
            f"初期を乗り越えた顧客の多くはλ={lam+ltv_offset_days:.0f}日（約{(lam+ltv_offset_days)/365:.1f}年）以上継続する傾向があります。"
            f"LTV∞は¥{ltv_rev:,.0f}でCAC上限は¥{cac_upper:,.0f}ですが、"
            f"投資回収は比較的長期になるため、暫定LTVテーブルで現実的な回収期間を確認してCACを設計してください。"
        )
    else:
        k_summary = (
            f"k={k:.3f}の逓増離脱型です。最初の契約期間のみで解約した顧客は"
            f"{single_churn_rate:.0f}%です。"
            f"初期を乗り越えた顧客の多くはλ={lam+ltv_offset_days:.0f}日（約{(lam+ltv_offset_days)/365:.1f}年）以上継続する傾向があります。"
            f"LTV∞は¥{ltv_rev:,.0f}でCAC上限は¥{cac_upper:,.0f}、比較的短期での投資回収が見込めます。"
        )

if r2 >= 0.95:
    r2_summary = f"R²={r2:.3f}はモデル精度が非常に高く、この推定値は意思決定に十分活用できます。"
elif r2 >= 0.85:
    r2_summary = f"R²={r2:.3f}は許容範囲内の精度です。推定値に±15%程度の幅を見込んでください。"
else:
    r2_summary = f"R²={r2:.3f}はやや低めです。推定値の信頼性に注意してください。"

summary_text = f"{k_summary}{r2_summary}"

st.markdown(f"""
<div style='margin: 16px 0 4px 0; line-height: 1.9;'>
    <div style='
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #56b4d3;
        border-bottom: 1px solid #56b4d3;
        padding-bottom: 4px;
        margin-bottom: 10px;
        display: inline-block;
    '>結論</div>
    <div style='font-size: 0.95rem; color: #a8dadc; letter-spacing: 0.01em;'>{summary_text}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>分析モデルの信頼性</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

# ── グラフ描画用データ準備 ──
if business_type == "都度購入型":
    # 都度購入型：
    # durationには既にdormancy_daysが含まれているので、
    # KM・Weibullともにオフセット加算不要
    # Weibullはオフセット後のkm_dfでフィットしたk/lamを使用
    # t_smooth（オフセット後の時間軸）でS(t)を計算し、そのまま描画
    km_t_plot     = np.concatenate([[0], km_df_raw['t'].values])
    km_s_plot     = np.concatenate([[1.0], km_df_raw['S'].values])
    # Weibull：t=ltv_offset_daysから開始（それ以前は線なし）
    t_smooth_plot = np.linspace(ltv_offset_days, km_df_raw['t'].max() * 1.3, 600)
    S_wei_plot    = weibull_s(t_smooth_plot - ltv_offset_days, k, lam)
else:
    # サブスク：km_df（オフセット後）のt軸を+ltv_offset_daysで元スケールに戻す
    # これによりt=ltv_offset_days未満のデータが除外され、正しくt=30から落ちる
    t_smooth      = np.linspace(1, km_df['t'].max() * 1.3, 600)
    S_wei         = weibull_s(t_smooth, k, lam)
    km_t_plot     = np.concatenate([[0, ltv_offset_days], km_df['t'].values + ltv_offset_days])
    km_s_plot     = np.concatenate([[1.0, 1.0], km_df['S'].values])
    t_smooth_plot = t_smooth + ltv_offset_days
    S_wei_plot    = S_wei

with c1:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.step(km_t_plot, km_s_plot, where='post', color=ACCENT, lw=1.8, label='KM Curve (Observed)')
    ax.plot(t_smooth_plot, S_wei_plot, color=ACCENT2, lw=1.5, ls='--', label='Weibull Fit')
    ax.fill_between(t_smooth_plot, S_wei_plot, alpha=0.06, color=ACCENT2)
    ax.set(xlabel='Days', ylabel='Survival Rate S(t)', ylim=(0,1.05),
           xlim=(0, t_smooth_plot.max()))
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(True, alpha=0.25)
    ax.set_title('Survival Curve', color='#ccc', fontsize=10, pad=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with c1:
    st.caption("Survival Curve：実測のKM曲線（実線）にWeibullモデルをフィット（破線）。右に伸びるほど顧客が長く継続している。")

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
    st.caption("Weibull Linearization Plot：生存率を対数変換して直線化したもの。R²が1.0に近いほどWeibullモデルのフィット精度が高い。")

# Save chart images for export
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.step(km_t_plot, km_s_plot, where='post', color=ACCENT, lw=2, label='KM Curve')
ax1.plot(t_smooth_plot, S_wei_plot, color=ACCENT2, lw=1.8, ls='--', label='Weibull Fit')
ax1.fill_between(t_smooth_plot, S_wei_plot, alpha=0.07, color=ACCENT2)
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

st.markdown("<div class='section-title' style='margin-bottom:-1rem;'>暫定 LTV — 観測期間別</div>", unsafe_allow_html=True)

horizons = [180, 365, 730, 1095, 1825]  # 180日・1年・2年・3年・5年

# ── 折れ線グラフ（180日〜5年、λを縦線で表示）──────────────
# グラフ用の細かい点を生成（滑らかな曲線）
# λをグラフ最大値として扱う（5年超の場合はλが最大）
lam_actual = lam + ltv_offset_days  # 実際のλ位置（オフセット込み）
x_max = max(1825, round(lam_actual) + 100) if lam_actual > 1825 else 1825

t_range = list(range(1, x_max + 1, max(1, x_max // 300)))
if business_type == "都度購入型":
    _dorm_off = dormancy_days or 180
    rev_line = [ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, t, _dorm_off) for t in t_range]
    gp_line  = [ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, t, _dorm_off) for t in t_range]
else:
    rev_line = [ltv_horizon_offset(k, lam, arpu_daily, t, ltv_offset_days) for t in t_range]
    gp_line  = [ltv_horizon_offset(k, lam, gp_daily,   t, ltv_offset_days) for t in t_range]
cac_line = [v / cac_n for v in gp_line]

lam_int = round(lam_actual)

_col_ltv, = st.columns([1])
with _col_ltv:
    fig_ltv = go.Figure()
    fig_ltv.add_trace(go.Scatter(x=t_range, y=rev_line, name='LTV（売上）', mode='lines', line=dict(color='#56b4d3', width=2)))
    fig_ltv.add_trace(go.Scatter(x=t_range, y=gp_line,  name='LTV（粗利）', mode='lines', line=dict(color='#a8dadc', width=2, dash='dash')))
    fig_ltv.add_trace(go.Scatter(x=t_range, y=cac_line, name='CAC上限',    mode='lines', line=dict(color='#4a7a8a', width=1.5, dash='dot')))
    fig_ltv.add_hline(y=ltv_rev, line_dash='dot', line_color='#56b4d3', line_width=1, opacity=0.4,
        annotation_text=f'LTV∞ ¥{ltv_rev:,.0f}', annotation_position='right',
        annotation_font=dict(color='#56b4d3', size=10))
    fig_ltv.add_shape(type='line', x0=lam_actual, x1=lam_actual, y0=0, y1=1, yref='paper',
        line=dict(color='#a8dadc', width=1.5, dash='dash'), layer='above')
    fig_ltv.add_annotation(x=lam_actual, y=0.85 if k < 1.0 else 0.35, yref='paper',
        text=f'λ＝{lam_int}日', showarrow=False,
        font=dict(color='#a8dadc', size=10), xanchor='center', yanchor='middle',
        bgcolor='#111820', borderpad=2)
    tick_vals = [180, 365, 730, 1095, 1460, 1825]
    tick_text = ['180日', '1年', '2年', '3年', '4年', '5年']
    fig_ltv.update_layout(
        paper_bgcolor='#111820', plot_bgcolor='#111820',
        height=280, margin=dict(t=30, b=50, l=70, r=120),
        font=dict(color='#ccc', size=10),
        legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title='継続期間', gridcolor='#1a3040', tickvals=tick_vals, ticktext=tick_text, tickfont=dict(color='#888'), range=[0, x_max + 50]),
        yaxis=dict(title='金額（円）', gridcolor='#1a3040', tickfont=dict(color='#888'), tickformat='¥,.0f'),
    )
    st.plotly_chart(fig_ltv, use_container_width=True)

# ── 日本語フォント動的探索ヘルパー ──
def _find_jp_font_path():
    """環境に依存せず日本語フォントパスを返す。見つからなければNone。"""
    import os
    _candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
        '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
        '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
        '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    ]
    for _p in _candidates:
        if os.path.exists(_p):
            return _p
    try:
        import matplotlib.font_manager as _fmx
        _all = _fmx.findSystemFonts()
        for _kw in ['CJK', 'ipagp', 'ipag', 'Japanese', 'Gothic']:
            for _fp in _all:
                if _kw.lower() in _fp.lower():
                    return _fp
    except Exception:
        pass
    return None

_JP_FONT_PATH = _find_jp_font_path()

# ── PPTX用バッファ：matplotlib で別途生成 ──
try:
    import matplotlib.font_manager as fm
    if _JP_FONT_PATH:
        _jp_font = fm.FontProperties(fname=_JP_FONT_PATH)
        plt.rcParams['font.family'] = _jp_font.get_name()
except Exception:
    pass
fig_ltv_pp, ax_ltv_pp = plt.subplots(figsize=(10, 3.5))
fig_ltv_pp.patch.set_facecolor('#111820')
ax_ltv_pp.set_facecolor('#111820')
ax_ltv_pp.plot(t_range, rev_line, color='#56b4d3', lw=2, label='LTV（売上）')
ax_ltv_pp.plot(t_range, gp_line,  color='#a8dadc', lw=2, ls='--', label='LTV（粗利）')
ax_ltv_pp.plot(t_range, cac_line, color='#4a7a8a', lw=1.5, ls=':', label='CAC上限')
ax_ltv_pp.axhline(ltv_rev, color='#56b4d3', lw=0.8, ls=':', alpha=0.5, label=f'LTV∞ ¥{ltv_rev:,.0f}')
ax_ltv_pp.axvline(lam_actual, color='#a8dadc', lw=1.2, ls='--', alpha=0.7)
xtick_vals = [180, 365, 730, 1095, 1460, 1825]
xtick_text_pp = ['180日', '1年', '2年', '3年', '4年', '5年']
ax_ltv_pp.set_xticks(xtick_vals)
ax_ltv_pp.set_xticklabels(xtick_text_pp)
ax_ltv_pp.set_xlim(0, x_max + 50)
ax_ltv_pp.set_xlabel('継続期間', color='#888', fontsize=9)
ax_ltv_pp.set_ylabel('金額（円）', color='#888', fontsize=9)
ax_ltv_pp.tick_params(colors='#888')
ax_ltv_pp.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'¥{v:,.0f}'))
ax_ltv_pp.legend(fontsize=8, framealpha=0.2, labelcolor='white', loc='upper left')
ax_ltv_pp.grid(True, alpha=0.2, color='#1a3040')
for spine in ax_ltv_pp.spines.values(): spine.set_color('#1a3040')
fig_ltv_pp.tight_layout()
buf_ltv = io.BytesIO()
fig_ltv_pp.savefig(buf_ltv, format='png', dpi=150, bbox_inches='tight', facecolor='#111820')
buf_ltv.seek(0)
plt.close(fig_ltv_pp)
plt.rcParams['font.family'] = 'DejaVu Sans'

# λ時点と99%到達日数を逆算
try:
    if business_type == "都度購入型":
        _dorm_off = dormancy_days or 180
        days_99 = brentq(
            lambda h: ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, h, _dorm_off) / ltv_rev - 0.99,
            1, 365000
        )
    else:
        days_99 = brentq(
            lambda h: ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days) / ltv_rev - 0.99,
            1, 365000  # 上限1000年に拡大
        )
except Exception:
    # 上限でも99%に届かない場合は上限値で近似
    days_99 = 365000

def fmt_horizon(days):
    if days < 365:
        return f'{int(days)}日'
    elif days % 365 == 0:
        return f'{int(days//365)}年（{int(days):,}日）'
    else:
        return f'{days/365:.1f}年（{int(days):,}日）'

# テーブルデータ構築
tbl_rows = []
for h in horizons:
    if business_type == "都度購入型":
        _dorm_off = dormancy_days or 180
        lh_rev = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, h, _dorm_off)
        lh_gp  = ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, h, _dorm_off)
    else:
        lh_rev = ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days)
        lh_gp  = ltv_horizon_offset(k, lam, gp_daily,   h, ltv_offset_days)
    tbl_rows.append({
        'ホライズン':    fmt_horizon(h),
        'LTV（売上）':   f'¥{lh_rev:,.0f}',
        'LTV（粗利）':   f'¥{lh_gp:,.0f}',
        'CAC上限':       f'¥{lh_gp/cac_n:,.0f}',
        'LTV∞到達率':   f'{lh_rev/ltv_rev*100:.1f}%',
        '_type': 'normal',
    })

# λ行
if business_type == "都度購入型":
    _dorm_off = dormancy_days or 180
    lam_rev = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, lam + _dorm_off, _dorm_off)
    lam_gp  = ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, lam + _dorm_off, _dorm_off)
else:
    lam_rev  = ltv_horizon_offset(k, lam, arpu_daily, lam + ltv_offset_days, ltv_offset_days)
    lam_gp   = ltv_horizon_offset(k, lam, gp_daily,   lam + ltv_offset_days, ltv_offset_days)
lam_pct  = lam_rev / ltv_rev * 100
tbl_rows.append({
    'ホライズン':    f'λ  {round(lam + ltv_offset_days):,}日',
    'LTV（売上）':   f'¥{lam_rev:,.0f}',
    'LTV（粗利）':   f'¥{lam_gp:,.0f}',
    'CAC上限':       f'¥{lam_gp/cac_n:,.0f}',
    'LTV∞到達率':   f'{lam_pct:.1f}%',
    '_type': 'lambda',
})

# 99%到達行
if business_type == "都度購入型":
    _dorm_off = dormancy_days or 180
    rev_99 = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, days_99, _dorm_off)
    gp_99  = ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, days_99, _dorm_off)
else:
    rev_99 = ltv_horizon_offset(k, lam, arpu_daily, days_99, ltv_offset_days)
    gp_99  = ltv_horizon_offset(k, lam, gp_daily,   days_99, ltv_offset_days)
tbl_rows.append({
    'ホライズン':    f'LTV∞到達率: 99%  （{int(days_99):,}日）',
    'LTV（売上）':   f'¥{rev_99:,.0f}',
    'LTV（粗利）':   f'¥{gp_99:,.0f}',
    'CAC上限':       f'¥{gp_99/cac_n:,.0f}',
    'LTV∞到達率':   '99.0%',
    '_type': '99pct',
})

# HTML テーブルで描画
ACCENT   = '#56b4d3'
BG_HEAD  = '#0d1f2d'
BG_ROW1  = '#0d1520'
BG_ROW2  = '#0d1520'
SEP_COLOR = '#1a3a4a'  # λ・99%行の区切り線色

html_rows = ''
for i, row in enumerate(tbl_rows):
    rtype = row['_type']
    if rtype == 'normal':
        bg    = BG_ROW1 if i % 2 == 0 else BG_ROW2
        color = '#c8d0d8'
        border_top = ''
    elif rtype == 'lambda':
        bg    = BG_HEAD
        color = '#a8dadc'
        border_top = f"border-top:1px solid {SEP_COLOR};"
    else:  # 99pct
        bg    = BG_HEAD
        color = '#a8dadc'
        border_top = f"border-top:1px solid {SEP_COLOR};"

    html_rows += f"<tr style='background:{bg}; {border_top}'>"
    html_rows += f"<td style='text-align:left; padding:8px 14px; color:{color}; font-size:0.85rem; width:28%;'>{row['ホライズン']}</td>"
    for col in ['LTV（売上）', 'LTV（粗利）', 'CAC上限', 'LTV∞到達率']:
        html_rows += f"<td style='text-align:right; padding:8px 14px; color:{color}; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>{row[col]}</td>"
    html_rows += '</tr>'

tbl_html = f"""
<table style='width:100%; border-collapse:collapse; margin-top:4px; table-layout:fixed;'>
  <colgroup>
    <col style='width:28%;'>
    <col style='width:18%;'>
    <col style='width:18%;'>
    <col style='width:18%;'>
    <col style='width:18%;'>
  </colgroup>
  <thead>
    <tr style='background:{BG_HEAD};'>
      <th style='text-align:left; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT};'>ホライズン</th>
      <th style='text-align:right; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT};'>LTV（売上）</th>
      <th style='text-align:right; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT};'>LTV（粗利）</th>
      <th style='text-align:right; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT};'>CAC上限</th>
      <th style='text-align:right; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT};'>LTV∞到達率</th>
    </tr>
  </thead>
  <tbody>
    {html_rows}
  </tbody>
</table>
"""
st.markdown(tbl_html, unsafe_allow_html=True)


# 解釈ガイドを自動生成
if business_type == "都度購入型":
    _dorm_off = dormancy_days or 180
    ltv_1y = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, 365,  _dorm_off)
    ltv_2y = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, 730,  _dorm_off)
    ltv_3y = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, 1095, _dorm_off)
else:
    ltv_1y  = ltv_horizon_offset(k, lam, arpu_daily, 365,  ltv_offset_days)
    ltv_2y  = ltv_horizon_offset(k, lam, arpu_daily, 730,  ltv_offset_days)
    ltv_3y  = ltv_horizon_offset(k, lam, arpu_daily, 1095, ltv_offset_days)
pct_1y  = ltv_1y  / ltv_rev * 100
pct_2y  = ltv_2y  / ltv_rev * 100
pct_3y  = ltv_3y  / ltv_rev * 100

# CAC回収期間を逆算（売上ベース・粗利ベース両方）
def recover_str(days):
    return f"{days/365:.1f}年（{int(days):,}日）" if days >= 365 else f"{int(days)}日"

try:
    cac_recover_rev = brentq(
        lambda h: (ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, h, dormancy_days or 180) if business_type == "都度購入型" else ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days)) - cac_upper,
        1, 36500
    )
    cac_recover_rev_str = recover_str(cac_recover_rev)
except Exception:
    cac_recover_rev_str = "算出不可"

try:
    cac_recover_gp = brentq(
        lambda h: (ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, h, dormancy_days or 180) if business_type == "都度購入型" else ltv_horizon_offset(k, lam, gp_daily, h, ltv_offset_days)) - cac_upper,
        1, 36500
    )
    cac_recover_gp_str = recover_str(cac_recover_gp)
except Exception:
    cac_recover_gp_str = "算出不可"

# 後方互換用
cac_recover_days = cac_recover_rev if cac_recover_rev_str != "算出不可" else None
cac_recover_str  = cac_recover_rev_str

# λの解釈（都度購入型はlam+dormancy_daysで表示）
lam_display = lam + ltv_offset_days if business_type == "都度購入型" else lam
if lam_display < 180:
    lam_desc = f"λ={lam_display:.0f}日は非常に短く、顧客の大半が半年以内に離脱するビジネスです。"
elif lam_display < 365:
    lam_desc = f"λ={lam_display:.0f}日は比較的短く、多くの顧客が1年以内に離脱するビジネスです。"
elif lam_display < 730:
    lam_desc = f"λ={lam_display:.0f}日（約{lam_display/365:.1f}年）は中程度の継続期間で、1〜2年継続する顧客が多いビジネスです。"
else:
    lam_desc = f"λ={lam_display:.0f}日（約{lam_display/365:.1f}年）は長く、顧客が数年にわたって継続するビジネスです。"

# k の解釈（投資回収の観点を含む）
if k < 0.8:
    k_desc = (
        f"k={k:.3f}（強い初期離脱型）: {acq_label}直後に大量離脱するパターンです。"
        f"LTV∞は大きく見えますが少数の超長期顧客の分が含まれており、99%到達まで長期間かかります。"
        f"CAC投資判断には暫定LTV（現実的な期間）を使ってください。"
    )
elif k < 1.0:
    k_desc = (
        f"k={k:.3f}（緩やかな初期離脱型）: 離脱率がほぼ一定に近いパターンです。"
        f"LTV∞の回収にある程度の期間がかかります。暫定LTVと99%到達日数を確認してCACを設計してください。"
    )
elif k < 1.5:
    k_desc = (
        f"k={k:.3f}（逓増離脱型）: 初期の離脱は少なく時間とともに解約が増えるパターンです。"
        f"LTV∞の大部分を比較的短期間で回収できます。積極的なCAC投資が可能な構造です。"
    )
else:
    k_desc = (
        f"k={k:.3f}（強い逓増型）: 縛り期間や習慣化により初期継続率が高いパターンです。"
        f"LTV∞をほぼ数年以内に回収できます。ただし離脱が集中する時期のリテンション施策が重要です。"
    )
insight_html = f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-radius:10px; padding:18px 20px; margin-top:12px; line-height:1.9; font-size:0.85rem; color:#ccc;'>
  <div style='font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:#3a6a7a; margin-bottom:12px;'>このテーブルの読み方</div>
  <div>・{lam_desc}</div>
  <div>・{k_desc}</div>
  <div>・LTV∞（¥{ltv_rev:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。</div>
  <div style='margin-top:8px;'>
    <span style='color:#56b4d3;'>1年時点</span>でLTV∞の<b style='color:#a8dadc;'>{pct_1y:.1f}%</b>（¥{ltv_1y:,.0f}）、
    <span style='color:#56b4d3;'>2年時点</span>で<b style='color:#a8dadc;'>{pct_2y:.1f}%</b>（¥{ltv_2y:,.0f}）、
    <span style='color:#56b4d3;'>3年時点</span>で<b style='color:#a8dadc;'>{pct_3y:.1f}%</b>（¥{ltv_3y:,.0f}）に到達します。
  </div>
  <div style='margin-top:8px;'>・CAC上限（¥{cac_upper:,.0f}）の回収期間：売上ベース 約 <b style='color:#a8dadc;'>{cac_recover_rev_str}</b> / 粗利ベース 約 <b style='color:#56b4d3;'>{cac_recover_gp_str}</b>（{acq_label}から）</div>
  <div style='margin-top:12px; padding-top:10px; border-top:1px solid #1a3a4a;'>
    <span style='color:#56b4d3; font-weight:600;'>CAC設計の目安</span>：回収期間に迷ったら、
    <b style='color:#a8dadc;'>λ={round(lam_actual):,}日（約{lam_actual/365:.1f}年）時点の暫定LTV（粗利）¥{lam_gp:,.0f}</b>
    を用いてCAC上限を算出してください。λは{"リピート顧客の63.2%が離脱するまでの期間（初回購入起点）" if business_type == "都度購入型" else "多くの顧客が離脱するまでの期間の目安"}をデータが示した答えです。
  </div>
</div>
"""
st.markdown(insight_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# AI Prompt Generator
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>AI Prompt Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='help-box'>この結果の読み方や戦略への活用方法がわからない場合は、以下のプロンプトをClaude・ChatGPT・Geminiにコピペしてください。テキストボックス右上の <b>コピーアイコン</b> をクリックすると全文コピーできます。</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["分析結果の解釈", "マーケティング意思決定", "分析の限界と改善"])

dormancy_label = "なし（解約日ベース）" if dormancy_days is None else f"{dormancy_days}日"
churned_count  = int(df['event'].sum())
active_count   = int((df['event']==0).sum())
churn_rate     = churned_count / len(df) * 100
# ビジネスタイプで「契約」「初回購入」を切り替え
acq_label  = "初回購入" if business_type == "都度購入型" else "契約"
date_label = "初回購入日" if business_type == "都度購入型" else "契約開始日"

k_pattern      = f"初期集中型（{acq_label}直後の離脱が多い）" if k < 1 else "逓増型（時間とともに離脱が増える）"

prompt_base = f"""私はLTV分析ツール（Kaplan-Meier法 × Weibullモデル）を使い、以下の結果を得ました。

【分析結果】
・ビジネスタイプ: {business_type} / 休眠判定: {dormancy_label}
・顧客数: {len(df):,}件（解約済み: {churned_count:,}件 / 継続中: {active_count:,}件 / 解約率: {churn_rate:.1f}%）
・平均日次ARPU（売上）: ¥{arpu_daily:,.2f} / 平均日次GP（粗利）: ¥{gp_daily:,.2f} / GPM: {gpm:.1%}
・LTV∞（売上ベース）: ¥{ltv_rev:,.0f} / LTV∞（粗利ベース）: ¥{ltv_val:,.0f}
・CAC上限（{cac_label}）: ¥{cac_upper:,.0f}
・Weibull k（形状）: {k:.4f} → {k_pattern}
・Weibull λ（尺度）: {lam+ltv_offset_days if business_type=="都度購入型" else lam:.1f}日 / R²（フィット精度）: {r2:.4f}
・分析手法: Kaplan-Meier法 + Weibullモデルによる生存分析"""

with tab1:
    p1 = prompt_base + f"""

【質問】―― 以下の数値を具体的に使って答えてください ――
1. k={k:.4f}という値はこのビジネスの離脱パターンと投資回収のしやすさについて何を示していますか？k<1は「LTV∞の回収に長期間かかる・少数の超長期顧客の影響が大きい」、k>1は「比較的短期にLTV∞を回収できる」という観点で解釈してください。
2. 99%到達まで{int(days_99):,}日（{days_99/365:.1f}年）かかるという事実は、このビジネスのCAC設計にどう影響しますか？現実的な回収期間として何年を目安にすべきですか？
3. 解約率{churn_rate:.1f}%（解約{churned_count:,}件・継続{active_count:,}件）という比率から、このビジネスの健全性をどう評価しますか？
4. R²={r2:.4f}のフィット精度はWeibull分析として許容範囲ですか？この値が示す信頼性の限界を教えてください。"""
    st.code(p1, language=None)

with tab2:
    p2 = prompt_base + f"""

【質問】―― 上記の数値から直接導ける意思決定を具体的に答えてください ――
1. このビジネスのλ={lam_display:.0f}日（約{lam_display/365:.1f}年）時点の暫定LTV（粗利）¥{lam_gp:,.0f}をCAC上限の基準とした場合、LTV:CAC={cac_n:.1f}:1の設定でCACの目安はいくらですか？またこの設定は適切ですか？
2. k={k:.4f}のビジネスにおいて、LTV∞をそのままCAC判断に使うリスクを説明してください。λ日基準・1年基準・2年基準それぞれのCAC上限（1年:¥{ltv_1y*gpm/cac_n:,.0f}、2年:¥{ltv_2y*gpm/cac_n:,.0f}、λ={round(lam_display)}日:¥{lam_gp/cac_n:,.0f})を比較して、最も実務的な基準はどれですか？
3. λ={lam_display:.0f}日を踏まえると、{acq_label}後何日目にリテンション施策を打つのが最も効果的ですか？
4. {k_pattern}に対して最も効果的なリテンション施策のタイミングと種類を教えてください。"""
    st.code(p2, language=None)

with tab3:
    p3 = prompt_base + f"""

【質問】―― モデルの信頼性・限界・改善策を具体的に答えてください ――
1. {len(df):,}件・解約{churned_count:,}件のサンプルサイズでWeibull推定の信頼区間はどの程度ですか？十分なサンプル数の目安を教えてください。
2. k={k:.4f}はWeibull分布の単調ハザード率の仮定を満たしていますか？仮定が崩れる典型例とチェック方法を教えてください。
3. R²={r2:.4f}を改善するにはどうすればよいですか？データ量・期間・外れ値処理の観点から具体的に教えてください。
{"4. 解約日ベースで分析していますが、解約データの欠損や遅延がある場合にLTV推定にどんな影響が出ますか？" if dormancy_days is None else f"4. 休眠判定{dormancy_label}の設定はこのビジネスに適切ですか？最適な判定日数を決める感度分析の手順を教えてください。"}
"""
    st.code(p3, language=None)

# ══════════════════════════════════════════════════════════════
# Export buttons
# ══════════════════════════════════════════════════════════════

st.markdown("<div class='section-title'>Export</div>", unsafe_allow_html=True)

# ── Excel export ─────────────────────────────────────────────
if True:
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
            ('λ（尺度パラメータ・日）', round(lam + ltv_offset_days if business_type == '都度購入型' else lam, 2)),
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
        # t=0：生存率100%
        ws2.append([0, 1.0, 1.0])
        if business_type == "都度購入型":
            # 都度購入型：オフセット後のkm_dfを使い、t軸を元に戻す
            ws2.append([int(ltv_offset_days), 1.0, 1.0])
            for _, row in km_df.iterrows():
                t_orig = int(row['t'] + ltv_offset_days)
                ws2.append([t_orig, round(row['S'], 6), round(float(weibull_s(row['t'], k, lam)), 6)])
        else:
            # サブスク：km_df（オフセット後）のt軸を+ltv_offset_daysで元スケールに戻す
            # t=0はヘッダー直後に既に追加済みなので重複しない
            if ltv_offset_days > 0:
                ws2.append([int(ltv_offset_days), 1.0, round(float(weibull_s(0, k, lam)), 6)])
            for _, row in km_df.iterrows():
                t      = row['t']
                t_plot = int(t + ltv_offset_days)
                ws2.append([t_plot, round(row['S'], 6), round(float(weibull_s(t, k, lam)), 6)])

        # Horizon sheet
        ws3 = wb.create_sheet('暫定LTV')
        ws3.append(['ホライズン（日）', '暫定LTV（¥）', 'LTV∞比（%）', f'CAC上限（¥）'])
        for h in horizons:
            if business_type == "都度購入型":
                # 都度購入型：LTV_short（固定）+ LTV_long（Weibull積分）
                _dorm_off = dormancy_days or 180
                _h_short  = min(h, _dorm_off)
                _h_long   = max(h - _dorm_off, 0)
                _lh_short = _h_short * arpu_0_dorm
                _lh_long  = ltv_horizon_offset(k, lam, arpu_long, _h_long, 0) if _h_long > 0 else 0
                lh = _lh_short + _lh_long
            else:
                lh = ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days)
            ws3.append([h, round(lh, 0), round(lh/ltv_val*100, 1), round(lh/cac_n, 0)])

        # セグメント別シート追加
        if segment_cols_input.strip():
            seg_cols_xl = [c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
            for sc in seg_cols_xl:
                # ── セグメント概要シート ──
                ws_seg = wb.create_sheet(f'SEG_{sc}'[:31])
                ws_seg.append(['セグメント', '顧客数', 'LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）', 'k', 'λ（日）', 'R²', '獲得効率'])
                seg_vals = df[sc].dropna().unique()
                seg_rows = []
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = _compute_km_df(df_s)
                        k_s, lam_s, r2_s, _ = _fit_weibull_df(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['revenue_total'].sum() / df_s['duration'].sum() if billing_cycle == '日次（都度購入）' else df_s['arpu_daily'].mean()
                        gp_s   = arpu_s * gpm
                        ltv_r_s, _ = ltv_inf_offset(k_s, lam_s, arpu_s, ltv_offset_days)
                        ltv_g_s, _ = ltv_inf_offset(k_s, lam_s, gp_s,   ltv_offset_days)
                        eff = round(ltv_g_s / cac_input, 2) if cac_known else '-'
                        seg_rows.append([sv, len(df_s), round(ltv_r_s,0), round(ltv_g_s,0), round(ltv_g_s/cac_n,0), round(k_s,4), round(lam_s,1), round(r2_s,4), eff])
                    except Exception:
                        continue
                seg_rows.sort(key=lambda x: x[2], reverse=True)
                for row in seg_rows:
                    ws_seg.append(row)
                ws_seg.column_dimensions['A'].width = 20

                # ── セグメント別暫定LTV詳細シート ──
                ws_seg_hor = wb.create_sheet(f'SEG_{sc}_暫定LTV'[:31])
                hor_header = ['セグメント', 'ホライズン', 'LTV（売上）', 'LTV（粗利）', 'CAC上限', 'LTV∞到達率（%）']
                ws_seg_hor.append(hor_header)
                hor_points = [180, 365, 730, 1095, 1825]
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = _compute_km_df(df_s)
                        k_s, lam_s, r2_s, _ = _fit_weibull_df(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['revenue_total'].sum() / df_s['duration'].sum() if billing_cycle == '日次（都度購入）' else df_s['arpu_daily'].mean()
                        gp_s   = arpu_s * gpm
                        ltv_inf_s, _ = ltv_inf_offset(k_s, lam_s, arpu_s, ltv_offset_days)
                        lam_s_actual = lam_s + ltv_offset_days
                        # 通常ホライズン
                        for h in hor_points:
                            label = f'{h}日' if h < 365 else f'{h//365}年（{h}日）'
                            lh_r = ltv_horizon_offset(k_s, lam_s, arpu_s, h, ltv_offset_days)
                            lh_g = ltv_horizon_offset(k_s, lam_s, gp_s,   h, ltv_offset_days)
                            pct  = round(lh_r / ltv_inf_s * 100, 1) if ltv_inf_s > 0 else 0
                            ws_seg_hor.append([str(sv), label, round(lh_r,0), round(lh_g,0), round(lh_g/cac_n,0), pct])
                        # λ行
                        lam_r = ltv_horizon_offset(k_s, lam_s, arpu_s, lam_s_actual, ltv_offset_days)
                        lam_g = ltv_horizon_offset(k_s, lam_s, gp_s,   lam_s_actual, ltv_offset_days)
                        lam_pct = round(lam_r / ltv_inf_s * 100, 1) if ltv_inf_s > 0 else 0
                        ws_seg_hor.append([str(sv), f'λ（{int(lam_s_actual)}日）', round(lam_r,0), round(lam_g,0), round(lam_g/cac_n,0), lam_pct])
                        # 99%到達行
                        try:
                            days_99_s = brentq(lambda h: ltv_horizon_offset(k_s, lam_s, arpu_s, h, ltv_offset_days) / ltv_inf_s - 0.99, 1, 365000)
                            r99_r = ltv_horizon_offset(k_s, lam_s, arpu_s, days_99_s, ltv_offset_days)
                            r99_g = ltv_horizon_offset(k_s, lam_s, gp_s,   days_99_s, ltv_offset_days)
                            ws_seg_hor.append([str(sv), f'LTV∞到達率: 99%（{int(days_99_s):,}日）', round(r99_r,0), round(r99_g,0), round(r99_g/cac_n,0), 99.0])
                        except Exception:
                            pass
                        # 空行で区切り
                        ws_seg_hor.append([''] * 6)
                    except Exception:
                        continue
                ws_seg_hor.column_dimensions['A'].width = 20
                ws_seg_hor.column_dimensions['B'].width = 18

        xl_buf = io.BytesIO()
        wb.save(xl_buf)
        xl_buf.seek(0)
        import base64 as _b64
        _xl_b64 = _b64.b64encode(xl_buf.read()).decode()
        _fn_xl = f"LTV分析_{client_name or 'report'}.xlsx"
        _xl_href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{_xl_b64}" download="{_fn_xl}" class="dl-btn">.xlsx</a>'
        _xl_html = _xl_href
    except Exception as e:
        _xl_html = f'<span class="dl-btn-err">.xlsx エラー</span>'

# ── PowerPoint export ─────────────────────────────────────────
if True:
    try:
        _TMPL_B64 = (
            "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"
            "A57rhoBBzwOhBQOKLUgEMFwxK1iX/l7hHFgY3isMsDDAwgALAywMsHDFsIBu5G8QDvZ9P/CA43oYOA5i"
            "ACPmg4BhH5uicmCzK2YF+1JYgSLDJIQOv0EMrDCwwsAKAysMrHCdXiys/m18k2DBCjCkxDVE7wECmNkE"
            "hCQQxAAJMxwvJMyDVwwL9NJhYXixMMDCAAsDLAywMMDCNYGF1ZuFnaRJuSb3QF3P/zNizwocwQOuYwHi"
            "QAgoQS6wQx9j07Yd06CvfptOI210KVSAsS3AoHuDYBPLOckEEBJkiE8LBRZxLKJc5BUwgRbl8ayoxvqu"
            "ap4X7rwppkknoq3zEmxos43CJyQzoLartrvtRjVPE7mr0FBiL57+zlnFz8pk64XN/ZwDqZLWevr/D53V"
            "NrgdOW3HIdu6rr/euR60MfaAwUwGWOhZwIEsACiAoR8SwzbxK91A1EaR9Kkzt7qhSwotCnHL2xQR0q7K"
            "54YWhcSyrxi3tSyqNtUezCSfCGTug2R3fq/I2/16x8IKWiqs1g4oSl3X89YOqBVidlqZxEayw/9QrRX/"
            "Ip12UhxomheRcgoSOykQ2+3o1xVzCtZ6MRRRlZvWFXPZ0LTzuGhTz/Z8V60h62Ufdeq3FvdxqEpdNvE8"
            "x0KMesCDZghM37GBG1rymzM2TeZRl+FAZpMSmi9mE3FzvWxSFo95VRaJ2pMNjS6hKF8TsyO+0xLiKDga"
            "Kd368yprbMttn+KcVnej8v5CmSxT4MjUrVKmqrbqURU59n4T+u1/AQAA//8DAFBLAwQUAAYACAAAACEA"
            "le8a4yUCAAA4BAAAIQAAAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQ4LnhtbIxTwW7bMAy9D9g/"
            "CNrZVWInw2rUKRon26VrA7g9F6qtxMZkSZNkL94woL+1fU6/ZJQcx92WQy8iRYqPfE/SxeW+5qhl2lRS"
            "JHh6NsGIiVwWldgl+P7uY/ABI2OpKCiXgiW4YwZfLt6+uVCx4cU17WRjEWAIE9MEl9aqmBCTl6ym5kwq"
            "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"
            "nbkw4G4E5voFY4DASOdo2+NCRRy6UBpRvy9gcDCyQC8EZaFVQky/zda6kuN538p4ZE0ujNQhDZGg0OlU"
            "JAg5ULmdr2BWy7tiXhk5o7zPFOrKNPsckmPCBrp6W9p+D0WzbpI7wuAWg2bvc2cMQ12ob+vkk6XN644H"
            "c0EZ/brCSk2/9FWwdTYVXvOrNutYS+LqzbW/alO4piD9Dxo3FT6bz7cDfgjRR8VEiSARL2SDB9KlmK2G"
            "oHN2mEnTrP7bMWoegkLufzh8lpxdjEsLzv53cW/u7Hxl+bqcRw5XV5ZLVI9Hs4uM2S39qsWH90D2HtyP"
            "JkzJ7N3TfbiUdme/RwCfTKIh3fkHAAD//wMAUEsDBAoAAAAAAAAAIQAH802mvcgAAL3IAAAUAAAAcHB0"
            "L21lZGlhL2ltYWdlMy5wbmeJUE5HDQoaCgAAAA1JSERSAAAECgAAAkcIBgAAABmytKQAAAA6dEVYdFNv"
            "ZnR3YXJlAE1hdHBsb3RsaWIgdmVyc2lvbjMuMTAuOCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/BW3XO"
            "AAAACXBIWXMAABcSAAAXEgFnn9JSAADIKUlEQVR4nOzdd3iTVfsH8G92mqaL0UVLF7RA2XtPQUQEBQTF"
            "jSiun/q65+vWV1BBRRTFhSyRIQiyZ4GyKZuWVbo3nWma8eT3R2kgNKVtkjZt+v1cF5fmPOc5525PWnju"
            "nCHy8Q81gYiIiIiIiIgIgNjZARARERERERFRw8FEARERERERERGZMVFARERERERERGZMFBARERERERGR"
            "GRMFRERERERERGTGRAERERERERERmTFRQERERERERERmTBQQERERERERkRkTBURERERERERkxkQBERER"
            "EREREZkxUUBEREREREREZkwUEBEREREREZEZEwVEREREREREZMZEARERERERERGZMVFARERERERERGZM"
            "FBARkcv44rMPsGXdCnTuFO3UOLasW4Et61Y4NYam4qP/vol/Vy+Fn29LZ4dCtfDHz/OwZd2KSuP26ovP"
            "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"
            "ERERkRkTBURERERERERkxkQBEREREREREZkxUUBEREREREREZkwUEBEREREREZEZEwVEREREREREZMZE"
            "ARERERERERGZMVFARERERERERGZMFBARERERERGRGRMFRERERERERGTGRAERERERERERmTFRQERERERE"
            "RERmTBQQERERERERkRkTBURERERERERkJnV2AE2FsmUQIBLDZDQ4OxQiIiIiIiJyYSKJFDAJ0Gan2HQ/"
            "ZxTUF5EYEDk7iBoSNZZAqUY4nq6F4+laOJ6uhePpWjieroXj6Vo4ntUTofwZ1EacUVBPKmYSlOWmOTmS"
            "6onlSgg6rbPDIAfheLoWjqdr4Xi6Fo6na+F4uhaOp2vheFZP0TzQrvs5o4CIiIiIiIiIzJgoICIiIiIi"
            "IiIzJgqIiIiIiIiIyIyJAiIiIiIiIiIyY6KAiIiIiIiIiMx46gEREREREVEtiXhEn9OIRKIm9/03mUz1"
            "2h8TBURERERERNUQiUTw8FDDTamEVMrHKGcSyxQQ9GXODqPeGQwGlGq1KCoqrvPEAd/hREREREREtyAS"
            "idCiRXPIZTJnh0IATBCcHYJTSKVSeKjVUCgUyMnJrdNkARMFREREREREt+DhoYZcJoMgCCgoKIS2rKze"
            "p4LTdSKZHCa9ztlh1CuRSASlQgEvL0/IZTJ4eKhRWFhUZ/0xUUBERERERHQLbkolAKCgoBCa0lInR0Mw"
            "mZpcosZkMpnfez4+3nBTKus0UcBTD4iIiIiIiG6hYk8CbVnTWxdPDUvFe7Cu98lotDMK3H394RUSBrV/"
            "INT+AVB4eAIAYr/61Kb2JAolgvsNQrM2kZCp3KHXlCDvQjySY2Ng5C8EIiIiIqIm6cbd9Zvap9jU8Nz4"
            "HhSJRHX2nmy0iYKgvgPQrE2UQ9qSKt3Q8f5H4ObTDNr8q8i7mABV85YI6N4b3qEROLXsdxi0Wof0RURE"
            "RERERNSQNdpEQVF6KjQ52SjOSENxRjq6T38WYhunX4QOGwk3n2bIPX8OCetWA9eyMqHDRiKgWy+EDLkN"
            "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"
            "iX/4p97+1DcmCsjpDm21fKDvd8coJ0VCRERERERETBSQ0x2P2Qt9WZn5dfdhQyHj0TNEREREREROwUQB"
            "OZ1Wo8HJ2APm127uKnQdPMCJERERERERETVdTBRQg3B4283LD0Y7KRIiIiIiIqKmjYkCahAS4k6gICfX"
            "/FrppoREykM5iIiIiIiI6hufxKhBMAkC9v67ET4tW+LIth24dOYMjDyjloiIiIiIqN4xUUANxrY/VwAA"
            "3NTuEEukEEskEIxGJ0dFRERERETUtHDpATU4Br0eUpkUMrnc2aEQERERERE1OUwUUINj0BsglcogkUoh"
            "EomcHQ4REREREVGTwkQBNTgmQYAgGCGVSSGVyZwdDhERERERUZPCPQqoQTLoDQgICUH3YYORn5ODjX8s"
            "cXZIRERERERETQITBdTgiCUSTP/gHUR27QIAKC4owLY/V0Cv0zk5MiIiIiIiItfHpQfU4AhGI/Rl15MC"
            "ai8v9Bo53IkRERERERERNR1MFFCDtGfteovXwydNcFIkRERERERETQsTBdQgJcQdR8aVJPPrVhHhiOzW"
            "xYkRERERERERNQ1MFFCDFXPzrIJ7JzopEiIiIiIioqaDiQJqsI7s2AlNUZH5daf+fdEiMMCJERERERER"
            "Ebk+JgqowdKX6bB/4xbza7FYjGGT7nFiRERERERERK6PiQJq0Pau+xdGo9H8esCdY+Du6enEiIiIiIiI"
            "iFwbEwXUoOXn5OJ4zF7za4WbEkMn3u28gIiIiIiIiFwcEwXU4G3/a5XF66ET7obCTemkaIiIiIiIiFwb"
            "EwXU4KUnXsGZQ4fNr9Venhgw9k4nRkREREREROS6mCigRmH7cstZBd2HDrKpnYB+wxwRDhERERERkcuS"
            "OjsAopq4fOYsLp8+CzcPNXau+hsHN2+1qZ2oKdORvO0fmATBwRESERERERG5hkadKBBLpWjVuz+aR7WH"
            "wsMLBm0p8hMvIXnfLuiKi2vVllfrUAR07w21fyAkCgWMujKUZGYg88RR5F1IqKOvgGrj14//h9KSEiiU"
            "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"
            "AREREREREaVMKBBAR2PTlO1EjQbWDDtsDgesTgfc7R0IBQIQBAEYG6UgCNAbjZBlGaKYuHzkZMKhEZis"
            "tusjBxQFChQ4srMhiiJsTidszumtHhGNRPBvf/TnaLx0eUbXsFwxUUBEREREREQLTpYkdaWH8cZ/+w8A"
            "v/ru03j5ez+AJcMeG5ngzIItwwa70zE6asEJq2NspEIGLHY7RFHESCAIg8kIQRhNMIxOaZjN0pJanQ6R"
            "cBg6vR6KoiAaSe+VG5goICIiIiIioiVNlmUMDXjhG/ACYjsgRyFM0FYQRVhsNgT9fkjRaNx2AUBHQxMU"
            "BbA67LBmZCQt1JhMNBKByWaBLMkYCYbSungiEwVERERERESUNhRZxvDgYNLtCoCXn/5h3HaNVgtLhh22"
            "jIzYqAWHA9aM2OgE2+iflgw7AkPD0OmNiEZiBRzTGRMFREREREREdNOSolH4+j3w9XumbKsoCtI8RwAA"
            "mFk1CCIiIiIiIiJKa0wUEBEREREREZGKiQIiIiIiIiIiUjFRQEREREREREQqJgqIiIiIiIiISMVEARER"
            "ERERERGpmCggIiIiIiIiIhUTBURERERERESkYqKAiIiIiIiIiFRMFBARERERERGRiokCIiIiIiIiIlIx"
            "UUBEREREREREKiYKiIiIiIiIiEjFRAERERERERERqZgoICIiIiIiIiIVEwVEREREREREpGKigIiIiIiI"
            "iIhUTBQQERERERERkYqJAiIiIiIiIiJSMVFARERERERERComCoiIiIiIiIhIxUQBEREREREREamYKCAi"
            "IiIiIiIiFRMFRERERERERKRiooCIiIiIiIiIVEwUEBEREREREZGKiQIiIiIiIiIiUjFRQEREREREREQq"
            "JgqIiIiIiIiISMVEARERERERERGpmCggIiIiIiIiIhUTBURERERERESkYqKAiIiIiIiIiFRMFBARERER"
            "ERGRSrvYFzAXolaLwh17kFW9GgZbBqKhILzNjWg7cgDh4eEZ9aUxGFG8ey8yK1dCZ7YgEvDDU1+LtqOH"
            "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"
            "v5a0iCERERHNDkcUEBER0bJSsGoNVt16J3RGIy6+8yaOvfjzxb4kIiKitMIaBURERERERESk4ogCIiIi"
            "IiIiIlIxUUBEREREREREKiYKiIiIiIiIiEjFRAERERERERERqZgoICIiIiIiIiIVEwVEREREREREpGKi"
            "gIiIiIiIiIhUTBQQERERERERkYqJAiIiIiIiIiJSMVFARERERERERComCoiIiIiIiIhIxUQBERERERER"
            "EamYKCAiIiIiIiIiFRMFRERERERERKT6/wH6FsLsq+mWXgAAAABJRU5ErkJgglBLAwQKAAAAAAAAACEA"
            "ytHHH0O8AABDvAAAFAAAAHBwdC9tZWRpYS9pbWFnZTQucG5niVBORw0KGgoAAAANSUhEUgAABAoAAAJH"
            "CAYAAAAZsrSkAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8v"
            "bWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAXEgAAFxIBZ5/SUgAAu69JREFUeJzs3Xd8Vud9///X"
            "Oeee2hsthgCJIYYZHniAt3G8d7zSlTZpOtM23WnTpE2aJm3T9pv8Oty0jfeM7Xhg4wHY4IEBGzPFEAgt"
            "hPa69zm/P27pBqGbKQkJ6f18PBzDOdc55zr3EQ7nfV/X5zKyC6c5iIiIiIiIiIgA5mh3QERERERERETG"
            "DgUFIiIiIiIiIpKgoEBEREREREREEhQUiIiIiIiIiEiCggIRERERERERSVBQICIiIiIiIiIJCgpERERE"
            "REREJEFBgYiIiIiIiIgkKCgQERERERERkQQFBSIiIiIiIiKSoKBARERERERERBIUFIiIiIiIiIhIgoIC"
            "EREREREREUlQUCAiIiIiIiIiCQoKRERkXHv4gXtZ/cpzPPzAvQO2L5hfyepXnuPR//7JOe3P6leeY/Ur"
            "zw3a/sPv/Q2rX3mOBfMrR/T6J/o8xqMTfdZjyaP//RNWv/IckwryR7srI+J8eAYiIjKYa7Q7ICIi40d+"
            "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"
            "PP3cizQebjrzGx8jenp6+cM/+Sa33LSSq1dczuTSElwui8bDTXz48Saeef7FpEucDuUzFRGR02dkF05T"
            "VRgRERERERERAVSjQERERERERESOoaBARERERERERBIUFIiIiIiIiIhIgoICEREREREREUlQUCAiIiIi"
            "IiIiCQoKRERERERERCRBQYGIiIiIiIiIJCgoEBEREREREZEEBQUiIiIiIiIikqCgQEREREREREQSFBSI"
            "iIiIiIiISIKCAhERERERERFJUFAgIiIiIiIiIgkKCkREREREREQkQUGBiIiIiIiIiCQoKBARERERERGR"
            "BAUFIiIiIiIiIpKgoEBEREREREREEhQUiIiIiIiIiEiCa7Q7cL7y5ZeCYeLEoqPdFREREREREZEEw3KB"
            "YxM8UntWx2tEwdkyTDBGuxOnyThfOipDpmc9cehZTwx6zhOHnvXEoWc9Meg5Txxj9VkbxN9Zz5JGFJyl"
            "/pEEoZb6Ue7JqZkeH3Y4ONrdkHNAz3ri0LOeGPScJw4964lDz3pi0HOeOMbqs/bmFg/peI0oEBERERER"
            "EZEEBQUiIiIiIiIikqCgQEREREREREQSFBSIiIiIiIiISIKCAhERERERERFJUFAgIiIiIiIiIgkKCkRE"
            "REREREQkQUGBiIiIiIiIiCQoKBARERERERGRBAUFIiIiIiIiIpKgoEBEREREREREEhQUiIiIiIiIiEiC"
            "ggIRERERERERSVBQICIiIiIiIiIJCgpEREREREREJME12h0QOZmyKaUsnDebnOxMotEYdY2H+XjTVrq6"
            "e87oPKmpKSyeP5fSkkJS/D5C4TAtre18um0XDY1NiXa/8Uv3nfQ8dQ2HefXNNYnfz6mYwdTJxeRkZeLz"
            "eYlEo3R29bCrah9V+w7gOE7S80wpLWLenArycrJxuSx6egM0NjWz4aPNRKLRM7q3sS41xc89t92Ix+Pm"
            "08938vHmrad9rMtlsWRhJdOnTcHv99HT00vVvgN8+vnOAZ9tWmoKD9x9y0nPtWvPftZt2AhA0aR8bll5"
            "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"
            "ZeXl3mP/+PPvVFnp0Fe/9Hl99UuflyS9tmiZ/rFoqSRp6YrVWrpidYPlvBcrVq/RgrmzNPKRYdq5dYO2"
            "vbNTcbGxemLKYzp/IUu9enaX2+32ed37Rz9QYqt02Ww2pXdop+e+9AWtWPw3Pf/vP9LfX1tiwEdSfyFX"
            "FLQbPkr22DidWLlYHj8bSwSUySRzWHjDvkcAmKxhd9y10mK1avC4RzV+9gy1aNNakjR+3mzt3bxN7o/9"
            "/rkkbV25RpKaxMfdHH3SWiO0sNbNA+vcfLDWzQdrHVpMJpPMNrs8cstkC5M+/Gm1yWKr74lrz/fxw9b7"
            "PO8dzlcft8/3+roN+sLX/lWS1Dq1pb7x1S/q2Wfm69Xf/ErPfPnr3vmtuzz4ieepj7JKhyQpNi7W7/li"
            "YmNqf2Jvtcn0CeepkTR9/uf1b998TlMey9Dnn1mgGzdz9etX/0/nsy7pr7/7H+UXFtd5j4+udY2kc5ey"
            "9c0XfqjWbVrrpR//UJu37dT1m7W3ad8uTmITElVUXFznvWPj4yVJJeUVn/h7YjKZZLLZZJJZ5rDwO18h"
            "YTJ5/zzejyZXFHSa8LjPsYLzZ1V44ayiU1sptU9/3Tr5gUqyfe/7uB9HX/uT3+P2pNpvqt3VjoC8T0My"
            "yzenzW7X8McnauycmUpMSakzlpTaUv1HDNWBTVuCmBKB4G+tEZpY6+aBdW4+WOvmg7UOLSaTSW5n7dPV"
            "PM7qOt+0eZy1m+rlX7/xqc9beCvP+/qPclU57ut8Jfn5fs9XH97zud3e/76ana3nX/iB0lq30uRJE/RE"
            "xjiteXPDPZ2vPnsUXDxfu2lgh7ZpPh+nyWRS27Q2uph16Z5+D8qd1XrxJy/pxZ+8VOf4d771L5KkY8eO"
            "+ZzH33l37tqjMaMeUd8HeupaTu1GiBcvZumhB3urQ5tWOpJ3q8789mltaudcuPjJOU0meZxOeVT7/5I7"
            "FgX1KAmkJlgUpPTybaKqSopVeOGs4tM7y2Q2KzK5hXo+Na/OnIjEJElSm0HDlNL7IRVduqhrAbg1oakJ"
            "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"
            "4eP65bc/1Lx1G73YrpMq13xQk6d+Z7WqVqWSjtoSP17eXjJr3VS2btFcIenTqWPXnmrf5TU91KCxda05"
            "d2Px9PRU7lw549e0SQTakx4eHh4qkD+vFixeKh8fXz3V6AktmPOTBg96Q+nTpbuxG17flQCNEEAAAQQQ"
            "QAABBBBAAAEEEEDA9QWI0F0FPN018MSOe82wvkqJcj9xbNmyTRkzZrDeKqhmjary9PDU8pWrb9tl7pw5"
            "dPbcee3bf8BqY34B/OkXX+v1/m9ZpVvPvtYv3+3Hq1avu+XbS6W1/SK9dctm6tiujVXMWw9ZHdo2N/Zt"
            "O0zQ99dDP9Twrq/dsZyxJQP+q/PE6ue/7mPOr9+4SVFR0dbbYeW3JRAK5s8nk9Qw525VsmTJrOiYGJl/"
            "vW/Ot2zeVFUqVbQ+f6J8uTIqWCC/rl69qoVLlmnr9h2Kjo4xzazi4eEh80SHdXCHjenb3MPc6w7N7nhq"
            "7+J5WvH5OKuc278nvm1S18ff6IYXBWymPrbEzdLlK26o/ftlqZIltGr1Wp06ddpmWcE6UaRwQevJCPMW"
            "aFbFbTb+qVLJvH2WfU0/17K5ChTIF9/aPClTuGAB2zxEadHiZRo74TN9/c00Xbp0Ob4NLxBAAAEEEEAA"
            "gWQT4EYIIIAAAggggAACCDioAAmLRJqYi7u2KCXK/Qx/6YqVioqOVhnzVlG2X9aeP3/+tv+i336f67Zf"
            "kttfm1/Kdu34iurUrP6v8vCDda3Pw7jV+/qfO39B/Qa8rbr1n7SK+Zfu9j5jbP1fv37dfpig/eGdu7Vn"
            "w6Y7lshr1/6z78Tq5z9vZGuwcdMWmbfPKl6siCqULytfX1/rc0Jsp277bazM23CZBuYpjWUrVsnXx1eb"
            "Nm/V9h07rQTIjB9m6dDhI7oeG2c65uOJ2rV7r2JjY5XVlvQw/+p/718JKNPPjcX0be5xY50zv86SKZPN"
            "JEoXLlz8Vxjm7aDMW2OtWLVGm2yJvJLFi1lP+aQNDpZZr/+64B8VYbbk0KixH1vr2azrZ1r+T2ZOTTNj"
            "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"
            "24s5thd7ndnb68zeHNuLObYXe53Z2+vM3hzbizm2F3ud2dvrzN4c24s5thd7ndnb68zeHNuLObYXe53Z"
            "2+vM3hzbizm2F3ud2dvrzN4c24s5thd7ndnb68zeHNuLObYXe53Z2+vM3hzbizm2F3ud2dvrzN4c24s5"
            "thd7ndnb68zeHNuLObYXe53Z2+vM3hzbizm2F3ud2dvrzN4c24s5thd7XXan/HMypf58dsL72n6+sc+5"
            "2cfPO/Xx/601LvZybz53uR7sndv28f2n1P/vmPdEmHf+3mVbytZ3/HpmXbGubCuC9WBDsH3jYEOwfSfU"
            "IV2R0rar//v7/Pb1VoIiIGsOpStcUpcP7pVvUBpFnD+ryEsXrHJi5SKlypBZp9Ytt46v2ZIF186dsdVl"
            "ir9B4eZtFZg9l3I98Lj8bW0LPdNG3gGB8edvfHFh5xZtn/SRVr3dXRd2bVGRFq/IfIaFaWPeysokFi7s"
            "2WYObyrntm2Qedunos93VqEmL8rTy/um8/88MMmXf9a50jEJiwTM5rHFs2UvN15urzN76uOMcMDB/P/B"
            "XlgPrrseIq9cunF64/+MNHN/4wlzbC+JXx97y/ua+9jvafbm2F7MsVWW/GGvsvZW3V9/1lsVf21uXZ9S"
            "91XC47XF9ldI1u7WcdG/hWPb4GNDsH3jYEOwfbutg1P+OZlSfz474X3570L8f1Nt/zeP/06c/7+zHuyg"
            "iePJ30/wnG0nsPasK4sh/s8w4xFXE7c1x/YSVxO3tdeZfVxN3NYc20tcTdzWXmf2cTVxW3NsL3E1cVt7"
            "ndnH1cRtpWX+DQAAEABJREFUzbG9xNXEbe11Zh9XE7c1x/YSVxO3tdeZfVxN3NYc20tcTdzWXmf2cTVx"
            "W3NsL3E1cVt7ndnH1cRtzbG9xNXEbe11Zh9XE7c1x/YSVxO3tdeZfVxN3NYc20tcTdzWXmf2cTVxW3Ns"
            "L3E1cVt7ndnH1cRtzbG9xNXEbe11Zh9XE7c1x/YSVxO3tdeZfVxN3NYc20tcTdzWXmf2cTVxW3NsL3E1"
            "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"
            "/ozT2qpypUoqEROjqKgol2ajSwb11+nt2uqZ515SnwGX6qef5+jAwQOqVrWKSpeO1Sefz9SDj45xw39e"
            "nWCrpBoyy3fRgAu0fsNGV+6YZ5736tZUFw3o59Zv2qShTmzaWHGlS6tokWiXxgiBEBWg2QgggAACCCAQ"
            "BAIEB4KgE2kCAsEg8MH0T7Rp8xad0e5UlSxZQmecdop+/OkXbdi4yTXvsaee1fRPPtfBQ4fcvI0s36lt"
            "T9a338/Wl19/p30JCRo38U0tW/6XIiMilJQkV+Yvc36XDUuWLrfVUg2Z5Rv9+NOy7Vq5Vv7WrdtVr04t"
            "t/633/+km267J7l+LpERAkErQMMQQAABBBBAINgFCA4Eew/TPgQCRODw4cP6YuYstWjWRN27dvZO7JP0"
            "0adfZFr7KpUqqWjRotrgXd2/9OIB7rGEunVqu3WKFCmiyMgItTu1jf49ZrRuuPYq2aMKbmGKUVbzWSAi"
            "witv3fqNKdZmEoEgEqApCCCAAAIIIBDSAgQHQrr7aTwC/iNQoUJ5bd6yVQkJierfp6eWLl+hil5aMe/k"
            "P6Na2gm7BQe6ntvRnfjXrVNLD913p9q2bqWdu3a5Owi++2G2pn/8uVqf1EK33XRd8jsDfGVmNV+ns85U"
            "yRIlNPfoYw6+9flEIJAEqCsCCCCAAAIIIJCRAMGBjGRIRwCBAhWoVaOa2pzc0gsQbFHi/v1u2ye3aqHY"
            "2FJuOqOR3XHw1uSpeuTJZ3T7PQ9o586dOu2U1rLHER545ElN/fAjfTbjS73nfVauXCnVywytzKzkq1e3"
            "tnp276IffvxZP/z0s63GgIC/ClAvBBBAAAEEEEAgRwIEB3LExkoIIJDXAnXr1NapbVurQf0T3EsHW7Vs"
            "rrYnn6RyZctkuCk7sd+3b5/KlolzeSxQsGPnLpVOJ6Bw+HCSwsLCFBERocz+pM1XOjZWI4ZfoV2792ji"
            "m5MyW5VlCBSQAJtBAAEEEEAAAQTyXoDgQN6bUiICCORA4K1J7+nifw3TO1Pe18ZNm3X3fQ/rymtv1rwF"
            "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"
            "UqufcjHoN64f2otnjfovrU0pkYUKcuVOEmdsYoKRmZKsTDVOHl541ayPo7snj6If5FtO3szSigb9huLf"
            "uBXRt4M5sHIx6qf5LxqUNipavf0eZf2rEHLqMIdWfUdmppqIyxdQGBvhVrk6CZH3SYi8h5mlFZVadqBq"
            "685U79iTwLbdcXArT2TINYK+nZsv2537uWX1T/EqkYk2/zp+DZqhMDVFaWWDe5XqWNjZUy6wNid+XUVG"
            "WipXDwShzsggIfI+F3dtJSk2mst7t5OamEDpCvlXdQy/dJZygbW5efwQJzas4mlGBiZKJaV9A3Ap74Nv"
            "g+ZAFqe3/kJ6UiKB7bpjX9YdpZVViU3oyysSM7kT2hqZmqG0tKa0T0VK+1TE2dOH1MQErcf+K9RrgpGJ"
            "aYET42akpaK0VuFRqTqmFpZc3rs937xI8Q/ucumP3zm/YxMhJw/z9HF6Ttm27BWrjuzTetTzcWoyDu7l"
            "KeVZgVunj5Kpfqp55C/qdjDlq9XBq1YDwi+f5+GdW1oT81o5OuNeuRoRl85iam6Bpb2jZqVAa0dn3CtX"
            "51H0A60VBM1tVBibmPA4Z8SkjXMpKrfqRNjZE9zLs1DCi3oV4sajak3sSrmCATiX89HEjbGpKY/y9BkK"
            "W9AgV1JMJOWq1cG2jBtpj+I58/t6rcmTUxPiuLJvJ+e2b+L6kb2kJSaQmhDH1f27OLd9I1f27dQsUkDO"
            "6EZza1tcAwK5d+0ij1OSsbRzoFa3fty9ch5zaxXhl89xcuPPOHv6aE30nHehAgMDQ6wcnDm/YxPlq9ch"
            "9u4dLu/dTnJcLMFH93Pn/KkC+0IGBoZU69CdpxlPuBi09ZWZCDot6RGGCgVl/CrjWbMBTuW8sXUpQ0Cz"
            "dtTu1o+yAVWJiQildIWKlKtaizvnT7FhxkStFXrdK1XHxNyCWyePkBgdiZW9A6V8AvCu0wi7su44uJcn"
            "sG1XAtt1pbRvQLF+u5mZmfjUaUR0WAibZ0/m8t7t2JVxw76sO6HnTlDWtxKqUqXJVKsJPrpX096oMzIo"
            "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"
            "hBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTV"
            "hBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTV"
            "hBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQ00tLqtWoUYO5c+cya/Zs3Nzc"
            "dTe/NDNmzmTGzJm6xSXCx8ebFStX0qt3L91N/zkqW1u+XvQ1o8eM0d0kxL9a69atWbpsGd8tXYq1tbXu"
            "ZiGEKFSpUqV4b+JENmzcSJ++fXU3CyHEP5q0YUIIkZ+BhcolS7fwRXTt2hVfX19mz56t+Xd8QgLNmjbj"
            "wsULeHl5kZGRwfx583R3fa7uPXpQtWpVPp8zh8TERN3N+Pn5MWXKB6xf/wtbt/6muxlykmoAH37wge4m"
            "LZMnT+b69ets2rSpSPuobG2ZNWsWt2/d4osvvtCUzZ49m8zMTCZPmqT5zAqFgs/mfEbp0mWYNvUjgoNv"
            "olAoqF69Op06d6Za9eqcPXMm3/upbG156623aNSoIebmFqjVaiIfPODHH39k9+7dWnV1tWrVijeHDMHR"
            "wYGsrCzCw8NZvGgRp0+f/svqVawYwOTJk9j621bW/bJOa3/x13OwdyAmNka3WBSTUqnk/fcnUat2LQ4d"
            "PMiaNT8RHn5Ht5oWla0tjRo1omPHjpQpW5bvly1jw6+/atWZO3cu1apX1yoDWL9uHYsXL9YtBmD0mDF0"
            "6tSJ48eP52snXsSrFjO9eveif78BrPxhJZs2btSUN2/enLFjxxH0RxBfLlwIRfiuFAoFgwcPpkXLltjb"
            "22NgYEBSYhLbt29j2bJlqNVqTV03N3cmvDcBPz8/FAoFjx49YvOmTfzwww+aOgWpXKUKw4YNo0KFChgZ"
            "GfE44zHXrlxl/vwFmljr3qMHw4a9hYmJse7uABw5ckQrJt544w26dO2KjY0NarWaa9eu8cXnXzw3dvXx"
            "qsVNUehzfs/VqHEjRo0eTVpKGmt/WcuunTu14oZitgcAlSpVol379jSo34DIqEiGDhmitd3NzZ2hbw2l"
            "WmCgpv9w/959ln2/lIMHDmrq+fn58cnMGdjb2Wvtn5KSzPTp0zl18pRWeXH9F2Mml76x06xZc15/4zXK"
            "lnVFoVCQmprCyRMnWfjllyTEx2v2fV4fMi8PDw9mf/YZLs7OupsAiIyKYvL77xMWFoaLszMtW7WidZvW"
            "2Fir+PiT/HFQlM/4ov7LMVOSint+oohtWI0aNRgydCienp6a89ipU6dY/PUiHjx4oHe9FyExU/Ke1f9U"
            "2doyfvx4atWuhamJKampKfwR9AdffvmlJk707cPk9SLXwfqQuCkZxe1/du3WjX79+2FvZ1/ofgqFgqFD"
            "h2Wfl2xsyMrKIjY2lq1btrB69WqteiPfeYfmzZpjZW1FVlYWD2Ni+PGHH9i+bZum3otysHco+ZFqgYHV"
            "KFO2jObfDRo0oEWLFgQH36BXr14EVAzgyuXLWvsUVdSDSPz9/encubPuJgBUKhXGpiY8iIzU3aS38p6e"
            "lCpVSre4UN27d8fCwoI1a9ZoyhLi45k/by421tYMeO01TXm/fv3wKFeOb75ZrOk0jRo9ismTJwMQHxen"
            "qZtLqVTy8ccfU6tWLZYtXUa7tm15c9CbxMXH8/aIEdStW1d3F42GjRoyYuRI7kaEM2jgIIa/9RZZmZmM"
            "f/ddfHy8/7J6V65c5sDBA7Rv316vv6UQ/zQKhYIPPvyQCn6+fPS//zF79uznnhSsra357NPP6N2rF/fu"
            "3uXJ4wzdKgBY29iwZ88fNG3SROtV2AV0w0YNqVuvHsnJybqbhI51v6xjz949dOvaDQ8PD8jplPUfMIAL"
            "Fy+w6OuvoYjfVb9+/Wjbti27d++me/fudOvWjR07dtC5c2cGDhqkqaeytWXylMk4OTszY8YM2rdrx8kT"
            "J+jdu/czRzHXqFGDaVOnYWBgwPvvv0/TJk34Ys7nuLq78f6k97VGRaYkJzNx4nta8fL111+T+CiRY8eO"
            "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"
            "ytK5X+n9+N8/icTNy/dvb28kZl6+f3t7IzEj9CUxI4pD4kboS5JqQrzC5KQg9CUxI4pD4kboS2JG6Eti"
            "RuhLYkYUh8SN0Je9nV3JzKkmhBBCCCGEEEIIIcR/iSTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQ"
            "kyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQ"
            "kyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQ"
            "kyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQk4F7ed8s3UIhxL+fna0dcfFxusVCFEpiRhSHxI3Q"
            "l8SM0JfEjNCXxIwoDokboS87WzsMLKwdJKkmhBBCCCGEEEIIIYQe5PFPIYQQQgghhBBCCCH0JEk1IYQQ"
            "QgghhBBCCCH09H9iCsPtwT8wdAAAAABJRU5ErkJgglBLAwQKAAAAAAAAACEA4SyDZORmAADkZgAAFAAA"
            "AHBwdC9tZWRpYS9pbWFnZTgucG5niVBORw0KGgoAAAANSUhEUgAAAokAAAF2CAYAAAAC+wJwAAAAOnRF"
            "WHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcv"
            "wVt1zgAAAAlwSFlzAAASdAAAEnQB3mYfeAAAZlBJREFUeJzt3Xd4lFXax/HvlPQGqaRQQgst9C5FwYKK"
            "vbe1r73XV9d1bVhXV7Hr2guuHVFRpEgn1NBCCElII5X0npnJ+0dkJCTEJCSZSfL7XJeXM0+b+5mbubg5"
            "5znnGHr26leLiIiIiMhhjI4OQEREREScj4pEEREREWlARaKIiIiINKAiUUREREQaUJEoIiIiIg2oSBQR"
            "ERGRBlQkioiIiEgDKhJFREREpAEViSIiIiLSgIpEEREREWlARaKIiIiINGB2dAAiIk05efbxnHXGqUSE"
            "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"
            "Lswm02h1VQghBiRJohBCjIKOnoIUAJfHA4BBrwfA4XBiNBqGfC+jwUC3wzGyHRRCiKuQJFEIIcZYfWMT"
            "iZb4oGNXDjFfLjHRQn1D02h3SwghgkiSKIQQY6yishqTyURCfFzgWGdnF0DQMQBFUciZlElZecUY9lAI"
            "ISRJFEKIMedyuSg+XcKcwhmBY23tVo4cO8nDD97Dxz/0bhYvmAPAlNxJOBwOKiqlslkIMbaU+NScwcc4"
            "hBBCjAqj0cC7H3mIp579D3Z7x6DtHnv7g+x6vYiq6tox7J0QQkiSKIQQQgghBiDDzUIIIYQQoh9JEoUQ"
            "QgghRD+SJAohhBBCiH4kSRRCCCGEEP1IkiiEEEIIIfqRJFEIIYQQQvQjSaIQQgghhOhHkkQhhBBCCNGP"
            "JIlCCCGEEKIfSRKFEEIIIUQ/kiQKIYQQQoh+JEkUQgghhBD9SJIohBBCCCH6+f9/YpmwgYtXyQAAAABJ"
            "RU5ErkJgglBLAwQUAAYACAAAACEA3U+q8nkzAAA2MgIAIQAAAHBwdC9jaGFuZ2VzSW5mb3MvY2hhbmdl"
            "c0luZm8xLnhtbOydW49bR5Kg3xeY/yDoPY4zIjPyQoxnkNdFA92Yh579AbWSbAtjS4Kk7t7BYv/7Rh6y"
            "itRhiUWyyOpDkQZcLlkqKvkxT9wv//rv/+eP31/9/d3nL+8/fvj5NQ7q9at3H958fPv+w68/v/5f/9nA"
            "v3715evdh7d3v3/88O7n1//97svrf/+3f/kf//rpzeLNb7/+6cMvH1/JS3z4srj7+fVvX79+Wvz005c3"
            "v7374+7L8PHTuw/ye798/PzH3Vf55edff3r7+e4f8tJ//P4TKWV/+uPu/YfXq5//vM/Pf/zll/dv3pWP"
            "b/72x7sPX5cv8vnd73df5fhffnv/6cv9q9292Xq5P96/+fzxy8dfvg5vPv6xeqX7A8kroR6P85P85h/y"
            "fu9f6NOeL/Tp4z/eff708f14qulr/VvH9fbjm/zbr3/+8nX8lcArd1/vXn24+0Ow/vbq6+tXf/vy7vOf"
            "3v78+s6qX+6U9nznAv/vX9zrV58+f/z7+7fL3/3z+7+/+5Mc783v/Vf/txRbqnUBtC8IRpODqFMAn0ok"
            "W0oshv7f6582TvBK/uqfX//tw9uPrz6/ky9v/vbl61/f/f7q7u3bv/7+9tXbd7/3//zxcfzVl9/f/sfn"
            "8Vd/kTf0l7svX999Xr6dl30Dr95+/fk1KbKgNJD+T6KF5oXWA3m5sn//+bX3Orx+dffma39RYl6/5b/8"
            "1z3z8Rfy/3+a/oa8ywcwHcOnkcKnEcKnDqZ/N4e3jbxQZmHcEJRdvm1n1PptK3Zu9cblLW2/8fv//+pN"
            "/+Pyg/KrJS/bf6xz+ebnvnx6wLIEcPdPB0AL8gtWA7MbAVhtzQaA8e3f9ZMf/O5XP/XqvfwvlmN+fjeK"
            "lfGwavUPPPLl/p/xKftp8+8eeXaEU5hyu2YBUy+MX2B4uE2Og32AaZw/DUx7LTDl0dRhUBiWN1Ne4vQw"
            "3QQmRTlb1h40IUFlxVA9V8gqs7bVOGvyATDXsu6fD3SUdZoHpZZAvfXr2/kg656P1E+QuppaSlyASkwQ"
            "GyXIPjcgG2zIFjkmvFCk4wNPSh54un/gTwQxTCDKOZRpGKEl06DImaBob4GZDYfksVh3yffySR38fKSo"
            "pkwVNcc5QIkuQWulAAbrhGlylBJpp3c/6ysF/8B3w+xZStF/uoGjeMG4MDRoXMK1ivSGFN3fujEB5aTG"
            "dwdmh5WzA8mHd//oduCMsISB0d9j2VQu+2PB4E0go+RnjuXyXQu5/5t///zX7ie963/2z3f//fFvX+fi"
            "L6jxoQ1m9dA6mmrnw83mEfwus3kuxgktjMh+Hoxfvn3L2j/bONmUc6PEorNberNRArhguVI0WM2rB9LQ"
            "aSDq80Lsz+qMQO71YJ4eahJo2eoI5HUATMlDIO/BlCyfpA5FtXCBPggtlBYHeWC7tO8sqrX2JDwNS3P+"
            "Czq3y2nUKsrk3JkgMmOLPlZw0SKIq6OhFrGco7w5jJ444CH28sxEpfjGOCCtQjaKTwTxzGGa+cS8uiPs"
            "B2+Xt1Ce6nXMC605DcxpZOEG8xkwt3y3G83n0DyzTXRlNM+swK+M5k0JnZLmmZMFV0bzptNPSXOaKLjR"
            "fE6s6KbTT0nzzDp9RgEO13OC4g4vaVLXGCePY15JyvqFaJ5ZC10ZzTNroSujOc1b32g+J8p+5qjHRu7i"
            "1cfPs9DvPTPLCzKD7271GN3U65C7U+pEZHFCNlNsQVEGn62G4o0HhSZDcmycQRML8+WT1XJtB3NfUKX0"
            "OTJDW0lLUwK5FsAlwcs1aVAFPbhogtXRkwlHVK7MheoqE6wGs8q3WdbuDPd1SpUjRYyxgInaQavsoEZl"
            "wBG6ogwpVePl31e7UH7A/rAu7yufgezUb0LmhtE7MMQebFYFQq0BmsLsFZeKrl422X5n1ULjgPdFQcat"
            "K67OR1bnpFPVFtA0C96UBOiDAydMsSSXQ1vW1t/u7BNkt7xTl0JiQsDiPZSuwkjkA6Rkqq+KQ2nmdmeP"
            "IWt0xhKCAUw+Q47Oi1wwBqzOWsjGoLK+cLJj6Q3ZQZTG/Z1d18Kdjuw0rp+SiyVEUV5FRGxszsh32YEv"
            "xCUbubUp3cjuQ3YaXUEtD39mcQvYVqi6IURC0WDGFq1zo+QOKca5YrLTSItvWQ6RCWJyFQxb8RdS1uAp"
            "U1CJm7OHdgfMUM6+hD07jbpon2PB7hZY1cCz0WDJITRdo9WGc40X7n8tNZixA/Za4aUGO1WJ4zTq4hxr"
            "auJuFSoKBKEHjM7KMbX2OlLFeOlaS2jiQtmB7apgVNz0E5WSbfUJuIZVFXnqrSGwXjO01iL4oNCJ5WXR"
            "H2q3ziWbYhZE3V6lVQOLp24B3WdTuvF6GqQvUIR7bUhfpmx0Ns97L3ns0tOtutesteuK+1WD8gmoTq1U"
            "rYLOVh520Yf9sVfYpQBCaSJFVRX56p+UpL9+nmUA24imfygTt6jWjupDZbOc/K9HAV394OOxVmWzfHhU"
            "QUwLBLatgRKbH7gwCutAvqxv6uYROtYlzPEMn96v++9nqKL2LcWXt3EM4vHHlpd2C/AhomD913e8S6aP"
            "4p2TNNhXyD6f7bN6cB5ne1E9cn6wPUW67JFbewIHtQ6SChQ8Gbu71evgHrn7B36+fXK9QxgX5AbmVYew"
            "+WYkwHFtcqPkeKJNbjb2EPUeaWUGz6uQKHM4SllvJqheorNrPgDZLVQYTJ/MMQLsTbhrWXdc4m6LJp85"
            "dT8zY1KJdBtCz7Avjcnnh+m3iPop0eKdb01O21yRL61qyFwq+JqtGJPimbtLD9O/DNkwddE5YTMtGWja"
            "VaAcKrgQMujWisKsg/U3snuRndqS2mU5rU4QuRXIzSgIITjwLXgMWRflLz3k+UJkp5akCc3p5AWlKgWi"
            "6EfAEBNkbeU3jJM3feHFPC9FdqsbmQhrFWmArTTgag246CskRkabc7FqdwLkwiz00ONrSwt9cwrB/sYl"
            "KYsWtQ27jczd804eM81nYYPT/bCPbuIsDahNP/w4G3y0YS/IBjd9CoznlSvH5riW4U3L/SW6XGc072PM"
            "N9qHK6TwuYMAtlheTSXyC7A8hzOzGRCbEUzV22F4NVHBGvcIzAPDYPc0N8JgZ48wzgoohyF0720J9JGR"
            "H88G+qwu14PDihccPlPYJ2pSn7uyNCCfHT6zo9n5xHDWH9qEXhLYpcaTKqpi6imapiGrYqFg8cDKW/Ka"
            "MsabP70P2KlOp2qqaqFAkuPJOdmB0+JY14AxZazO50Or4a8T7FTBk3aOMTlAS0EOqz30Bgf5zunI1Eq5"
            "9NaYFwK7NWpFXpmt6eUvSWRBtA0h2+jBWxtM8pzTLWq5H9lpbC1yS6lQguhSgtrbj3r3AbhedJCJdTJH"
            "xyluKn/zsxjttqtW+Zt+1aMq3xRtUkAGSs1CrCFAYzKi8jEwylup6tJrBl8G7FTlZ0VF5d5BZIKXRz0S"
            "IJUCSs6JJZlq9KWXY78M2KnKD6J8kktZJKerUFwvyY4YIWvy3vocqNzA7gN2S+W3mFOqsUJqPkCIJQOL"
            "cADbalFOk71ZqXuSnar8lGzN6BmC1QGq1SJqbWqQk47UsvYUdncX3lT+np/FWFt/1Sp/s7vgUZVvlRIJ"
            "qhqg8RmotxMHhxVyFlO/maarufSmi5cBO1X5lSjZxAl8w9JPzL2lxYIhnVoszZC72VL7gN1qbGkciw8s"
            "2h4TJJ8TOIzdZk2VU2Ctni7HvoF9TOXn4HJr7EAbqhCpiIPfC5ScSxWJk92j0P1G9jGVX5iN3FEP5BUB"
            "uSYCNoiLJbo/VFbGZLO70fVHVPlmQdiTgPd5K099KskzVb6+epU/Etg56KY2pTNWAp3F7BRhSsBevisu"
            "6WSM96Tt7Tnfh+y0nivE6HJxGkIVVV9bbEDi3EOylEvOBf1BI+evmOxWYTcVJhOSeKKRIHIc81IZMjvx"
            "T2NVCS99VMALkZ3m8cXOr6RFLcmdFZ7ZeFH4WrSUNkpOHZJXN7J7kZ0m9NHHpnPREI2Yq62Kb5UjaejW"
            "lHUNjQ27BzTdtP6en8XY9n0Ju1b1wtiF4kF8lNW2QLduqDq2M/ibibcvsTXqh3+2t4BujQFhjbmYVkDF"
            "mqFGXfvEwAjELpRUo0m53aTmPmSngZPWEJFDBULxQrNxDBFDAc+MjrMWtDcvdC+y08iJFguqZ+zB9ISz"
            "KHrx71V1ULLc15KxyZFvZPcguzUehErUptfwNCxiOWEfxUji5CtKJTiMVL2/kd2H7Javikqed+WhmhaA"
            "q0aIiRWI6NXeI6Wodt/Zmw2152cxmh5XHTnZNL4en1lTbXXVYITE2YNFamCrfEemFMfel8sfsvhCZKde"
            "qDXBoqkKEpGQlb9dnvgcICRLvuYUar70kasvRHbqhbrWatfu4MSjBx20ljsbEtjELHc2hBhu0b69yE6H"
            "LKbmTImKwGdCqK4USOLbQ1SoUZsYxWK9kd2H7NS7ql65lpFAo6vCMysQrypB8DV5Haq2YXdW+qb19/ws"
            "HtlTfmVafySw0wYNWKJLYn7qohM0DrarJfFIvWq2Kq+VutVI7EV2GpfKyQSN2YK49qrXR2RQLjlQnhv7"
            "zCq6mwTdi+zUUmXt5HaWPqVOoBoXFPTUlJxWuJKR36u3Xoi9yG6tldaZassOuIkVhVEXMKl50FR8UD1P"
            "xbeS073ITi3VSNHWyNhTJSy63rPc2T4T3JFyJpQaypPe1Rz7SfWCwwJpCH3pbg/6B7sO+h/ZT/qAc91P"
            "+qw+/Itq0B2BajtQfziXQNeW0OmAPqsZ/3oadM9ilz4yG/DK7NKnR3xj8sU6OaiiHCCVEICVY1FMIiyJ"
            "bXV8i0btQ3Yrnu+cq2ydB9E5Tcj207gqZqq31LRiavZm8e9FdupLeetiTFwgGGWAe0U09Vli1fR9FTm2"
            "xBe+tOqlyE59Ka2jGJ8kB8WixeK3GYJjDa4oF5MNOdEtb7oX2a2of5ZjEhHI7aygG1UovbLfGvEFVKUU"
            "n2iF2q315zJLbImXeBjHnSzxrueeHKvID5shNjscaIfQZx4vcTx7Zph9ZGbYFeNwj8xhuWYcj/SoXzOO"
            "R/r3rhnHI70N14zjkaLPa8bxSP3GNeN4JLF1zTieu2uhz7ydfQs5hoXWA62WVTjq8dZ1bfV4I/Yjh1Yb"
            "uUKqj8J+QPjkyoV57UbTco/0wH2nT79EemNXKqlNp+dYFDsLzsVVMiaKc1+tAucKgcfAoGOlUMR78uGQ"
            "keHzGaKruO9h0LSaDIknJ/kCa9JmuDHZjXeVV9st+kKLY9z64wgn5xOpyJDYJPBaJUiVDBj0xoVoSZlD"
            "wqhXdFdfYP/cld/VreRzM74kRHCtFShJLmysIlcza4NoQw7+kGa9K7qrz5q2e7urRxCmYG2NIYMYZBqc"
            "DxF0MA1ai0EX7YjKoWtDZmZhYR+77fxqqZXezOqf/P5Oi099DHYcIKXZGiAV7DLmrwmLLp5qOHik3LxC"
            "/egXqIdgl6F+h71c5IFub5s8Ld+tElTCWpLDvjpZgQmJwdXYoE/otZyQS7r0RNVS6trB36dTUK1nbp1e"
            "PmxPSEJDHhWwpwQl1AKqegKKKTWyqfqDCtOvSJdtTe75YdfenR/lmW3Ya0I5VVE3lEejPPNOnWtCeeaV"
            "OleEkp61ZPqGchPlmSd0zNTcdAs0A/ZdTx0r4XHVO3sinqojU9lh7ydjiwmUxwLY+rQjxf0NYav5Qu1N"
            "u2A/4L0VL2b1qVFejww9N8r+yjeUp0F5Ncu/z4/yarwgEpRuIytqN1blnT6Xdz0e0dlv6PV4ROdGudXt"
            "cEN5NMrr8YjOjvKmzU+G8uXy8nPJFT1d+HXKROdUgFo5THNZgw/YQPcJxSa6AsZ7bwLlpvjShxfRAuXa"
            "8mAenPe+R/x8ueStvrFsqrdVQ2y1QgslQFDNgtMKU0bMvl16R95LI94KQSUbXc4KGpsGeSwr8U6+s4U4"
            "52KCvvQhhi+NeKrTyNoWrY5QjbOgXZ8NHSoB9XcT0LG++D1wL414quuca6plTlD7MrjibABRBw58jlSi"
            "oRwvvsGUem2E4qHTXCK2a8TmG0YnQbw1pyM6Rcp60W/VCuIcIObaC3tKJZtScvSkOfHr5zkO5xY7gpTY"
            "Eas5HRjMY3aEnP2vz2O7eoVlcHXLs7XZWk99wYFDYNsaKOICXBizbYF8WePdPEsnvOS6hXhO5trSAraD"
            "uS+c2jTXHqZ3nBby1g6Uk0D+wTooiBYUFsoN3C3cPsnDbQiW++n0+3VQkEbvqI9B3PhkHml+nfNQD8QF"
            "q4F7902/pr4P4DhCkX0fxa4OCgqeWt+2LecTn8IGBVklI8YZqyxHVyrRAXpsLqN7aGH8gvRg+46sTpXd"
            "OalODbAoP9Coj5PsSw5TYOy7eTxgCYghVMVPrNu+UX2kKt34EpQX/1eTZpAHhcQJTrUv56Mg70i84iNt"
            "rjnQ7WLRLTQO5O/FIm9ueT0HUfG4TI1swHttgRI2EDMrg2HPXtWArA4hOi/1/6J3dWq8BrFPlcsMTYuT"
            "gEr8MOpzp7lop9lRi/VAstdIdZrYEluqWfkCmQUt+2AAI0UI1YvjVX004dDVEldsC0w9AmzBBw6xl6Ej"
            "WIcaIsbUh3lGDiiEyw+wXsL24iDXowVj03MXtmcjvLV0JhouxThIxonQDb1XJessJpdH44pr9APEvliP"
            "iQdetZWzX3u3QZ9ao02H1dki9zZ6BM+UIBFrQBdQJESqRsSw8ebSx9BT77diN/jVnE/LYaPnenNL7CkA"
            "bzUBpEqBc0igkhgOTicNdRwFFpqN1kftfoAQee9oC4PrzezLO3xOOdw/x8mMtaaaMRFc0gp08w6aYQti"
            "pBVWqjYdL3/G2ouquq2mAYNks/Pi8qoqIiKxhyTeLpSGSulsW3pi4c+jlsRM0PZBqW7gPrtvzFP2iur7"
            "8IoIwZPDnaq5rILRQTEYymJIqGJFRNQGNYtbrFzUmS9/a93Lhm2mUrgYa4zVbZxaDYwlgzJiCWNKIXFT"
            "aOohcZt5Ita9lnvATrbfY31k/mFPxNPYba4t5t71ijE4EcXEIh3kbH0JsKeAqTy9x+7Xz3effnv/pn0W"
            "hFtuh/z7P5e/P5f4w9Nh2TFMvX5Pz0L/7Qs97kYboxqN7yJWcaNjAddDPqqgCfIIeJvXmxseOdgyjP7N"
            "Z7A1bny24sXgEPyqgtRv9NSv7/4es8d3fgLrIeS4lab3tqoqYsW0aqAlsaGrYRJRgzXFloOza0N6r7Hu"
            "M+S8jLkpP5iHO9+/OSfn6Q3XFLPI6ggNk2jIUsX97nuzAjajagol4VqQHzztvWeD5jJwjBZ6uZ9ArVSm"
            "ses+8EMGjpFBUXrBBL076XMxXIz4c2YY59mP3kZPhR/BRTvnkXww/mgu380nzmT0mhapGAbVBwYvLYL1"
            "LIzV07rvFepTCJhDWKPyj0yJnWcUVix8uTR2UGGpHUTurEf3HRRj+S6GXdmt0Hz2vewltdbAqSKOqosI"
            "uVqjC7qQ0yFRrHnVeD59w05IdivD1VIfXeeBVY7ikFrf99Mo0HXcR+2zMoeQnZFH6sUpHVzfQj9SVRuL"
            "VA6asrIf16mSzTG24iwDsyjZ5E0vw3Dd268mo6hYH5805aemzLzMxb0Ewj5mzC6+318CREY0DTqGmrCC"
            "I6ehkalgxFJsOSqbaO3xX7QVYxak+/R6z3a1hkYfp62FtCh6+ekNg9EfPJn7kV0+c9HWptvWpAazeup9"
            "n8N+lLY2Sv44B98DTQ+oHpnaPVfpJzpF8cC9jWeUfr3h/yjp930QO+clBlN9dzFIWfkSjYLsowXxWAg5"
            "+WwOWiA7H67Gdl2Nq7lzjjfqKM/BdatAtU+XC4EhRNsgFmcguBYgJqdcCpUVHmgHzYerluf2fiYa+839"
            "HPvbQPtRnUbmay3BiRqBwMXJqXODUnMFrJqsadXGeMFVPntJxRPSnSrqZFSh3hoQiTNkNBY86wo+uJxc"
            "klvrj8zeXSPdrVlzJQvd6KAGI3KgtQxFZCx4jaZhqja3IxN310h3mtsvtrUSowFqSUGNcirlu6DIKskl"
            "JlvV7rTSjzaK3vQFlUYPWi2rWbztc+amGnDP4JFYA4p0+MYcfWQVyJwrqY1aMA2hl0N0vWU27IFDUnLf"
            "R7GzkrqEYpohwIIaXNUFYm8dNNZHrHJtnbv07swxI8c80GquhdNunVk+PeGtqupYWSSsgaRSn0BrPFRU"
            "DNTHqvf+jMSXXtvzwoSnnkLUbH1FD9yKhlL6eHpjHBRdYo0ejY6HrvqcE9395OUJ+W55DME3hyIUIgrk"
            "iDlDDBF7QTu6iNomOvQGX7EEnnoOmqIVl6yCIRcBu4Frcy2QW2iJtXXVX37ZlPYLzYPpM39Gwj3feDbC"
            "U+8hZg46OwRkK8YXY4TmRD6Y2IuptPVIh85anyHhsTsTe8nNkvA6vndIbmU/wlseRA4xtyISOPWjK2+X"
            "vRmsQgxNzpLU5a8EFsLaDLaPBlgS3n90xn5Up55DFVvBd1HgrJZDEvW6axEPKhhsQVUdzJG22XzkrjGD"
            "6YnjpdzdXOW1v1+2H93tYdbMyEb1oEIMUGJQQIkq1FxUrGhjdIcEGOd5aV9UtW3Vq+p+S5tygFgZlLwJ"
            "cBwzNDEZqjXFVvwBEONCqYFX1pkgPqfk3R5+bRxl2ww4hSTyt499qWIG2xwjklzjEJ4sqfxuvd897HnV"
            "/C2vNdGgHq71xv6LteQ4pORvJ/1HSv62Ks90yPKPd70LkZajNUoWjYhZK12MZ7UR5vmhav66FLeDXc2D"
            "dmOB03RgwR453J2fwEYp2rTgtZaichX/moy41lbJdyyv1FvDVaKSglXrruW9S/5mhtguFA3UR+aMqQ33"
            "zeiN+xLXE0KemiIhR1Ieg5jQJI6fz3Lycb9OYBtKKr5u9HYcAnlGgHFQfbLWEvDpL+5WMXxlbUq2CnJO"
            "KG6JLmATKzCVa+yzO6pbhy0Orj6YdU0ckbiD41S6+67xzUYE4gNmaaCzxmoduvh5oP/I9uM5V28Z0zPC"
            "3q/qjEwvYzm8M+77JHaFJ1lTs2QIIjsv1lnIYG1o4DD27kNy5C6zeMuEPkfH9x64DtX27M9xK7L24jr1"
            "LpIRo6S6DH14BtQ+Q6cbveK8iXYy0YtRcEg30Vy4cl//IG4wrrh6ua1n5ToNRlaVbXDo+vg9gtBjZrrm"
            "AM5rzr6kJJrqosdpaNM9tk5iyXfDKT65MJj6a83J1RzLQlRzkLr9Gou1kDWX7JmCMkf6a3OB2+/vgPeZ"
            "YLfhDh80G24vulMjKlqsjXPf5JZcz7AxOJd650RrLSqVXThiK+F8yD6hz09Idqu9myhkXcUgTRgjlJIj"
            "cIwBmJqp3ji2Ty8knLcLwNTFAvbmtq7OfB/atj19bw9zdSfh79fKxqI5pGag9cZYNiWCr6mB3Gdmudeu"
            "bIzcOpG1uhF+mG8xg9ELMkOgZRDTm977dkRhbXBihNleYf7woYxPzeXUMTAvGAfbZ5WPN7Q39BwRavwe"
            "iF1VDCpyKlqLKVBaBuNKBg6OwcfKpFNUFQ/pi59fjOUF2W6NdaDqmYnAspxU+RZATNgMWpinyFW+HjIf"
            "/arZbnXO5OQrix2gVY1yIvJQojNgjdIYOBjVDsnwXDXbrQ3bFEPsc6GiUyxfohPHVond5REdklYmHzrX"
            "f15VIeQWxg26+wfLqsZjq0L2obvlKTTh23u+XC9trCpasKH0qgXli62xGXvZG7bZLZAH61cVC0Htb8Pu"
            "w3NqXgXSImZV7CvKrZyziG9r+0TTwFUlkpt80OS9eWYKOHRTie6lQbD7e1z7MJ1WJRhvWx4nw7kSwPWj"
            "oFFV7NZaVNTK1oufqmUWSi3QDuMwrZHpiZ/7qQ+LwSGiiNHqmgOvXQMXkoYs7pclsbaUu/RKD2GKCyO+"
            "Vh8+vmS6OXbl+Uy3N5T7lp1NDnLUGpzzPUOODJiUT1zEGCiHjnWa5UUVqPdFjAL1tA//Vl5Qt6xjd1it"
            "CWK8tjauR3Hgakara6rx6crmuSaszLjkIAyhbzPqyn+MCzyEA9wB4YAdbHfNWTlkadLF5QNHvEYPpldb"
            "jHg3MjLnwPus9V6XiDd005X7oufRdO2c7/GS898MoDkF36nfxS15H5HAMWow1bC4BUlBaJxzK9lbvvE9"
            "gO/W5ILAqaBJUK3cWosmQzXd/PLsi6ux1o0e09MGC+fSRB7GlO5qYK+n7kcc025P3hntXdduDx/AKH4u"
            "YeKLKP0xOOru+0LCclTAfjMzvvvWd0WrWvROF2bxnqxYpd6LUWVMBXYtKCf2QKFL3AwxkjR2ILXqGg/f"
            "lAaeguRURlalQ05i6SeMWYz8mMBZFiNKMNbgxNKnQ/oSZkVS91na99b9Ny2cpyC5FYkKti/oS0Dkelet"
            "b1D7nOfC4vM302wN6VJJMg+qp6KWJPdf1rkfyWnUCTXm7Pv8UCs2vS6WoSTPkGtugYhqaYd0es1HYavQ"
            "61l1L50c7U11wMymvUhO403GeE2RAnCHmKK48NSLz2yi3Azqyu6QeNOMSPo+gZLtvWOk9u/S2I/kNMrk"
            "5M4Zp3tWxFtoBT1k68T0IRWzuJ2o+Mk7OV9LUnCiH3j1iAvOdXZzcyjzPjOadtFd25LTgJO2zepeJeF6"
            "qWQiW0GZ4sBHn6Mc1RblLtlUf3HAW9GnZktq2VW5wBV7D22FquUWN+NzNabJ2S7aGeq7M/Tg+qTkTlht"
            "VKacifBWiXoTUYDiblrlxYlvnMC72MQytRiKKy7Xp+aMzZuw6ZN3xpKfkXAPFx1Vnr4v4O0pBlo7ua5Q"
            "rKY+cNmAUVRAPPkoHmywaWN0+wn8zW8L2+bgb/ah42bQfTZt/wh03/Z9hL9piA0F5j5I8uEDGH9+l785"
            "v4kvYk9xGFQv7xulKu7vd34fwS5vKTuTbPNe7CkTIPcGFOJK4GrUqmFDPnjH07yIohoDzitPXoju7zXt"
            "R3Srp1gceK3EzDdyMsDiCapDA1gVydkDI11y/B4X2g7j0OMR56ZSOixCt5Pu9+2qgoTUWoTiTBSz1TXQ"
            "zYr+N0ZltozxwkOgOOZFe2PgEvA3WwEOiODvyXe7JNUHj9b05XldHdVYQIWWQGWtnHJkIq1rqfcCfMVw"
            "pwq/Nc2+Ng8JWZwuK14BGWyAmk0JfW7rRlHaXnBnFFUJCx0G7Bp8KWmPnO2/L9tpCDWbUmNNBTwygram"
            "AGHIIhl6ja9yxmx0WFzexX1ZsbuVehJDgEO2IgyMki+kq4BFObpjY3Pl4suBGyrmdXPtYFe1KOLLrume"
            "5eZOA62tmcxZYNog9zWTVhDkKoNnpSkROdVOWqM+PzegF6P7Ae+jX+bItBOSGF8oftxmS+v4AV6WG0B+"
            "of1gV42V/tuZ68ci2OUGBBVLw9r3CiYDyjoNlZiAPFUbSwytHDqKbGZEQ1+kq1fLUPy3vTmnILoVXtXI"
            "Ojglhwyi5LXNEIpGYcvITiSFfL1sor73qdpu4iyJ7u+q7kd0S8EHTOKRugIuEEPV47ZGFyFmxd7VZgXw"
            "pV9STcO4JHeJ9ID+s/2QThNTfVgbKXGfxKvSELKPyw1U2XIwfV6DfrrYfJ46ve/AoMGF+70BG23UB+n0"
            "nWC/76ZGUrkEynKgcc27mEspirUffVMlibjVeLHmUkdrBqNXST+xP8+KdrvsVOeUUp/2bG2Aii1C09mD"
            "S8k5hSWUjbnPF8gWB36Qqm7TEDoD26mDGmry3N19tKlCcVFsACwRdI9m2Zi83shUXyBb+7D4y/dtXWdl"
            "OzWySsq51wEAKTm4eAkerCYDBXu3bzPGbwjbCwxdfddqPSHTaT2ANjVg7veVq7j6KThoXgxXo4J1KjtT"
            "9UWnWbuV5Qd37/j3AT0PQ9rsfSj7hHynZqxSVRmtImAhBbnv38hW/AS5rsQp9wzhBQ9Q2uEXnJDpVmmA"
            "jxRFbcnTby2kkhSgjgEi+tqKD8mWdV3VZTJ93Iw9HdOtoVS+VOWNqCyMmCC4ymASefG5qpxcBW7uGZnU"
            "uYZN+upxO7C6J92v2jFhE7bWymH7LIAnwiYzDY+a3vSI96USuLHS87C2h++T2Gnts0mGciGIMWrAWhuo"
            "1Bc/NZuVNdTb9y429DwWSejViDlhe2zHzp5st8z9KAfURmj6cQSrUxZs7oufUFVXo+kL7y4YLqoB702m"
            "cbnWURUo+8KdlvgY1X9Q7qtpfal9CQnMWHwevKaCqJ1ezxi/PLgiHc2qHVLgbnSTnOXmTp0pzoFD7+J3"
            "NVSo0SE00e4Qs6McjBfr9KlU6u64/pz2NGKvBwx2lVqlbkkes1UZDXoed6A9kD5sqfKcUxxhcD3lvExx"
            "bAakDtDVnix7b7XaTWi2T6VWC3YDm3uLu2fdVhyYvxk/eiyI3R6jKqmvwypifVcDKYm5mIKt0FpQhcSf"
            "KeWpHq/5ssU+b9fed9dSWA/bOgvbLW+xVjZBjXtW+pBDccRdcQToiimh+GbSxapqPdaOhPt7q7taOU5V"
            "7wl3amQ6r1TQpUAPcMqdbaJXWpGTe4rokFHrA13x64W75T8WjRZbVKARI0Qr+tqMokH8RuI0Lq24yjFx"
            "PC5i4PsAFG98MgeNibPeyr+0odfDGB68lDlxpg84VTyEVf+d1/2KHTMT6nskdg2KiyHJs55EacXgwcsh"
            "QZu+9ttWjZGtXNNLHhT3snCneeXSuDfgBUgGCUoTFz6RYaCQkg4pZ/fEpssb3B3tuISlu0/gRUlB7Zus"
            "msoOTBCRmjyGHC9xbfPLQp0WjDmRBlqnCqEUsa5MtkA+JkAvksCLjVDrZXbmviTUrSZdJG9VtVCbNnJo"
            "I0ZWrBEyNyYizer/s3clO5LjRvSLKDAWbnUwQJEU4INP/gLDHo8PxhjwzMGf76CUi1KZnSVVLq0sZaPR"
            "U1PVXSW9ICNeBIMvRkf2bzdwHdyzSdgRAWsAS/Ks9Wp+Ua3XWXFCsGQEeLdQ32DD4E7TrtaVkmJVhLEp"
            "Cbj1jr6EMfEOErrkKbPXrzgLoU6X6LW3duMXvTkZILpAMG4OqNN0K+Xab9YalXxHSoIYqtIWVjqSbx3J"
            "o5slPnZ9OpxkPgAbv+/XZ5ivzDEL0PMLuy0itCZJ2urErUoMUyn5rAxl58Am7Rc52HXKxVF/FGXo4Afm"
            "t0XOQ3VawfemCJeSXCAhZlUPluvUDqdM0C7FEDjAEva6WlTRH9uiiOa3785DdZpwtVk8aIm1zhIkMdDe"
            "qy6QV8FDke0fKcYv7P61zGXWfQPf/kjP4LHJbFRtuRe002iF1FLxEJVOEqjQoFbaZ6NsfWT2Pqbwac/p"
            "Dyf/rXDqn4QxVy85h90YNG+qGNJZGFsy9e8a8JeG/k2prtbJ6lyCMroVO3j5SFY3Km8YgMhgGLGxG4b+"
            "rYhJmA90x06J/oOvCSNegf7K5TQwyXljkyo+auFpHahQOCqOqQ1ZfLTho27NC/b81Ok/1FRx1B2ZOBnz"
            "saCAOxPfaZnBl9ZlCkYhEyhMVtZ213J9+i77ouuomuX4rut6ANl+kPNeOJ1HlwAX9a/OhHhadIjojdN1"
            "tFKVqIPWyRI2CVQXwYvfJu3xhhL58dh2LUe2puopy6K2uznP3toLo9jmlb8tgSPCGmgPkPfE8JPy91pm"
            "p80AYhZj+CEO1zKxkDrPPrWKLNRG1BxUqA1AASXtlVcJ3l6f7vV6y66/A7CLVf2E2skt9ZnNFOjqXWgz"
            "HpkU+qbhF1l2+kPbxu0beOxIHGbZsvsxENfWXdtF+dm5Pwkk5R2jJAJRqFLySWuW51506LKq7fzZArsj"
            "rGd1gJAleXJVV7IOjpCnlbS1y51iZ1sWOpQiX79semU/r6QhStf5MX3tqnJvAdnqGm6/omwkzJCAXV37"
            "O4zJXpgTuyVMoO+EgHE/HtkLU8c2tU5YOxd6CbMDJBcEl7cEiWS0vYjakeeSvaAJtiVIEIykoRr5SMBo"
            "mGazYUzIIlEgfxTvI7us5/T7YeIRdUCqfVJ7TNyFfp1NYYJaew8QRpHYXUjitoSJs8FoDrXcdYBk4+SE"
            "iD2B1nWyzQGTjZMToOpOjNfjdbJxdoKAaI0z9XzkgMkyejIUL1Z0zeGOuyho57EumxE6GycqDi04w6Pi"
            "GV2atbwtZxvAMdSqwxGTjTsW8oSGGHm0dS5pIW8qFZRtA7rvHzhAsnFvwhoYXcAwKqJ86ard94w/Em3E"
            "sZhx2e3ShYVXu4h4P4QMkmwoGldXLh1pbYrmsqRDyL7OrztgcuG8ZVPRSAsmrO24Vhv6TsO3m6mH7AFZ"
            "EsY6WvqAzrIE+vxu2Wn73rcKWQwWQ50ydURrWRj/MVrrvYV3x+WmJeA7dHZU6gwbL3UyMGvS4I50WajR"
            "xp12nZJo/bjUyXrjgQy8qfdU6kiWIybLAtm3w0TIDluEUUVCXPTGl4klD05gcCNMllVprkb0bxuaxAuj"
            "GbT3D7gtq+SscqbyHQvGTuhP0LX37gDQV+rF3zYjJeGGRFWf7IDPreTw2y6meuGyHnqPPDcsK29c17So"
            "nmoLjJq894ExjJqpa+/7bUBO7mBtAkg0ThhWgCrYfgByGeV8A9mvSCYIYH294XIAchlPfQO501cmDjw6"
            "RpFU8Y3j8p1tg3bBhnrd8ADkrYx4i0BykGgjMI5xXMaQ3zgOrbKWnDzrONIsI9JvHHedTgFImOR4Qd7K"
            "uDfpIcGisQbrMPUDkMvKsm8ghxQZBTjqZZT3QOKtac25IsJ3IjlgPNHoAg/jnbOX71ZjcKFOxuIxYvdM"
            "U7awTT15J/94HDdwVobyl7/9/scv/72CZEXw91/+aH/tP/vvfwzo9V/c/88qLnnSB+pBkGCQ6nXj0Q3T"
            "u8X9S/8Q091XB2RHqqfAjj1ZPgnIk2+1wtuJ9KF5mNw2qGVYOxovhmaGOMksQK6JberdL3Xhj/2vF54z"
            "iFVIgP1BbdfyaFzGUUjgnghP1TDujPB61u6cXX1nbKfyOvLExmkmxcmxYl2vLvtoFIfoAOSlLCzBdi3a"
            "kNgra2Ej/2BYtZLCPXjVng8evMOyvSAxsh6vYKCxVVpl8ApLRw/OBfjKHB3UmkJnq0SsVgxdqzxQVjkR"
            "6eA1R5gxR+fv//tt5c7XNlivwgwwH8WgjjDLK/z19pW8+y4XJWQ7qDqRSMr6FJSx3ipuA6vsAhdr65eO"
            "g1/Gj1PBHhA+A3tFHEJ+64b8zlfY2tyzVzhAdxCEvz/MZ9MmlriMKzD/vqevB7B7mrsKMgsfhuV349wO"
            "bj3QtPFj3wby6Xf6wd91JyWTkx978rk3pD+CFENAZrR2NHxv+EchjAv2b2xvX66sT7pb3pDeBdJbPcCk"
            "lrCSWoFEszrIzA/iehLNjuJ6O8rwWKABratSGqOTpz3iJ00j05+/wjRiDph3SCO+hOIGM2b8wPq7gbDL"
            "OqjOrloi8f0cc5xrIWbdsaQnFq0k2RhQefZaxahDLEZH65eILq/HHFWqVjfODjPSralCMeszx1TP1qdO"
            "d8ZZBWSdkvxF3jQ6Vtp1ABkixNFMtyuKYp+Fh4sF5/UECTKN3onliV/D5wYJMR0DV73zM/uNT5SuxIg1"
            "KdqKV6IPcMc8nU7qQ88G71q5OlB01JVY9fNAdTaxMpqzokidYW91R+4FC35zVvQKrPHgQL2e8mDdD8Kb"
            "qizwsB/Gg+DWYo1pnDacW61bq0ILnfJo61CULkjYztAih865JXtjRVVE90HUuKpZNlQRx9Koj4zTS6xx"
            "Ljt/4954+TA9ECzfcJ16OxCsJ4Tm80T5pCH4lUKyDk3YjbcVFzRDqvbuoF0LxTalNmtvlY0mKVtnMElw"
            "7pQWtykEtfOuLJ3Fsq7U4Hzl/kT0p6E3MqS21LG4iYsKOmZVOMmbMSbg1lDuruuEr9jZo23oeGQ0ll9/"
            "pLOfY4W7O/kVEyDxPtrsUmPiR3RNfNUK03MnZDKS+lqVUuoUd3Ub+AQqcmwTOpsCmJe0QvVCrjFVXXHw"
            "Qs+ioXOscNaL4W1bSAcV5DUVt9Ypr428iiRlUEi4qTuO7P7GdQnfuNpNMGRxZ+fdzyY/PO7XfCXyg9TA"
            "vi2OzIyhkncH7Rr5cbZDLm0WtuNJaSNvVhuOlIuEBtEbaxY6/tWgP2cV/0RLbKkGgdhgVUsaQvClCUg/"
            "ywrT2kPAxJ1ukzh/U+er+ag86KSSdZ2Rl7TcLRl5uyo6yrW5ZheCuTKin0RHz63wYDr6tsIsK0yJUKd1"
            "67Q8ezKQ61tmFZONygYXtfBRND5sgAgxNbw/WrPuLH94OhF61SoQ6caHXemTzHj1PQu0a0QoRTaxS1bl"
            "ko2KPgaVyBQVIrbFo+0sLBnHvaoDmU9W8E+0woNJ0OpqceCavgW+z4KfnQycoT8lPzb5En195Jycsp0u"
            "imIOKrZZo0vgUjh2cr8aBQ1NqPx/8D7Pah+aY4Wz4eBdpFxCS0qSL3lNz0n5UJtVUvDovUklXI+7Ky4D"
            "gW0s74JAfx9zPWaYRgSsTfZFHt5hQsVOYkPbQlDZeflPTCHG12wXqiTUNlgFj3sSao999k8noedmmIYE"
            "663HJMlY1FB3QwhK+CcrW0yMqQMX0pKh7m8zzDPDdFYzcA5tzkkRYFKcIcpuEB9FvbhJ25k7JQOHNGA9"
            "CYDRDew7KKw7u2D17ATgZN7Mq3TvfoLh01b6yW3Ox1DQ+Qv9tQQCHrXCSbMObFythUysdTLu4ytXA1be"
            "VILChg6HYkNK/OBLAeAJg3cjvc091ieDMl6prmBIXMv+frHpxd4e1e78GXrXCgyywjmR8apjU5QusahS"
            "gFRmTh36zsaX7PjseYwJjamqpAOPeVbH5yJzTPmM9dnorhMeCRaVTkYSXkmvlPWekjfORhNfNdc12ODx"
            "yPERN+5vNse09IBoyWGUl0i6VRCpqGxZ0i6bNbYcg1nE8leV86Jp3P5Ss3la6WGROaYHMCE5KCFn4fiW"
            "JfNCrVoXg7JtyyAeizm87AFMvfIIu/Z0rkM1nnNVZok5po1BxKkezctLeI0KoIDSMUoosa6ANeShvGxH"
            "9AuYY3owxuhTsSWqejCgyIROyaecCp31mrCYFN+743HmmAqQRCYbI0aFbQcqtQlUNrkoH01J9bZA0vw2"
            "x8PMcaa4Y0zxufVJmViSamPsVIK2FW8FRX4sGDHKKx/iXMjaVmGH6TlC7tgncOKmIHfyBwvZRUCJ5EKy"
            "JGhkcWL3qGRsXdJAMgjDejR0cmcjOJk09BaLWI4sE5A493p2PYX2rcNxo2sxJD7A6NG0sAO2t1bdNi/J"
            "o4MVVk5n5Wc40el9Q/sVxYEQNAO4c5eA4zkPb2y/4hKwziromxKn2N7qbldeiCdo3L4p0ww15AdjbTUF"
            "sWot002xnnm+t6I8xddLVq5WGoc85VhyfHCe8hmKVwvygXLOpVUhtqRCwKQIU1YlUaY6aL24o5jiSxXk"
            "fdVcdbQ7buWqDPeckuMSc2ynIM8foJtQVYWHgvyzGjAXmWNLBXnx9rqONh+8/dPOR5aYYzsF+ZeIHdsp"
            "yL+EObZTkH8Jc2ynIP8S5nh0Qf5tj4X2mBbmbSHwJmrVdQUVJWdUJztDkVhCh7YTarVE82RlHVlCdyWb"
            "rmdVA919pN7PMjucZYBIrS1BgI/BKhMAlDfitlyXidi5wHrJna/VHVRdqGqswg7TllsTW3BErWR9nFTq"
            "qvQbpKKwQJc7ox1/0hp3XoQ6mZ10+Iy87uTjo2z/v37982///M+f/g8AAP//AwBQSwMEFAAGAAgAAAAh"
            "AF8q2dwXAQAArAEAABQAAABwcHQvcmV2aXNpb25JbmZvLnhtbIyQzWrDMBCE74W+g9Bd1kp2HMdEDklK"
            "oNBj+gBClmOBJRlJTVpK371ufgillx6HZb6Z2eXq3Q7oqEM03gnMMsBIO+Vb4w4Cv+53pMIoJulaOXin"
            "Bf7QEa+ax4flyGYM6qCPz67zaKK4WEuB+5TGmtKoem1lzPyo3XTrfLAyTTIcaBvkaaLbgXKAklppHL76"
            "w3/8vuuM0k9evVnt0gUS9CDTtCD2Zow32rnhH6I1Kvjou5Qpb68wOvqTDqM3Zx6bUQaXXs195ktMN6UG"
            "MyUj0wr8OYctK9l6Q3g1B1Lwak3WUJSkArbLd1W5zTflF0ZHgRcLjNokMAdeEsgJz/ec1znUxTwrADBt"
            "lvR32l3+/Lj5BgAA//8DAFBLAwQUAAYACAAAACEAEjWJOb4BAAAyAwAAEQAAAGRvY1Byb3BzL2NvcmUu"
            "eG1sfJKxbtswEIb3An0HgbtMkYrdmJAVoC0yNYCBumiQjSUvNhuJFEgmitYI6Cv0Ebp069Alb6MXKSVb"
            "StwGBTTo5/3/h7sjs7P7sojuwDpl9AqRWYIi0MJIpbcr9GlzHp+iyHmuJS+MhhVqwKGz/PWrTFRMGAtr"
            "ayqwXoGLAkk7JqoV2nlfMYyd2EHJ3Sw4dCheG1tyH6Td4oqLG74FTJNkgUvwXHLPcQ+Mq4mIDkgpJmR1"
            "a4sBIAWGAkrQ3mEyI/jJ68GW7sXAUHnmLJVvKnjROhYn971Tk7Gu61mdDtbQP8GXFx8+DqPGSve7EoDy"
            "TArmlS8gX5sa7Noo7aOu/d61P7uHx6791bXfuvaxe/jdtT+CzPAU6KPu9stXED4fjicR/oUF7o3Nd5Ef"
            "aqPu7+MGmtpY6ULqSAVRcOcvwq1eK5Bvm3363+PeaeFO9Y8hnw+OSY6UtQ2TgMxpQmicLGNCN2TB6JyR"
            "xdXEHE3Z4Tr2bYKMwhrZfulj5XP67v3mHAVe8iYOH1luyJKlhNGTq37Co/wTsDx0/V8iXcRJGtN0QylL"
            "5+wkeUYcAcNgIsC3xjb7zf2ljl55/gcAAP//AwBQSwMEFAAGAAgAAAAhAE0/m8skAwAAiwcAABAAAABk"
            "b2NQcm9wcy9hcHAueG1srFVdT9RAFH038T80fVbaNQpCZkv4CGAisgkLJL6N7ezuxG6n6Ywr+iQdQxb1"
            "ASNqQhCQKH5g9MEYNS7hvzCwyBN/wdsWyq42JKhPPfeee8/cztw7g3qnq65WIwGnzMvruQ5T14hnM4d6"
            "5bw+URw6f1nXuMCeg13mkbx+h3C91zp7BhUC5pNAUMI1kPB4Xq8I4fcYBrcrpIp5B9AeMCUWVLEAMygb"
            "rFSiNhlk9q0q8YRxwTQ7DTItiOcQ57yfCuqJYk9N/K2ow+yoPj5ZvOODnoWKpOq7WBALGcewyAR2i7RK"
            "rG6z08wBlTrQFAscbuVM00RGglGf77vUxgJ2yhqldsA4KwltLF5eK7DbJCgw6glktAbCPhEOdcXWUFy2"
            "peQnFb5Scm5v4cf+izVkZMSgAg5wOcB+Baro6oKYYxuNu9QhUXXIOIToGhPwAUcC0Ah1HOIdsuBus9Ho"
            "6IBL/Zg4gmjcxi4ZgF2zStjlBKRTBxohOOqIAqYBRNZET43YggUap3ehJzp17QbmJNrrvF7DAcWe0JOw"
            "xIix63MRWDubW3sLb9XMUxU+UjPraua+Ch8qCeY7JT8rWUdGGhvDVolWTC9al+IAACcGJlpKzirZUHL5"
            "FPq50+iH35V8lxyrpsKtGNWV3DjNemb2grERbzjg9qMoUuESPlaC7hAZJ5NrO5q4iuRgkoL6r1zXJgYL"
            "w0xUqN1aaIr6oAQ3kxkmMHwUZ3LRCPFMZgpKj8rPZoex697yM6nj+dKUfK7kBxU24oaJjzX8quQbMDNT"
            "d+uzzeX5vS/zzeUlFX6BHlDyPWSdEKxmPu5svdxfbTTvvckMay5u7H5c1K4WJ7Xtewvaz/XPzW8fmksr"
            "+8+e7NZf/+8/+IdUuOq8nqjO7Xpb6/8/7cMsuaLCNSVXDxr1g63X3ecum+ZBY+7EvMfxCptKPoA5SfJy"
            "F86ZfyS29f9vHT+KPVwmEZGiq9S7ySf8IhuMbvnDy6zdicYrOCAOPBrpZZc60AgMS+BCfD9MTjRx7XZq"
            "8oEK9srEOZL4k4iejMnkbbVynR1m/Jq0+KKb/+jRs34BAAD//wMAUEsDBBQABgAIAAAAIQBgLQHVDQEA"
            "AJIBAAATAAAAZG9jUHJvcHMvY3VzdG9tLnhtbJzQzW6DMAwA4PukvUOUexqTDQoIqAYBabcdut0RhBaJ"
            "JChJWdG0d1/QfnrfLZbtz3ayw1VOaBHGjlrlONgBRkJ1uh/VKcevx4bEGFnXqr6dtBI5XoXFh+L+Lnsx"
            "ehbGjcIiTyib47Nzc0qp7c5Ctnbn08pnBm1k63xoTlQPw9gJrruLFMpRBhDR7mKdlmT+4/C3ly7uv2Sv"
            "u207+3ZcZ+8V2Q++okG6sc/xBw8rzkMICauTigQQlCR5SPYEYgBWsqpJnupPjOatmGGkWulPr7RyfsaG"
            "PvdeXVw6ze/WmQKu4A2Apt7XAUQxhIwl/DEsWdnwsm4iD4Yx32f01pPR36388/aZxRcAAAD//wMAUEsD"
            "BBQABgAIAAAAIQD+RdGHAAEAAOQCAAALAAAAX3JlbHMvLnJlbHOskk1LAzEQhu+C/yHMvTvbKiLS3V5E"
            "6E1k/QFDMvtBNx8kU2n/vbEgulDXHnrM5J0nzwxZbw52VB8c0+BdBcuiBMVOezO4roL35mXxCCoJOUOj"
            "d1zBkRNs6tub9RuPJLkp9UNIKlNcqqAXCU+ISfdsKRU+sMs3rY+WJB9jh4H0jjrGVVk+YPzNgHrCVFtT"
            "QdyaO1DNMfAlbN+2g+Znr/eWnZx5Avkg7AybRYi5P8qQp1ENxY6lAuP1ay4npBCKjAY8b7S63OjvadGy"
            "kCEh1D7yvM9XYk5oec0VTRM/NiEIhsgpF0/pOaH7awrpfRJv/9nQKfOthJO/WX8CAAD//wMAUEsDBBQA"
            "BgAIAAAAIQDZkRv83QEAAAcNAAAfAAAAcHB0L19yZWxzL3ByZXNlbnRhdGlvbi54bWwucmVsc7yXXUvD"
            "MBSG7wX/Q8m9TTPdnGInggi7EMQP8Da2p22wTUqSTffvzaZ0adkOXoRe5m1y8vCec5L05va7qaM1aCOU"
            "TAmLExKBzFQuZJmSt9eHszmJjOUy57WSkJINGHK7OD25eYaaW7fIVKI1kYsiTUoqa9trSk1WQcNNrFqQ"
            "7kuhdMOtG+qStjz75CXQSZLMqPZjkEUvZrTMU6KXudv/ddPCf2KrohAZ3Kts1YC0B7agphY5uIBcl2BT"
            "shv+qRexi0boYQh2PhLFFUoR1ItWg3nSyhnfkXQSRhHUimxlrGre3W4dRBzvVSosNOcYzYSFxLH8o4YX"
            "u6nBc8UTMZLLkUoEtYNNRqKYoxRBvchUs/10t7KV0l5e+jpaJGOX7AR1ZxYSp3Jns1rZR24s6D1STx7M"
            "YqhXSdCGcmu9CtoNf0UUImhT/ydhKE3QfCE9hVdNUE8QikuMYjoSBF4dQSmksmCG7eOJvRl46xy7GhuR"
            "aWVUYWN3Zv0hORQ2pSwZ0GhYi+1bbCkLtcfxVdSZoM2L5GeGUlyFpFgL+Bq8UzoJo7gIbsWwSDyxNwMt"
            "kqDeIBmaohkKb86RPk7Qhjl2MR9rmBllbHi6u6utBNPvF0/szegyQ3u/L4sfAAAA//8DAFBLAwQUAAYA"
            "CAAAACEAdD85esIAAAAoAQAAHgAAAGN1c3RvbVhtbC9fcmVscy9pdGVtMS54bWwucmVsc4zPsYrDMAwG"
            "4P3g3sFob5zcUMoRp0spdDtKDroaR0lMY8tYamnfvuamK3ToKIn/+1G7vYVFXTGzp2igqWpQGB0NPk4G"
            "fvv9agOKxcbBLhTRwB0Ztt3nR3vExUoJ8ewTq6JENjCLpG+t2c0YLFeUMJbLSDlYKWOedLLubCfUX3W9"
            "1vm/Ad2TqQ6DgXwYGlD9PeE7No2jd7gjdwkY5UWFdhcWCqew/GQqjaq3eUIx4AXD36qpigm6a/XTf90D"
            "AAD//wMAUEsDBBQABgAIAAAAIQBcliciwgAAACgBAAAeAAAAY3VzdG9tWG1sL19yZWxzL2l0ZW0yLnht"
            "bC5yZWxzjM/BisIwEAbg+4LvEOZuUz2ILE29LII3kS54Dem0DdtkQmYUfXuDpxU8eJwZ/u9nmt0tzOqK"
            "mT1FA6uqBoXRUe/jaOC32y+3oFhs7O1MEQ3ckWHXLr6aE85WSognn1gVJbKBSSR9a81uwmC5ooSxXAbK"
            "wUoZ86iTdX92RL2u643O/w1oX0x16A3kQ78C1d0TfmLTMHiHP+QuAaO8qdDuwkLhHOZjptKoOptHFANe"
            "MDxX66qYoNtGv/zXPgAAAP//AwBQSwMEFAAGAAgAAAAhAHvzAqPDAAAAKAEAAB4AAABjdXN0b21YbWwv"
            "X3JlbHMvaXRlbTMueG1sLnJlbHOMz8GKwjAQBuD7gu8Q5m5TFRZZmnpZBG8iXfAa0mkbtsmEzCj69oY9"
            "reDB48zwfz/T7G5hVlfM7CkaWFU1KIyOeh9HAz/dfrkFxWJjb2eKaOCODLt28dGccLZSQjz5xKookQ1M"
            "IulLa3YTBssVJYzlMlAOVsqYR52s+7Uj6nVdf+r834D2yVSH3kA+9CtQ3T3hOzYNg3f4Te4SMMqLCu0u"
            "LBTOYT5mKo2qs3lEMeAFw99qUxUTdNvop//aBwAAAP//AwBQSwMEFAAGAAgAAAAhAMgHXCtkAQAATxAA"
            "ACwAAABwcHQvc2xpZGVNYXN0ZXJzL19yZWxzL3NsaWRlTWFzdGVyMS54bWwucmVsc8SY22rDMAxA3wf7"
            "h+D3xZF7H3X6MgaFPY3uA0yiXFhih9gdy9/PbDAaKGKDgl4CuVg+HMlCZH/47LvkA0ffOqsFpJlI0Bau"
            "bG2txdvp+WErEh+MLU3nLGoxoReH/P5u/4qdCXGRb9rBJzGK9Vo0IQyPUvqiwd741A1o45vKjb0J8Xas"
            "5WCKd1OjVFm2luNlDJHPYibHUovxWMb9T9OAf4ntqqot8MkV5x5tuLKF9F1b4ouZ3DnEsGasMWiRppfP"
            "Zx9t07iFkNfJYMGJBguSjVUbkN7U+pZsIa7FGdX3k58rUBys6SOzp4ATTZHWNpxoG7LmFWvNK5KNVRuQ"
            "3tSKtdpWJBsrGpnRm3axf2d0TVrLWLVlpDdWbRQZa0LJfAKvNNKaWrLW2pJiY21sZF8DVjSgey7rYKTo"
            "uZa1swHd2XasbDuKjfWQkmeUVRrpDFilAWlN8Q5Gv5ORnP0GyL8AAAD//wMAUEsDBBQABgAIAAAAIQCQ"
            "Auz1wQAAADgBAAAgAAAAcHB0L3NsaWRlcy9fcmVscy9zbGlkZTEueG1sLnJlbHOMz71qwzAQB/C9kHcQ"
            "t0eyDSmhWM5SCoFOwX2AQzrborYkdEqp3z4abciQ8b5+f669/C+z+KPELngNtaxAkDfBOj9q+Om/jmcQ"
            "nNFbnIMnDSsxXLrDW3ujGXM54slFFkXxrGHKOX4oxWaiBVmGSL5MhpAWzKVMo4pofnEk1VTVu0pbA7qd"
            "Ka5WQ7raGkS/RnrFDsPgDH0Gc1/I5ycRimdn6RvXcM+FxTRS1iDltr9bak6yZIDqWrX7t3sAAAD//wMA"
            "UEsDBBQABgAIAAAAIQBL9T3svQAAADcBAAAgAAAAcHB0L3NsaWRlcy9fcmVscy9zbGlkZTIueG1sLnJl"
            "bHOMz70KwjAQB/Bd8B3C7SbVQUSauoggOIk+wJFc22CbhFwU+/ZmtODgeF+/P1cf3uMgXpTYBa9hLSsQ"
            "5E2wznca7rfTageCM3qLQ/CkYSKGQ7Nc1FcaMJcj7l1kURTPGvqc414pNj2NyDJE8mXShjRiLmXqVETz"
            "wI7Upqq2Kn0b0MxMcbYa0tmuQdymSP/YoW2doWMwz5F8/hGheHCWLjiFZy4spo6yBim/+7OljSwRoJpa"
            "zd5tPgAAAP//AwBQSwMEFAAGAAgAAAAhAC6iG5reAAAARQIAACAAAABwcHQvc2xpZGVzL19yZWxzL3Ns"
            "aWRlMy54bWwucmVsc7yRz0oDMRCH74LvEOZusm1VpDTbiwgFT1IfYEhms8HNJCSpuG9vxEsXingQj/Pv"
            "+30wu/1HmMQ75eIja1jJDgSxidaz0/B6fLp5AFEqssUpMmmYqcC+v77avdCEtR2V0aciGoWLhrHWtFWq"
            "mJECFhkTcZsMMQesrcxOJTRv6Eitu+5e5XMG9AumOFgN+WA3II5zot+w4zB4Q4/RnAJxvRChfGjZDYjZ"
            "UdUgpQpkPX73b2ViB+qyxvrfNDY/aaz+UqNM3tIzzvFUFzJn/cXSnWwRX2Zq8fz+EwAA//8DAFBLAwQU"
            "AAYACAAAACEAS/U97L0AAAA3AQAAIAAAAHBwdC9zbGlkZXMvX3JlbHMvc2xpZGU0LnhtbC5yZWxzjM+9"
            "CsIwEAfwXfAdwu0m1UFEmrqIIDiJPsCRXNtgm4RcFPv2ZrTg4Hhfvz9XH97jIF6U2AWvYS0rEORNsM53"
            "Gu6302oHgjN6i0PwpGEihkOzXNRXGjCXI+5dZFEUzxr6nONeKTY9jcgyRPJl0oY0Yi5l6lRE88CO1Kaq"
            "tip9G9DMTHG2GtLZrkHcpkj/2KFtnaFjMM+RfP4RoXhwli44hWcuLKaOsgYpv/uzpY0sEaCaWs3ebT4A"
            "AAD//wMAUEsDBBQABgAIAAAAIQDuDIdh1wAAAL4BAAAgAAAAcHB0L3NsaWRlcy9fcmVscy9zbGlkZTUu"
            "eG1sLnJlbHOskLtqAzEQRftA/kFMb2m94BCCtW5CwOAqOB8wSLNakdUDjRy8fx+FNF5wkSLlvM49zP5w"
            "DbP4osI+RQ1b2YGgaJL10Wn4OL9tnkFwxWhxTpE0LMRwGB4f9u80Y21HPPnMolEia5hqzS9KsZkoIMuU"
            "KbbJmErA2sriVEbziY5U33VPqtwyYFgxxdFqKEfbgzgvmf7CTuPoDb0mcwkU650I5UPLbkAsjqoGKVUg"
            "6/G3v5M5OlD3Nbb/qcGzt3TCJV3qSuamv1rqZYv4MVOrrw/fAAAA//8DAFBLAwQUAAYACAAAACEAkALs"
            "9cEAAAA4AQAAIAAAAHBwdC9zbGlkZXMvX3JlbHMvc2xpZGU2LnhtbC5yZWxzjM+9asMwEAfwvZB3ELdH"
            "sg0poVjOUgqBTsF9gEM626K2JHRKqd8+Gm3IkPG+fn+uvfwvs/ijxC54DbWsQJA3wTo/avjpv45nEJzR"
            "W5yDJw0rMVy6w1t7oxlzOeLJRRZF8axhyjl+KMVmogVZhki+TIaQFsylTKOKaH5xJNVU1btKWwO6nSmu"
            "VkO62hpEv0Z6xQ7D4Ax9BnNfyOcnEYpnZ+kb13DPhcU0UtYg5ba/W2pOsmSA6lq1+7d7AAAA//8DAFBL"
            "AwQUAAYACAAAACEAPVVr9tYAAAC+AQAAIAAAAHBwdC9zbGlkZXMvX3JlbHMvc2xpZGU3LnhtbC5yZWxz"
            "rJC7agMxEEV7Q/5BTB/NegsTjLVuQsCQyjgfMEizWpHVA0kO2b+3QhovuEiRcl7nHuZw/Paz+OJcXAwK"
            "trIDwUFH44JV8HF5e34BUSoFQ3MMrGDhAsfhaXM480y1HZXJpSIaJRQFU61pj1j0xJ6KjIlDm4wxe6qt"
            "zBYT6U+yjH3X7TDfM2BYMcXJKMgn04O4LIn/wo7j6DS/Rn31HOqDCHS+ZTcgZctVgZTo2Tj67e9kChbw"
            "scb2PzXK7Ay/0xKvdSVz118t9bJF/Jjh6uvDDQAA//8DAFBLAwQUAAYACAAAACEAs58fMtcAAAC+AQAA"
            "IAAAAHBwdC9zbGlkZXMvX3JlbHMvc2xpZGU4LnhtbC5yZWxzrJC7agMxEEX7QP5BTG9pvYUTgrVuQsDg"
            "KjgfMEizWpHVA40cvH8fhTRecJEi5bzOPcz+cA2z+KLCPkUNW9mBoGiS9dFp+Di/bZ5BcMVocU6RNCzE"
            "cBgeH/bvNGNtRzz5zKJRImuYas0vSrGZKCDLlCm2yZhKwNrK4lRG84mOVN91O1VuGTCsmOJoNZSj7UGc"
            "l0x/Yadx9IZek7kEivVOhPKhZTcgFkdVg5QqkPX423+SOTpQ9zW2/6nBs7d0wiVd6krmpr9a6mWL+DFT"
            "q68P3wAAAP//AwBQSwMEFAAGAAgAAAAhAGq007neAAAARQIAACAAAABwcHQvc2xpZGVzL19yZWxzL3Ns"
            "aWRlOS54bWwucmVsc7yRTUsDMRCG74L/IczdZFtRamm2FxEKnqT+gCGZzQY3H2RScf+9ES9dKOJBPM7X"
            "8z4wu/1HmMQ7FfYpaljJDgRFk6yPTsPr8elmA4IrRotTiqRhJoZ9f321e6EJazvi0WcWjRJZw1hr3irF"
            "ZqSALFOm2CZDKgFrK4tTGc0bOlLrrrtX5ZwB/YIpDlZDOdhbEMc502/YaRi8ocdkToFivRChfGjZDYjF"
            "UdUgpQpkPX73H2SODtRljfW/aWx+0lj9pQZP3tIzzulUFzJn/cXSnWwRX2Zq8fz+EwAA//8DAFBLAwQU"
            "AAYACAAAACEAarTTud4AAABFAgAAIQAAAHBwdC9zbGlkZXMvX3JlbHMvc2xpZGUxMC54bWwucmVsc7yR"
            "TUsDMRCG74L/IczdZFtRamm2FxEKnqT+gCGZzQY3H2RScf+9ES9dKOJBPM7X8z4wu/1HmMQ7FfYpaljJ"
            "DgRFk6yPTsPr8elmA4IrRotTiqRhJoZ9f321e6EJazvi0WcWjRJZw1hr3irFZqSALFOm2CZDKgFrK4tT"
            "Gc0bOlLrrrtX5ZwB/YIpDlZDOdhbEMc502/YaRi8ocdkToFivRChfGjZDYjFUdUgpQpkPX73H2SODtRl"
            "jfW/aWx+0lj9pQZP3tIzzulUFzJn/cXSnWwRX2Zq8fz+EwAA//8DAFBLAwQUAAYACAAAACEAtM9YGbkA"
            "AAAkAQAALAAAAHBwdC9ub3Rlc01hc3RlcnMvX3JlbHMvbm90ZXNNYXN0ZXIxLnhtbC5yZWxzjM/BCsIw"
            "DAbgu+A7lNxttx1EZO0uIuwq8wFKl3XFrS1tFff2FnZx4MFLIAn/F1I373kiLwzROMuhpAUQtMr1xmoO"
            "9+56OAGJSdpeTs4ihwUjNGK/q284yZRDcTQ+kqzYyGFMyZ8Zi2rEWUbqPNq8GVyYZcpt0MxL9ZAaWVUU"
            "Rxa+DRAbk7Q9h9D2JZBu8fiP7YbBKLw49ZzRph8nWMpZzKAMGhMHStfJWiuaPWCiZpvfxAcAAP//AwBQ"
            "SwMEFAAGAAgAAAAhAJOqfZi5AAAAJAEAADAAAABwcHQvaGFuZG91dE1hc3RlcnMvX3JlbHMvaGFuZG91"
            "dE1hc3RlcjEueG1sLnJlbHOMz8EKwjAMBuC74DuU3G03BRFZt4sIu8p8gNJmXXFrS1vFvb2FXRx48BJI"
            "wv+FVM17GskLQzTOcihpAQStdMpYzeHeXXcnIDEJq8ToLHKYMUJTbzfVDUeRcigOxkeSFRs5DCn5M2NR"
            "DjiJSJ1Hmze9C5NIuQ2aeSEfQiPbF8WRhW8D6pVJWsUhtKoE0s0e/7Fd3xuJFyefE9r04wRLOYsZFEFj"
            "4kDpMlnqgWYPWF2x1W/1BwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAALAAAAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxLnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF"
            "9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFc"
            "aMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4viP9Y4eu"
            "c4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwMEFAAGAAgAAAAhANXR"
            "kvG8AAAANwEAACwAAABwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0Mi54bWwucmVsc4zP"
            "vQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd"
            "7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lr"
            "qtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYD"
            "AAD//wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9z"
            "bGlkZUxheW91dDMueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi6Nub0YKD"
            "4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyOeHCRRVE8axhyjjul"
            "2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7BPCby+UeE4tFZOiNn"
            "SoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAALAAAAHBw"
            "dC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ0LnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOI"
            "NHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScOb"
            "GPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4"
            "viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwMEFAAGAAgA"
            "AAAhANXRkvG8AAAANwEAACwAAABwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0NS54bWwu"
            "cmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZa"
            "ViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlX"
            "Ec0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGq"
            "bdTs3fYDAAD//wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAAcHB0L3NsaWRlTGF5b3V0cy9f"
            "cmVscy9zbGlkZUxheW91dDYueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi"
            "6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyOeHCRRVE8"
            "axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7BPCby+UeE"
            "4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAA"
            "LAAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ3LnhtbC5yZWxzjM+9CsIwEAfwXfAd"
            "wu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0"
            "FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJ"
            "akgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwME"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAABwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0"
            "OC54bWwucmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4"
            "UmIXvIZaViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX"
            "0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/"
            "tlTLEgGqbdTs3fYDAAD//wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAAcHB0L3NsaWRlTGF5"
            "b3V0cy9fcmVscy9zbGlkZUxheW91dDkueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k"
            "2gbbJOSi6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyO"
            "eHCRRVE8axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7B"
            "PCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwA"
            "AAA3AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxMC54bWwucmVsc4zPvQrC"
            "MBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXc"
            "rsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo"
            "9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD/"
            "/wMAUEsDBBQABgAIAAAAIQDKDhnb1QAAAL4BAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlk"
            "ZUxheW91dDExLnhtbC5yZWxzrJA9SwQxEIZ7wf8Qpjeze4WIXPYaEa6wkfMHDMlsNrj5IBPF+/dGbG7h"
            "CgvL+Xreh9kfvuKqPrlKyMnAqAdQnGx2IXkDb6fnuwdQ0ig5WnNiA2cWOEy3N/tXXqn1I1lCEdUpSQws"
            "rZVHRLELRxKdC6c+mXON1HpZPRay7+QZd8Nwj/WSAdOGqY7OQD26HajTufBf2Hmeg+WnbD8ip3YlAkPs"
            "2R1I1XMzoDVGdoF++6MuyQNe1xj/U0PW4PiFpHHdyFz0N0uj7hE/Zrj5+vQNAAD//wMAUEsDBBQABgAI"
            "AAAAIQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDEyLnht"
            "bC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8"
            "hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmU"
            "qVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsS"
            "Aapt1Ozd9gMAAP//AwBQSwMEFAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAABwcHQvc2xpZGVMYXlvdXRz"
            "L19yZWxzL3NsaWRlTGF5b3V0MTMueG1sLnJlbHOskLtqAzEQRftA/kFMH2l3CxOCtW5CwEUa43zAIM1q"
            "RVYPNEqw/94ybrzgIkXKeZ17mO3uFBbxS4V9ihp62YGgaJL10Wn4On68vILgitHikiJpOBPDbnx+2h5o"
            "wdqOePaZRaNE1jDXmt+UYjNTQJYpU2yTKZWAtZXFqYzmGx2poes2qtwzYFwxxd5qKHs7gDieM/2FnabJ"
            "G3pP5idQrA8ilA8tuwGxOKoapFSBrMdbf5A5OlCPNfr/1ODFW/pErlRWMnf91VIvW8TVTK2+Pl4AAAD/"
            "/wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlk"
            "ZUxheW91dDE0LnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9"
            "/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgM"
            "NCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qF"
            "xdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAABwcHQv"
            "c2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0MTUueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0"
            "dRHBwUX0AY7k2gbbJOSi6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY"
            "9u1y0VxoxFyOeHCRRVE8axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+"
            "I/1jh65zhg7BPCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAA"
            "ACEAyg4Z29UAAAC+AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxNi54bWwu"
            "cmVsc6yQPUsEMRCGe8H/EKY3s3uFiFz2GhGusJHzBwzJbDa4+SATxfv3Rmxu4QoLy/l63ofZH77iqj65"
            "SsjJwKgHUJxsdiF5A2+n57sHUNIoOVpzYgNnFjhMtzf7V16p9SNZQhHVKUkMLK2VR0SxC0cSnQunPplz"
            "jdR6WT0Wsu/kGXfDcI/1kgHThqmOzkA9uh2o07nwX9h5noPlp2w/Iqd2JQJD7NkdSNVzM6A1RnaBfvuj"
            "LskDXtcY/1ND1uD4haRx3chc9DdLo+4RP2a4+fr0DQAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3"
            "AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxNy54bWwucmVsc4zPvQrCMBAH"
            "8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfV"
            "FgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1A"
            "OzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMA"
            "UEsDBBQABgAIAAAAIQDKDhnb1QAAAL4BAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxh"
            "eW91dDE4LnhtbC5yZWxzrJA9SwQxEIZ7wf8Qpjeze4WIXPYaEa6wkfMHDMlsNrj5IBPF+/dGbG7hCgvL"
            "+Xreh9kfvuKqPrlKyMnAqAdQnGx2IXkDb6fnuwdQ0ig5WnNiA2cWOEy3N/tXXqn1I1lCEdUpSQwsrZVH"
            "RLELRxKdC6c+mXON1HpZPRay7+QZd8Nwj/WSAdOGqY7OQD26HajTufBf2Hmeg+WnbD8ip3YlAkPs2R1I"
            "1XMzoDVGdoF++6MuyQNe1xj/U0PW4PiFpHHdyFz0N0uj7hE/Zrj5+vQNAAD//wMAUEsDBBQABgAIAAAA"
            "IQAZV/VM1gAAAL4BAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE5LnhtbC5y"
            "ZWxzrJC7agMxEEX7QP5BTB9pdwsTgrVuQsBFGuN8wCDNakVWDzRKsP/eMm684CJFynmde5jt7hQW8UuF"
            "fYoaetmBoGiS9dFp+Dp+vLyC4IrR4pIiaTgTw258ftoeaMHajnj2mUWjRNYw15rflGIzU0CWKVNskymV"
            "gLWVxamM5hsdqaHrNqrcM2BcMcXeaih7O4A4njP9hZ2myRt6T+YnUKwPIpQPLbsBsTiqGqRUgazHW3+Q"
            "OTpQjzX6/9TgxVv6RK5UVjJ3/dVSL1vE1Uytvj5eAAAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3"
            "AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyMC54bWwucmVsc4zPvQrCMBAH"
            "8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfV"
            "FgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1A"
            "OzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMA"
            "UEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxh"
            "eW91dDIxLnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5c"
            "s39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHL"
            "EMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT"
            "1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwMEFAAGAAgAAAAhAMoOGdvVAAAAvgEAAC0AAABwcHQvc2xp"
            "ZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0MjIueG1sLnJlbHOskD1LBDEQhnvB/xCmN7N7hYhc9hoR"
            "rrCR8wcMyWw2uPkgE8X790ZsbuEKC8v5et6H2R++4qo+uUrIycCoB1CcbHYheQNvp+e7B1DSKDlac2ID"
            "ZxY4TLc3+1deqfUjWUIR1SlJDCytlUdEsQtHEp0Lpz6Zc43Uelk9FrLv5Bl3w3CP9ZIB04apjs5APbod"
            "qNO58F/YeZ6D5adsPyKndiUCQ+zZHUjVczOgNUZ2gX77oy7JA17XGP9TQ9bg+IWkcd3IXPQ3S6PuET9m"
            "uPn69A0AAP//AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAABwcHQvc2xpZGVMYXlvdXRzL19y"
            "ZWxzL3NsaWRlTGF5b3V0MjMueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi"
            "6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyOeHCRRVE8"
            "axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7BPCby+UeE"
            "4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEAyg4Z29UAAAC+AQAA"
            "LQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyNC54bWwucmVsc6yQPUsEMRCGe8H/"
            "EKY3s3uFiFz2GhGusJHzBwzJbDa4+SATxfv3Rmxu4QoLy/l63ofZH77iqj65SsjJwKgHUJxsdiF5A2+n"
            "57sHUNIoOVpzYgNnFjhMtzf7V16p9SNZQhHVKUkMLK2VR0SxC0cSnQunPplzjdR6WT0Wsu/kGXfDcI/1"
            "kgHThqmOzkA9uh2o07nwX9h5noPlp2w/Iqd2JQJD7NkdSNVzM6A1RnaBfvujLskDXtcY/1ND1uD4haRx"
            "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"
            "AKENAAAlAAAAAAAAAAAAAAAAACl7AABwcHQvaGFuZG91dE1hc3RlcnMvaGFuZG91dE1hc3RlcjEueG1s"
            "UEsBAi0AFAAGAAgAAAAhALuah3qIAgAAmgoAABYAAAAAAAAAAAAAAAAATX8AAHBwdC9jb21tZW50QXV0"
            "aG9ycy54bWxQSwECLQAUAAYACAAAACEASy23T2UCAAASBgAAEQAAAAAAAAAAAAAAAAAJggAAcHB0L3By"
            "ZXNQcm9wcy54bWxQSwECLQAUAAYACAAAACEAIqEEIRMCAACrBgAAEQAAAAAAAAAAAAAAAACdhAAAcHB0"
            "L3ZpZXdQcm9wcy54bWxQSwECLQAUAAYACAAAACEAWV6hZR8GAAAHGQAAFAAAAAAAAAAAAAAAAADfhgAA"
            "cHB0L3RoZW1lL3RoZW1lMS54bWxQSwECLQAUAAYACAAAACEAMWRAIPwIAAAfjgAAEwAAAAAAAAAAAAAA"
            "AAAwjQAAcHB0L3RhYmxlU3R5bGVzLnhtbFBLAQItABQABgAIAAAAIQASYtDWcQQAAKQMAAAhAAAAAAAA"
            "AAAAAAAAAF2WAABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MS54bWxQSwECLQAUAAYACAAAACEA"
            "mzGncX4FAABdFAAAIQAAAAAAAAAAAAAAAAANmwAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDIu"
            "eG1sUEsBAi0AFAAGAAgAAAAhAHr1i2GlBAAAew8AACEAAAAAAAAAAAAAAAAAyqAAAHBwdC9zbGlkZUxh"
            "eW91dHMvc2xpZGVMYXlvdXQzLnhtbFBLAQItABQABgAIAAAAIQAHSzb7PAUAALUYAAAhAAAAAAAAAAAA"
            "AAAAAK6lAABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0NC54bWxQSwECLQAUAAYACAAAACEATqf5"
            "LXoFAABHHAAAIQAAAAAAAAAAAAAAAAApqwAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDUueG1s"
            "UEsBAi0AFAAGAAgAAAAhAFn+i0WvBgAAby4AACEAAAAAAAAAAAAAAAAA4rAAAHBwdC9zbGlkZUxheW91"
            "dHMvc2xpZGVMYXlvdXQ2LnhtbFBLAQItABQABgAIAAAAIQBRImNfIwcAAMs+AAAhAAAAAAAAAAAAAAAA"
            "ANC3AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0Ny54bWxQSwECLQAUAAYACAAAACEAle8a4yUC"
            "AAA4BAAAIQAAAAAAAAAAAAAAAAAyvwAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDgueG1sUEsB"
            "Ai0AFAAGAAgAAAAhAKgJpvIkBgAAthUAACEAAAAAAAAAAAAAAAAAlsEAAHBwdC9zbGlkZUxheW91dHMv"
            "c2xpZGVMYXlvdXQ5LnhtbFBLAQItABQABgAIAAAAIQBPB8SMggYAALYZAAAiAAAAAAAAAAAAAAAAAPnH"
            "AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTAueG1sUEsBAi0AFAAGAAgAAAAhAGg7nT5VBgAA"
            "ahUAACIAAAAAAAAAAAAAAAAAu84AAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQxMS54bWxQSwEC"
            "LQAUAAYACAAAACEAtmToy5IEAADoDQAAIgAAAAAAAAAAAAAAAABQ1QAAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDEyLnhtbFBLAQItABQABgAIAAAAIQAkel6soQMAAFkHAAAiAAAAAAAAAAAAAAAAACLa"
            "AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTMueG1sUEsBAi0AFAAGAAgAAAAhAHO71KudBQAA"
            "ShUAACIAAAAAAAAAAAAAAAAAA94AAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQxNC54bWxQSwEC"
            "LQAUAAYACAAAACEAoEOPW9sFAADQHgAAIgAAAAAAAAAAAAAAAADg4wAAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDE1LnhtbFBLAQItABQABgAIAAAAIQCJcEK9JAUAAGwNAAAiAAAAAAAAAAAAAAAAAPvp"
            "AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTYueG1sUEsBAi0AFAAGAAgAAAAhAMOi3AjsBgAA"
            "+DAAACIAAAAAAAAAAAAAAAAAX+8AAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQxNy54bWxQSwEC"
            "LQAUAAYACAAAACEAOOvGMrkFAABdDwAAIgAAAAAAAAAAAAAAAACL9gAAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDE4LnhtbFBLAQItABQABgAIAAAAIQC/SEJxnwMAAFoHAAAiAAAAAAAAAAAAAAAAAIT8"
            "AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTkueG1sUEsBAi0AFAAGAAgAAAAhACbF+x6dBQAA"
            "ShUAACIAAAAAAAAAAAAAAAAAYwABAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQyMC54bWxQSwEC"
            "LQAUAAYACAAAACEAlM1GaNsFAADQHgAAIgAAAAAAAAAAAAAAAABABgEAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDIxLnhtbFBLAQItABQABgAIAAAAIQC3KCqNJQUAAG0NAAAiAAAAAAAAAAAAAAAAAFsM"
            "AQBwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MjIueG1sUEsBAi0AFAAGAAgAAAAhAJPRwXfrBgAA"
            "+DAAACIAAAAAAAAAAAAAAAAAwBEBAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQyMy54bWxQSwEC"
            "LQAUAAYACAAAACEAvLDkkrkFAABdDwAAIgAAAAAAAAAAAAAAAADrGAEAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDI0LnhtbFBLAQItABQABgAIAAAAIQAaE/ML7AIAAFwHAAAiAAAAAAAAAAAAAAAAAOQe"
            "AQBwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MjUueG1sUEsBAi0ACgAAAAAAAAAhANyXZR64GgAA"
            "uBoAABQAAAAAAAAAAAAAAAAAECIBAHBwdC9tZWRpYS9pbWFnZTEucG5nUEsBAi0ACgAAAAAAAAAhAAXp"
            "1EzpEQAA6REAABQAAAAAAAAAAAAAAAAA+jwBAHBwdC9tZWRpYS9pbWFnZTIucG5nUEsBAi0AFAAGAAgA"
            "AAAhAAV25sPxAwAA7hAAABQAAAAAAAAAAAAAAAAAFU8BAHBwdC90aGVtZS90aGVtZTIueG1sUEsBAi0A"
            "FAAGAAgAAAAhALl/7nPtBQAAsBsAABQAAAAAAAAAAAAAAAAAOFMBAHBwdC90aGVtZS90aGVtZTMueG1s"
            "UEsBAi0ACgAAAAAAAAAhAAfzTaa9yAAAvcgAABQAAAAAAAAAAAAAAAAAV1kBAHBwdC9tZWRpYS9pbWFn"
            "ZTMucG5nUEsBAi0ACgAAAAAAAAAhAMrRxx9DvAAAQ7wAABQAAAAAAAAAAAAAAAAARiICAHBwdC9tZWRp"
            "YS9pbWFnZTQucG5nUEsBAi0ACgAAAAAAAAAhANX6HUGAZAEAgGQBABQAAAAAAAAAAAAAAAAAu94CAHBw"
            "dC9tZWRpYS9pbWFnZTUucG5nUEsBAi0ACgAAAAAAAAAhADcnUEOqaQAAqmkAABQAAAAAAAAAAAAAAAAA"
            "bUMEAHBwdC9tZWRpYS9pbWFnZTYucG5nUEsBAi0ACgAAAAAAAAAhANhD3k5kfgAAZH4AABQAAAAAAAAA"
            "AAAAAAAASa0EAHBwdC9tZWRpYS9pbWFnZTcucG5nUEsBAi0ACgAAAAAAAAAhAOEsg2TkZgAA5GYAABQA"
            "AAAAAAAAAAAAAAAA3ysFAHBwdC9tZWRpYS9pbWFnZTgucG5nUEsBAi0ACgAAAAAAAAAhADPkvpQkfQAA"
            "JH0AABQAAAAAAAAAAAAAAAAA9ZIFAHBwdC9tZWRpYS9pbWFnZTkucG5nUEsBAi0AFAAGAAgAAAAhAN1P"
            "qvJ5MwAANjICACEAAAAAAAAAAAAAAAAASxAGAHBwdC9jaGFuZ2VzSW5mb3MvY2hhbmdlc0luZm8xLnht"
            "bFBLAQItABQABgAIAAAAIQBfKtncFwEAAKwBAAAUAAAAAAAAAAAAAAAAAANEBgBwcHQvcmV2aXNpb25J"
            "bmZvLnhtbFBLAQItABQABgAIAAAAIQASNYk5vgEAADIDAAARAAAAAAAAAAAAAAAAAExFBgBkb2NQcm9w"
            "cy9jb3JlLnhtbFBLAQItABQABgAIAAAAIQBNP5vLJAMAAIsHAAAQAAAAAAAAAAAAAAAAADlHBgBkb2NQ"
            "cm9wcy9hcHAueG1sUEsBAi0AFAAGAAgAAAAhAGAtAdUNAQAAkgEAABMAAAAAAAAAAAAAAAAAi0oGAGRv"
            "Y1Byb3BzL2N1c3RvbS54bWxQSwECLQAUAAYACAAAACEA/kXRhwABAADkAgAACwAAAAAAAAAAAAAAAADJ"
            "SwYAX3JlbHMvLnJlbHNQSwECLQAUAAYACAAAACEA2ZEb/N0BAAAHDQAAHwAAAAAAAAAAAAAAAADyTAYA"
            "cHB0L19yZWxzL3ByZXNlbnRhdGlvbi54bWwucmVsc1BLAQItABQABgAIAAAAIQB0Pzl6wgAAACgBAAAe"
            "AAAAAAAAAAAAAAAAAAxPBgBjdXN0b21YbWwvX3JlbHMvaXRlbTEueG1sLnJlbHNQSwECLQAUAAYACAAA"
            "ACEAXJYnIsIAAAAoAQAAHgAAAAAAAAAAAAAAAAAKUAYAY3VzdG9tWG1sL19yZWxzL2l0ZW0yLnhtbC5y"
            "ZWxzUEsBAi0AFAAGAAgAAAAhAHvzAqPDAAAAKAEAAB4AAAAAAAAAAAAAAAAACFEGAGN1c3RvbVhtbC9f"
            "cmVscy9pdGVtMy54bWwucmVsc1BLAQItABQABgAIAAAAIQDIB1wrZAEAAE8QAAAsAAAAAAAAAAAAAAAA"
            "AAdSBgBwcHQvc2xpZGVNYXN0ZXJzL19yZWxzL3NsaWRlTWFzdGVyMS54bWwucmVsc1BLAQItABQABgAI"
            "AAAAIQCQAuz1wQAAADgBAAAgAAAAAAAAAAAAAAAAALVTBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlMS54"
            "bWwucmVsc1BLAQItABQABgAIAAAAIQBL9T3svQAAADcBAAAgAAAAAAAAAAAAAAAAALRUBgBwcHQvc2xp"
            "ZGVzL19yZWxzL3NsaWRlMi54bWwucmVsc1BLAQItABQABgAIAAAAIQAuohua3gAAAEUCAAAgAAAAAAAA"
            "AAAAAAAAAK9VBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlMy54bWwucmVsc1BLAQItABQABgAIAAAAIQBL"
            "9T3svQAAADcBAAAgAAAAAAAAAAAAAAAAAMtWBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlNC54bWwucmVs"
            "c1BLAQItABQABgAIAAAAIQDuDIdh1wAAAL4BAAAgAAAAAAAAAAAAAAAAAMZXBgBwcHQvc2xpZGVzL19y"
            "ZWxzL3NsaWRlNS54bWwucmVsc1BLAQItABQABgAIAAAAIQCQAuz1wQAAADgBAAAgAAAAAAAAAAAAAAAA"
            "ANtYBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlNi54bWwucmVsc1BLAQItABQABgAIAAAAIQA9VWv21gAA"
            "AL4BAAAgAAAAAAAAAAAAAAAAANpZBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlNy54bWwucmVsc1BLAQIt"
            "ABQABgAIAAAAIQCznx8y1wAAAL4BAAAgAAAAAAAAAAAAAAAAAO5aBgBwcHQvc2xpZGVzL19yZWxzL3Ns"
            "aWRlOC54bWwucmVsc1BLAQItABQABgAIAAAAIQBqtNO53gAAAEUCAAAgAAAAAAAAAAAAAAAAAANcBgBw"
            "cHQvc2xpZGVzL19yZWxzL3NsaWRlOS54bWwucmVsc1BLAQItABQABgAIAAAAIQBqtNO53gAAAEUCAAAh"
            "AAAAAAAAAAAAAAAAAB9dBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlMTAueG1sLnJlbHNQSwECLQAUAAYA"
            "CAAAACEAtM9YGbkAAAAkAQAALAAAAAAAAAAAAAAAAAA8XgYAcHB0L25vdGVzTWFzdGVycy9fcmVscy9u"
            "b3Rlc01hc3RlcjEueG1sLnJlbHNQSwECLQAUAAYACAAAACEAk6p9mLkAAAAkAQAAMAAAAAAAAAAAAAAA"
            "AAA/XwYAcHB0L2hhbmRvdXRNYXN0ZXJzL19yZWxzL2hhbmRvdXRNYXN0ZXIxLnhtbC5yZWxzUEsBAi0A"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAARmAGAHBwdC9zbGlkZUxheW91dHMvX3Jl"
            "bHMvc2xpZGVMYXlvdXQxLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAATGEGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyLnhtbC5yZWxzUEsBAi0A"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAUmIGAHBwdC9zbGlkZUxheW91dHMvX3Jl"
            "bHMvc2xpZGVMYXlvdXQzLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAWGMGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ0LnhtbC5yZWxzUEsBAi0A"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAXmQGAHBwdC9zbGlkZUxheW91dHMvX3Jl"
            "bHMvc2xpZGVMYXlvdXQ1LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAZGUGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ2LnhtbC5yZWxzUEsBAi0A"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAamYGAHBwdC9zbGlkZUxheW91dHMvX3Jl"
            "bHMvc2xpZGVMYXlvdXQ3LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAcGcGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ4LnhtbC5yZWxzUEsBAi0A"
            "FAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAdmgGAHBwdC9zbGlkZUxheW91dHMvX3Jl"
            "bHMvc2xpZGVMYXlvdXQ5LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAAAAAAAA"
            "AAAAAAAAfGkGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxMC54bWwucmVsc1BLAQIt"
            "ABQABgAIAAAAIQDKDhnb1QAAAL4BAAAtAAAAAAAAAAAAAAAAAINqBgBwcHQvc2xpZGVMYXlvdXRzL19y"
            "ZWxzL3NsaWRlTGF5b3V0MTEueG1sLnJlbHNQSwECLQAUAAYACAAAACEA1dGS8bwAAAA3AQAALQAAAAAA"
            "AAAAAAAAAACjawYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDEyLnhtbC5yZWxzUEsB"
            "Ai0AFAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAAAAAAAAAAAAAAAAqmwGAHBwdC9zbGlkZUxheW91dHMv"
            "X3JlbHMvc2xpZGVMYXlvdXQxMy54bWwucmVsc1BLAQItABQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAA"
            "AAAAAAAAAAAAAMttBgBwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0MTQueG1sLnJlbHNQ"
            "SwECLQAUAAYACAAAACEA1dGS8bwAAAA3AQAALQAAAAAAAAAAAAAAAADSbgYAcHB0L3NsaWRlTGF5b3V0"
            "cy9fcmVscy9zbGlkZUxheW91dDE1LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhAMoOGdvVAAAAvgEAAC0A"
            "AAAAAAAAAAAAAAAA2W8GAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxNi54bWwucmVs"
            "c1BLAQItABQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAPlwBgBwcHQvc2xpZGVMYXlv"
            "dXRzL19yZWxzL3NsaWRlTGF5b3V0MTcueG1sLnJlbHNQSwECLQAUAAYACAAAACEAyg4Z29UAAAC+AQAA"
            "LQAAAAAAAAAAAAAAAAAAcgYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE4LnhtbC5y"
            "ZWxzUEsBAi0AFAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAAAAAAAAAAAAAAAAIHMGAHBwdC9zbGlkZUxh"
            "eW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxOS54bWwucmVsc1BLAQItABQABgAIAAAAIQDV0ZLxvAAAADcB"
            "AAAtAAAAAAAAAAAAAAAAAEF0BgBwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0MjAueG1s"
            "LnJlbHNQSwECLQAUAAYACAAAACEA1dGS8bwAAAA3AQAALQAAAAAAAAAAAAAAAABIdQYAcHB0L3NsaWRl"
            "TGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDIxLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhAMoOGdvVAAAA"
            "vgEAAC0AAAAAAAAAAAAAAAAAT3YGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyMi54"
            "bWwucmVsc1BLAQItABQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAG93BgBwcHQvc2xp"
            "ZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0MjMueG1sLnJlbHNQSwECLQAUAAYACAAAACEAyg4Z29UA"
            "AAC+AQAALQAAAAAAAAAAAAAAAAB2eAYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDI0"
            "LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAAAAAAAAAAAAAAAAlnkGAHBwdC9z"
            "bGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyNS54bWwucmVsc1BLAQItABQABgAIAAAAIQBtED1T"
            "qwIAAIIeAAATAAAAAAAAAAAAAAAAAJ16BgBbQ29udGVudF9UeXBlc10ueG1sUEsFBgAAAABuAG4APCEA"
            "AHl9BgAAAA=="
        )
        import tempfile, base64 as _b64mod, os as _os
        _tmpl_bytes = _b64mod.b64decode(_TMPL_B64)
        _tmpl_fd, _TMPL_PATH = tempfile.mkstemp(suffix='.pptx', prefix='ltv_tmpl_')
        try:
            with _os.fdopen(_tmpl_fd, 'wb') as _f:
                _f.write(_tmpl_bytes)
        except Exception:
            _os.close(_tmpl_fd)
            raise

        from datetime import date
        from pptx import Presentation
        from pptx.oxml.ns import qn
        from pptx.opc.constants import RELATIONSHIP_TYPE as RT
        import copy

        # ── テンプレートを直接開く ──
        prs = Presentation(_TMPL_PATH)

        # ── ヘルパー関数 ──
        def _copy_slide(prs, from_idx):
            """指定スライドを複製して末尾に追加（マスタ・背景・画像完全維持）"""
            src = prs.slides[from_idx]
            new = prs.slides.add_slide(src.slide_layout)
            src_sp = src.shapes._spTree
            new_sp = new.shapes._spTree
            for el in list(new_sp)[2:]:
                new_sp.remove(el)
            for el in list(src_sp)[2:]:
                new_sp.append(copy.deepcopy(el))
            for rId, rel in src.part.rels.items():
                if rel.reltype == RT.IMAGE:
                    new.part.rels._rels[rId] = rel
            return new

        def _set_text(shape, text):
            """shapeの最初のrun のテキストを設定"""
            if not shape.has_text_frame: return
            tf = shape.text_frame
            if tf.paragraphs and tf.paragraphs[0].runs:
                tf.paragraphs[0].runs[0].text = text

        def _set_multiline(shape, text):
            """改行を含むテキストを段落ごとに設定"""
            if not shape.has_text_frame: return
            lines = text.split('\n')
            tf = shape.text_frame
            for i, para in enumerate(tf.paragraphs):
                if i < len(lines) and para.runs:
                    para.runs[0].text = lines[i]
                elif i < len(lines):
                    para.text = lines[i]

        def _replace_image(slide, shape, buf):
            """shapeの画像をbufの内容で差し替え"""
            blip = shape._element.find('.//' + qn('a:blip'))
            if blip is not None:
                buf.seek(0)
                img_part, new_rId = slide.part.get_or_add_image_part(buf)
                blip.set(qn('r:embed'), new_rId)

        def _set_table_row(tbl, row_idx, values):
            """テーブルのrow_idx行目に値をセット（全runクリア後に設定）"""
            if row_idx >= len(tbl.rows): return
            row = tbl.rows[row_idx]
            for c_idx, v in enumerate(values):
                if c_idx < len(row.cells):
                    tf = row.cells[c_idx].text_frame
                    for para in tf.paragraphs:
                        for run in para.runs:
                            run.text = ''
                    if tf.paragraphs and tf.paragraphs[0].runs:
                        tf.paragraphs[0].runs[0].text = str(v)

        # ── データ期間計算 ──
        _date_cols = [df['start_date']]
        if 'last_purchase_date' in df.columns and df['last_purchase_date'].notna().any():
            _date_cols.append(df['last_purchase_date'])
        if 'end_date' in df.columns and df['end_date'].notna().any():
            _date_cols.append(df['end_date'])
        _all_dates_flat = pd.concat(_date_cols).dropna()
        _data_start = _all_dates_flat.min().strftime('%Y/%m/%d')
        _data_end   = _all_dates_flat.max().strftime('%Y/%m/%d')

        # ══════════════════════════════════════════════
        # Slide 1: タイトル（テンプレートSlide1を更新）
        # ══════════════════════════════════════════════
        s1 = prs.slides[0]
        for sh in s1.shapes:
            if sh.name == 'TextBox 5':
                _set_text(sh, client_name or '')
            elif sh.name == 'TextBox 6':
                _set_text(sh, date.today().strftime('%Y年%m月%d日'))
            elif sh.name == 'TextBox 7':
                _set_text(sh, analyst_name or '')

        # ══════════════════════════════════════════════
        # Slide 2: 分析結果サマリー
        # ══════════════════════════════════════════════
        s2 = prs.slides[1]
        for sh in s2.shapes:
            n = sh.name
            if n == 'テキスト プレースホルダー 6':
                # billing_cycleの内部コードを表示名に変換
                _bc_disp = billing_cycle_display if 'billing_cycle_display' in dir() else billing_cycle.split('←')[0].strip()
                info1 = (
                    f"データ期間: {_data_start} – {_data_end}　|　"
                    f"顧客数: {len(df):,}件　|　解約済み: {df['event'].sum():,}件　|　"
                    f"継続中: {(df['event']==0).sum():,}件　|　"
                    f"平均日次 ARPU: ¥{arpu_daily:,.2f}　|　GPM: {gpm:.0%}"
                )
                info2 = (
                    f"異常値の処理：除外なし　|　{business_type}　|　"
                    f"{_bc_disp}　|　解約時の日割り計算：{'ON' if ltv_offset_days == 0 else 'OFF'}"
                )
                if sh.has_text_frame:
                    tf = sh.text_frame
                    # para0: 全runをクリアして最初のrunに設定
                    if len(tf.paragraphs) >= 1:
                        p0 = tf.paragraphs[0]
                        for r in p0.runs: r.text = ''
                        if p0.runs: p0.runs[0].text = info1
                    # para1: 全runをクリアして最初のrunに設定
                    if len(tf.paragraphs) >= 2:
                        p1 = tf.paragraphs[1]
                        for r in p1.runs: r.text = ''
                        if p1.runs: p1.runs[0].text = info2
            elif n == 'グループ化 26':
                kpi_map = {
                    'TextBox 6':  f'¥{ltv_rev:,.0f}',
                    'TextBox 10': f'¥{cac_upper:,.0f}',
                    'TextBox 11': f'CAC上限（{cac_label}）',
                    'TextBox 14': f'{k:.3f}',
                    'TextBox 18': f'{lam_actual:.0f}日',
                    'TextBox 22': f'{r2:.3f}',
                }
                for grp_sh in sh.shapes:
                    if grp_sh.name in kpi_map:
                        _set_text(grp_sh, kpi_map[grp_sh.name])
            elif n == 'テキスト ボックス 49':
                if sh.has_text_frame:
                    tf = sh.text_frame
                    # para0: '結論'
                    if len(tf.paragraphs) >= 1:
                        p0 = tf.paragraphs[0]
                        for r in p0.runs: r.text = ''
                        if p0.runs: p0.runs[0].text = '結論'
                    # para1: 本文（全runクリアして最初のrunに）
                    if len(tf.paragraphs) >= 2:
                        p1 = tf.paragraphs[1]
                        for r in p1.runs: r.text = ''
                        if p1.runs: p1.runs[0].text = k_summary + r2_summary

        # ══════════════════════════════════════════════
        # Slide 3: 分析の信頼性（Survival Curve / Weibull Plot）
        # ══════════════════════════════════════════════
        s3 = prs.slides[2]
        if k < 0.7:
            k_insight = f"k={k:.3f}（強い初期集中型）: 利用開始直後の体験品質が生死を分ける構造。30日以内の離脱防止施策が最重要。"
        elif k < 1.0:
            k_insight = f"k={k:.3f}（緩やかな初期集中型）: 離脱率は一定に近いが初期にやや多め。オンボーディング改善とリテンション施策を並行実施。"
        elif k < 1.5:
            k_insight = f"k={k:.3f}（逓増型・中程度）: 継続期間が長いほど離脱リスクが増す。1年超の顧客へのエンゲージメント強化が鍵。"
        else:
            k_insight = f"k={k:.3f}（強い逓増型）: 長期顧客ほど急速に離脱。VIP施策・継続特典による長期繋ぎ止めが急務。"
        lam_disp_s3 = lam + ltv_offset_days if business_type == '都度購入型' else lam
        if r2 >= 0.95:
            r2_comment = f"R²={r2:.3f}: 非常に高精度。LTV∞推定値の信頼性は高い。"
        elif r2 >= 0.85:
            r2_comment = f"R²={r2:.3f}: 許容範囲内。推定値に±15%程度の幅を見込んで意思決定を。"
        else:
            r2_comment = f"R²={r2:.3f}: やや低め。データ件数不足または複数の離脱パターンが混在している可能性あり。"
        for sh in s3.shapes:
            if sh.name == 'Picture 3':
                buf1.seek(0); _replace_image(s3, sh, buf1)
            elif sh.name == 'Picture 4':
                buf2.seek(0); _replace_image(s3, sh, buf2)
            elif sh.name == 'TextBox 7':
                k_text = (
                    f"k（形状パラメータ） = {k:.3f}\n"
                    f"→ 左グラフ（Survival Curve）の曲線の急峻さを決める値。k=1で指数分布（一定離脱率）\n"
                    f"→ {k_insight}"
                )
                if sh.has_text_frame:
                    tf = sh.text_frame
                    for para in tf.paragraphs:
                        for run in para.runs: run.text = ''
                    if tf.paragraphs and tf.paragraphs[0].runs:
                        tf.paragraphs[0].runs[0].text = k_text
            elif sh.name == 'TextBox 8':
                lam_text = (
                    f"λ（尺度パラメータ） = {lam_disp_s3:.1f}日（約{lam_disp_s3/365:.1f}年）\n"
                    f"→ 大きいほどLTV∞到達が長期化する。λ日時点での暫定LTV到達率はk値により異なる（k=1のとき63.2%）\n"
                    f"→ {r2_comment}"
                )
                if sh.has_text_frame:
                    tf = sh.text_frame
                    for para in tf.paragraphs:
                        for run in para.runs: run.text = ''
                    if tf.paragraphs and tf.paragraphs[0].runs:
                        tf.paragraphs[0].runs[0].text = lam_text

        # ══════════════════════════════════════════════
        # Slide 4: 暫定LTVテーブル
        # ══════════════════════════════════════════════
        s4 = prs.slides[3]
        all_rows_pp = []
        for h in horizons:
            if business_type == '都度購入型':
                _dorm_pp = dormancy_days or 180
                lh_r = ltv_horizon_spot(k, lam, arpu_0_dorm, arpu_long, h, _dorm_pp)
                lh_g = ltv_horizon_spot(k, lam, arpu_0_dorm*gpm, arpu_long*gpm, h, _dorm_pp)
            else:
                lh_r = ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days)
                lh_g = ltv_horizon_offset(k, lam, gp_daily, h, ltv_offset_days)
            label = f'{h}日' if h < 365 else f'{h//365}年（{h:,}日）'
            all_rows_pp.append((label, lh_r, lh_g, lh_g/cac_n, lh_r/ltv_rev*100))
        all_rows_pp.append((f'λ {round(lam_actual):,}日', lam_rev, lam_gp, lam_gp/cac_n, lam_rev/ltv_rev*100))
        all_rows_pp.append((f'LTV∞到達率: 99%（{int(days_99):,}日）', rev_99, gp_99, gp_99/cac_n, 99.0))

        for sh in s4.shapes:
            if sh.shape_type == 19:  # TABLE
                tbl = sh.table
                for r_idx, (label, lr, lg, lc, pct) in enumerate(all_rows_pp):
                    _set_table_row(tbl, r_idx + 1,
                        [label, f'¥{lr:,.0f}', f'¥{lg:,.0f}', f'¥{lc:,.0f}', f'{pct:.1f}%'])
            elif sh.name == 'テキスト ボックス 17':
                from pptx.util import Pt as _Pt
                sh.top = int(4.55 * 914400)
                if sh.has_text_frame:
                    tf = sh.text_frame
                    k_type = '強い初期離脱型' if k < 0.7 else ('初期離脱型' if k < 1.0 else '逓増型')
                    _s4_lines = [
                        'このテーブルの読み方',
                        f'λ={round(lam_actual):,}日（約{lam_actual/365:.1f}年）は中程度の継続期間で、1〜2年継続する顧客が多いビジネスです。',
                        f'k={k:.3f}（{k_type}）: 契約直後に大量離脱するパターンです。LTV\u221e は大きく見えますが少数の超長期顧客の分が含まれており、99%到達まで長期間かかります。CAC投資判断には暫定LTV（現実的な期間）を使ってください。',
                        f'LTV\u221e（¥{ltv_rev:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。\x0b1年時点でLTV\u221eの{all_rows_pp[1][4]:.1f}%（¥{all_rows_pp[1][1]:,.0f}）、\xa02年時点で{all_rows_pp[2][4]:.1f}%（¥{all_rows_pp[2][1]:,.0f}）、\xa03年時点で{all_rows_pp[3][4]:.1f}%（¥{all_rows_pp[3][1]:,.0f}）に到達します。',
                        '',
                        f'CAC上限（¥{cac_upper:,.0f}）の回収期間：売上ベース 約\xa0{cac_recover_rev_str}\xa0/ 粗利ベース 約\xa0{cac_recover_gp_str}（契約から）',
                    ]
                    for i, para in enumerate(tf.paragraphs):
                        for run in para.runs: run.text = ''
                        if i < len(_s4_lines) and para.runs:
                            para.runs[0].text = _s4_lines[i]
                            para.runs[0].font.size = _Pt(10)

        # ══════════════════════════════════════════════
        # Slide 5: 暫定LTVグラフ
        # ══════════════════════════════════════════════
        # buf_ltvを日本語フォントで再生成
        import matplotlib.pyplot as _plt5
        import matplotlib.font_manager as _fm5
        _fp5 = _fm5.FontProperties(fname=_JP_FONT_PATH) if _JP_FONT_PATH else None
        if _fp5: _plt5.rcParams['font.family'] = _fp5.get_name()
        _fig5, _ax5 = _plt5.subplots(figsize=(10, 3.5))
        _fig5.patch.set_facecolor('#111820'); _ax5.set_facecolor('#111820')
        _ax5.plot(t_range, rev_line, color='#56b4d3', lw=2, label='LTV（売上）')
        _ax5.plot(t_range, gp_line,  color='#a8dadc', lw=2, ls='--', label='LTV（粗利）')
        _ax5.plot(t_range, cac_line, color='#4a7a8a', lw=1.5, ls=':', label='CAC上限')
        _ax5.axhline(ltv_rev, color='#56b4d3', lw=0.8, ls=':', alpha=0.5, label=f'LTV∞ ¥{ltv_rev:,.0f}')
        _ax5.axvline(lam_actual, color='#a8dadc', lw=1.2, ls='--', alpha=0.7)
        _xtick_vals = [180, 365, 730, 1095, 1460, 1825]
        _ax5.set_xticks(_xtick_vals)
        _ax5.set_xticklabels(['180日', '1年', '2年', '3年', '4年', '5年'])
        _ax5.set_xlim(0, x_max + 50)
        _ax5.set_xlabel('継続期間', color='#888', fontsize=9)
        _ax5.set_ylabel('金額（円）', color='#888', fontsize=9)
        _ax5.tick_params(colors='#888')
        _ax5.yaxis.set_major_formatter(_plt5.FuncFormatter(lambda v, _: f'¥{v:,.0f}'))
        _ax5.legend(fontsize=8, framealpha=0.2, labelcolor='white', loc='upper left')
        _ax5.grid(True, alpha=0.2, color='#1a3040')
        for _sp5 in _ax5.spines.values(): _sp5.set_color('#1a3040')
        _fig5.tight_layout()
        _buf5 = io.BytesIO()
        _fig5.savefig(_buf5, format='png', dpi=150, bbox_inches='tight', facecolor='#111820')
        _buf5.seek(0); _plt5.close()

        s5 = prs.slides[4]
        for sh in s5.shapes:
            if sh.name == 'コンテンツ プレースホルダー 6':
                _replace_image(s5, sh, _buf5)

        # ══════════════════════════════════════════════
        # Slide 6〜: セグメント別
        # ══════════════════════════════════════════════
        if segment_cols_input.strip():
            seg_cols_pp = [c.strip() for c in segment_cols_input.split(',')
                          if c.strip() and c.strip() in df.columns]

            # Slide 6: セグメント扉
            s6 = prs.slides[5]
            for sh in s6.shapes:
                if sh.name == 'TextBox 4':
                    if sh.has_text_frame and sh.text_frame.paragraphs:
                        p = sh.text_frame.paragraphs[0]
                        for run in p.runs: run.text = ''
                        if p.runs:
                            p.runs[0].text = '  |  '.join(seg_cols_pp)

            # 元のSlide7〜10を保持して後で複製に使う
            # （prs.slides[6]〜[9]がSlide7〜10）

            first_sc = True
            for sc in seg_cols_pp:
                # セグメントデータ計算
                seg_vals = df[sc].dropna().unique()
                pp_rows = []
                for sv in sorted(seg_vals):
                    df_s = df[df[sc] == sv]
                    if len(df_s) < 10 or df_s['event'].sum() < 5:
                        continue
                    try:
                        km_s = _compute_km_df(df_s)
                        k_s, lam_s, r2_s, _ = _fit_weibull_df(km_s)
                        if k_s is None: continue
                        arpu_s = (df_s['revenue_total'].sum() / df_s['duration'].sum()
                                 if billing_cycle == '日次（都度購入）'
                                 else df_s['arpu_daily'].mean())
                        gp_s = arpu_s * gpm
                        ltv_r, _ = ltv_inf(k_s, lam_s, arpu_s)
                        ltv_g, _ = ltv_inf(k_s, lam_s, gp_s)
                        pp_rows.append({'seg': str(sv), 'n': len(df_s),
                            'ltv_r': ltv_r, 'ltv_g': ltv_g, 'cac': ltv_g/cac_n,
                            'k': k_s, 'lam': lam_s, 'r2': r2_s})
                    except Exception:
                        continue
                if not pp_rows:
                    continue
                pp_rows.sort(key=lambda x: x['ltv_r'], reverse=True)
                best = pp_rows[0]
                avg_ltv = sum(r['ltv_r'] for r in pp_rows) / len(pp_rows)
                avg_cac = sum(r['cac'] for r in pp_rows) / len(pp_rows)
                premium = (best['ltv_r'] - avg_ltv) / avg_ltv * 100
                cac_diff = best['cac'] - avg_cac

                # ── Slide 7: LTV∞比較棒グラフ（常に複製）──
                s7 = _copy_slide(prs, 6)

                for sh in s7.shapes:
                    if sh.name == 'タイトル 4':
                        _set_text(sh, f'{sc}: LTV∞')
                    elif sh.name == 'テキスト プレースホルダー 5':
                        cac_diff_str = (f"+¥{cac_diff:,.0f}高く設定可能"
                                       if cac_diff >= 0 else f"¥{abs(cac_diff):,.0f}低め")
                        if sh.has_text_frame:
                            tf = sh.text_frame
                            lines = [
                                f"Top Pick　{best['seg']}",
                                f"LTV∞(売上): ¥{best['ltv_r']:,.0f}（全セグメント平均比 +{premium:.1f}%）　|　許容CAC上限 ¥{best['cac']:,.0f}（全セグメント平均より{cac_diff_str}）"
                            ]
                            for i, para in enumerate(tf.paragraphs):
                                for run in para.runs: run.text = ''
                                if i < len(lines) and para.runs:
                                    para.runs[0].text = lines[i]
                    elif sh.name == 'コンテンツ プレースホルダー 18':
                        # 棒グラフ生成（日本語フォント設定）
                        import matplotlib.pyplot as _plt_bar
                        import matplotlib.ticker as _mticker
                        import matplotlib.font_manager as _fm_bar
                        _fp_bar = _fm_bar.FontProperties(fname=_JP_FONT_PATH) if _JP_FONT_PATH else None
                        if _fp_bar: _plt_bar.rcParams['font.family'] = _fp_bar.get_name()
                        _fig, _ax = _plt_bar.subplots(figsize=(10.7, 4.2))
                        _fig.patch.set_facecolor('#111820'); _ax.set_facecolor('#111820')
                        _segs = [r['seg'] for r in pp_rows]
                        _ltvs = [r['ltv_r'] for r in pp_rows]
                        _cols = ['#56b4d3' if r['seg'] == best['seg'] else '#a8dadc' for r in pp_rows]
                        _xs = range(len(_segs))
                        _bars = _ax.bar(_xs, _ltvs, color=_cols, width=0.55)
                        _ax.set_xticks(list(_xs))
                        _ax.set_xticklabels(_segs, fontsize=8, color='#cccccc')
                        for _b, _v in zip(_bars, _ltvs):
                            _ax.text(_b.get_x()+_b.get_width()/2, _b.get_height()+max(_ltvs)*0.01,
                                    f'¥{_v:,.0f}', ha='center', va='bottom', fontsize=9, color='#cccccc')
                        _ax.set_ylabel('LTV∞（¥）', color='#888', fontsize=9)
                        _ax.tick_params(axis='y', colors='#888', labelsize=8)
                        _ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v,_: f'¥{v:,.0f}'))
                        _ax.grid(axis='y', alpha=0.2, color='#1a3040')
                        for _sp in _ax.spines.values(): _sp.set_color('#1a3040')
                        _fig.tight_layout()
                        _buf_bar = io.BytesIO()
                        _fig.savefig(_buf_bar, format='png', dpi=130, bbox_inches='tight', facecolor='#111820')
                        _buf_bar.seek(0); _plt_bar.close()
                        _replace_image(s7, sh, _buf_bar)

                # ── Slide 8: セグメントサマリー（常に複製）──
                s8 = _copy_slide(prs, 7)

                for sh in s8.shapes:
                    if sh.name == 'コンテンツ プレースホルダー 6':
                        # サマリーテーブル画像生成
                        import matplotlib.pyplot as _plt_t
                        import matplotlib.patches as _mpatch
                        import matplotlib.font_manager as _fm_t
                        _fp_t = _fm_t.FontProperties(fname=_JP_FONT_PATH) if _JP_FONT_PATH else None
                        if _fp_t: _plt_t.rcParams['font.family'] = _fp_t.get_name()
                        _n_rows = len(pp_rows) + 1  # セグメント行 + 加重平均行
                        _fig_h = max(1.2, _n_rows * 0.32 + 0.4)
                        _fig_t, _ax_t = _plt_t.subplots(figsize=(11.8, _fig_h))
                        _fig_t.patch.set_facecolor('#111820'); _ax_t.set_facecolor('#111820')
                        _ax_t.axis('off')
                        _hdrs = ['セグメント', '顧客数', 'LTV∞(売上)', 'LTV∞(粗利)', 'CAC上限(粗利)', 'k', 'λ(日)', 'R²']
                        _col_w = [0.22, 0.08, 0.13, 0.12, 0.13, 0.08, 0.10, 0.07]
                        _norm_w = [w/sum(_col_w) for w in _col_w]
                        _col_x = [sum(_norm_w[:i]) for i in range(len(_norm_w))]
                        # ヘッダ
                        _hdr_h = 1.0 / (_n_rows + 1)
                        _bg = _mpatch.FancyBboxPatch((0, 1-_hdr_h), 1, _hdr_h,
                            boxstyle='square,pad=0', facecolor='#0D1F2D', transform=_ax_t.transAxes)
                        _ax_t.add_patch(_bg)
                        for _cx, _cw, _h in zip(_col_x, _norm_w, _hdrs):
                            _al = 'left' if _cx == 0 else 'right'
                            _tx = _cx + (0.01 if _al == 'left' else _cw - 0.01)
                            _ax_t.text(_tx, 1-_hdr_h/2, _h, transform=_ax_t.transAxes,
                                      ha=_al, va='center', fontsize=7.5, color='#56B4D3', fontweight='bold')
                        # データ行
                        _wa_n = sum(r['n'] for r in pp_rows)
                        _wa_r = sum(r['ltv_r']*r['n'] for r in pp_rows) / _wa_n
                        _wa_g = sum(r['ltv_g']*r['n'] for r in pp_rows) / _wa_n
                        _all_rows_t = pp_rows + [{'seg':'加重平均','n':_wa_n,'ltv_r':_wa_r,'ltv_g':_wa_g,
                            'cac':_wa_g/cac_n,'k':None,'lam':None,'r2':None}]
                        for _ri, _row in enumerate(_all_rows_t):
                            _y = 1 - _hdr_h - (_ri+1)*((1-_hdr_h)/len(_all_rows_t))
                            _rh = (1-_hdr_h)/len(_all_rows_t)
                            _is_wa = _row['seg'] == '加重平均'
                            _is_best = not _is_wa and _row['seg'] == best['seg']
                            _bg2 = '#1A3A4A' if _is_wa else ('#0D1F2D' if _is_best else ('#0D1520' if _ri%2==0 else '#0A1018'))
                            _p = _mpatch.FancyBboxPatch((0,_y), 1, _rh,
                                boxstyle='square,pad=0', facecolor=_bg2, transform=_ax_t.transAxes)
                            _ax_t.add_patch(_p)
                            _vals = [_row['seg'], f"{_row['n']:,}",
                                f"¥{_row['ltv_r']:,.0f}", f"¥{_row['ltv_g']:,.0f}", f"¥{_row['cac']:,.0f}",
                                f"{_row['k']:.3f}" if _row['k'] else '—',
                                f"{_row['lam']:.1f}" if _row['lam'] else '—',
                                f"{_row['r2']:.3f}" if _row['r2'] else '—']
                            _tc = '#A8DADC' if _is_wa or _is_best else '#E8E4DC'
                            for _cx2, _cw2, _v2 in zip(_col_x, _norm_w, _vals):
                                _al2 = 'left' if _cx2 == 0 else 'right'
                                _tx2 = _cx2 + (0.01 if _al2=='left' else _cw2-0.01)
                                _ax_t.text(_tx2, _y+_rh/2, _v2, transform=_ax_t.transAxes,
                                          ha=_al2, va='center', fontsize=7.5, color=_tc)
                        _fig_t.tight_layout(pad=0)
                        _buf_t = io.BytesIO()
                        _fig_t.savefig(_buf_t, format='png', dpi=150, bbox_inches='tight', facecolor='#111820')
                        _buf_t.seek(0); _plt_t.close()
                        _replace_image(s8, sh, _buf_t)
                    elif sh.name == 'テキスト ボックス 9':
                        _wa_n2 = sum(r['n'] for r in pp_rows)
                        _wa_r2 = sum(r['ltv_r']*r['n'] for r in pp_rows) / _wa_n2
                        _diff_p = (_wa_r2 - ltv_rev) / ltv_rev * 100
                        _note_body = (
                            f"\xa0— 加重平均行は各セグメントを個別フィット後に顧客数で重み付け平均した値です。"
                            f"全体LTV∞（¥{ltv_rev:,.0f}）との差（{_diff_p:+.1f}%）は統計的に正常な現象です。"
                            f"広告投資にはセグメント別、全体評価には全体LTV∞を参照してください。"
                        )
                        if sh.has_text_frame and sh.text_frame.paragraphs:
                            p = sh.text_frame.paragraphs[0]
                            for run in p.runs: run.text = ''
                            if len(p.runs) >= 1: p.runs[0].text = 'NOTE'
                            if len(p.runs) >= 2: p.runs[1].text = _note_body
                            elif len(p.runs) == 1: p.runs[0].text = 'NOTE' + _note_body

                # ── Slide 9〜: セグメント詳細 ──
                for sv in sorted(seg_vals):
                    df_sv = df[df[sc] == sv]
                    if len(df_sv) < 10 or df_sv['event'].sum() < 5:
                        continue
                    try:
                        km_sv = _compute_km_df(df_sv)
                        k_sv, lam_sv, r2_sv, _ = _fit_weibull_df(km_sv)
                        if k_sv is None: continue
                        arpu_sv = (df_sv['revenue_total'].sum() / df_sv['duration'].sum()
                                  if billing_cycle == '日次（都度購入）'
                                  else df_sv['arpu_daily'].mean())
                        gp_sv = arpu_sv * gpm
                        ltv_r_sv, _ = ltv_inf(k_sv, lam_sv, arpu_sv)
                        lam_sv_disp = lam_sv + ltv_offset_days if business_type == '都度購入型' else lam_sv
                        is_best_sv = str(sv) == best['seg']

                        # Top Pickはindex=8、それ以外はindex=9をコピー
                        tmpl_idx = 8 if is_best_sv else 9
                        s9 = _copy_slide(prs, tmpl_idx)

                        # タイトル更新
                        for sh in s9.shapes:
                            if sh.name == 'タイトル 8':
                                # 全runをクリアしてから設定
                                if sh.has_text_frame:
                                    for para in sh.text_frame.paragraphs:
                                        for run in para.runs: run.text = ''
                                    if sh.text_frame.paragraphs and sh.text_frame.paragraphs[0].runs:
                                        sh.text_frame.paragraphs[0].runs[0].text = f'{sc}: {str(sv)}'
                            elif sh.name == 'Picture 6':
                                # Survival Curve
                                import matplotlib.pyplot as _plt_sv
                                _fig_sv1, _ax_sv1 = _plt_sv.subplots(figsize=(5.4, 3.1))
                                _fig_sv1.patch.set_facecolor('#111820'); _ax_sv1.set_facecolor('#111820')
                                _ax_sv1.step(km_sv['t'], km_sv['S'], color='#56b4d3', lw=1.5, label='KM Curve (Observed)')
                                _t_ra = km_sv['t'].values
                                _ax_sv1.plot(_t_ra, [float(weibull_s(t, k_sv, lam_sv)) for t in _t_ra],
                                            '--', color='#a8dadc', lw=1.2, label='Weibull Fit')
                                _ax_sv1.set_xlabel('Days', color='#888', fontsize=8)
                                _ax_sv1.set_ylabel('Survival Rate S(t)', color='#888', fontsize=8)
                                _ax_sv1.tick_params(colors='#666', labelsize=7)
                                _ax_sv1.legend(fontsize=7, framealpha=0.2)
                                _ax_sv1.grid(True, alpha=0.2)
                                _ax_sv1.set_title('Survival Curve', color='#ccc', fontsize=9)
                                _fig_sv1.tight_layout()
                                _buf_sv1 = io.BytesIO()
                                _fig_sv1.savefig(_buf_sv1, format='png', dpi=130, bbox_inches='tight', facecolor='#111820')
                                _buf_sv1.seek(0); _plt_sv.close()
                                _replace_image(s9, sh, _buf_sv1)
                            elif sh.name == 'Picture 7':
                                # Weibull Plot
                                import numpy as _np_sv
                                _km_fit = km_sv[km_sv['S'] > 0]
                                _ln_t = _np_sv.log(_km_fit['t'].values.astype(float) + 1e-10)
                                _ln_ns = _np_sv.log(-_np_sv.log(_km_fit['S'].values.astype(float) + 1e-15))
                                _valid = _np_sv.isfinite(_ln_t) & _np_sv.isfinite(_ln_ns)
                                _ln_t, _ln_ns = _ln_t[_valid], _ln_ns[_valid]
                                _sl, _it, _, _, _ = __import__('scipy').stats.linregress(_ln_t, _ln_ns)
                                _x_l = _np_sv.linspace(_ln_t.min(), _ln_t.max(), 100)
                                import matplotlib.pyplot as _plt_sv2
                                _fig_sv2, _ax_sv2 = _plt_sv2.subplots(figsize=(5.4, 3.1))
                                _fig_sv2.patch.set_facecolor('#111820'); _ax_sv2.set_facecolor('#111820')
                                _ax_sv2.scatter(_ln_t, _ln_ns, color='#56b4d3', s=18, alpha=0.75, label='Observed')
                                _ax_sv2.plot(_x_l, _sl*_x_l+_it, '--', color='#a8dadc', lw=1.5, label=f'R²={r2_sv:.3f}')
                                _ax_sv2.annotate(f'y = {_sl:.4f}x + {_it:.4f}', xy=(0.05,0.93), xycoords='axes fraction', color='#777', fontsize=8)
                                _ax_sv2.set_xlabel('ln(t)', color='#888', fontsize=8)
                                _ax_sv2.set_ylabel('ln(−ln(S(t)))', color='#888', fontsize=8)
                                _ax_sv2.tick_params(colors='#666', labelsize=7)
                                _ax_sv2.legend(fontsize=7, framealpha=0.2)
                                _ax_sv2.grid(True, alpha=0.2)
                                _ax_sv2.set_title('Weibull Linearization Plot', color='#ccc', fontsize=9)
                                _fig_sv2.tight_layout()
                                _buf_sv2 = io.BytesIO()
                                _fig_sv2.savefig(_buf_sv2, format='png', dpi=130, bbox_inches='tight', facecolor='#111820')
                                _buf_sv2.seek(0); _plt_sv2.close()
                                _replace_image(s9, sh, _buf_sv2)
                            elif sh.shape_type == 19:  # TABLE
                                tbl9 = sh.table
                                _sv_rows = []
                                for _h in horizons:
                                    _lr2 = ltv_horizon_offset(k_sv, lam_sv, arpu_sv, _h, ltv_offset_days)
                                    _lg2 = _lr2 * gpm
                                    # ラベルを重複しない形式で
                                    if _h < 365:
                                        _label2 = f'{_h}日'
                                    else:
                                        _label2 = f'{_h//365}年（{_h:,}日）'
                                    _sv_rows.append((_label2, _lr2, _lg2, _lg2/cac_n, _lr2/ltv_r_sv*100))
                                _lr_lam = ltv_horizon_offset(k_sv, lam_sv, arpu_sv, int(lam_sv), ltv_offset_days)
                                _lg_lam = _lr_lam * gpm
                                _sv_rows.append((f'λ {round(lam_sv_disp):,}日', _lr_lam, _lg_lam, _lg_lam/cac_n, _lr_lam/ltv_r_sv*100))
                                try:
                                    from scipy.optimize import brentq as _bq2
                                    _d99 = _bq2(lambda _hh2: ltv_horizon_offset(k_sv, lam_sv, arpu_sv, _hh2, ltv_offset_days)/ltv_r_sv-0.99, 1, 365000)
                                    _lr99 = ltv_horizon_offset(k_sv, lam_sv, arpu_sv, _d99, ltv_offset_days)
                                    _lg99 = _lr99 * gpm
                                    _sv_rows.append((f'LTV∞到達率: 99%（{int(_d99):,}日）', _lr99, _lg99, _lg99/cac_n, 99.0))
                                except Exception:
                                    _sv_rows.append(('LTV∞到達率: 99%', ltv_r_sv, ltv_r_sv*gpm, ltv_r_sv*gpm/cac_n, 99.0))
                                for r_idx, row_data in enumerate(_sv_rows):
                                    if r_idx + 1 < len(tbl9.rows):
                                        _row9 = tbl9.rows[r_idx + 1]
                                        _vals9 = [row_data[0], f'¥{row_data[1]:,.0f}', f'¥{row_data[2]:,.0f}',
                                                 f'¥{row_data[3]:,.0f}', f'{row_data[4]:.1f}%']
                                        for _ci, _v in enumerate(_vals9):
                                            if _ci < len(_row9.cells):
                                                tf9 = _row9.cells[_ci].text_frame
                                                for para9 in tf9.paragraphs:
                                                    for run9 in para9.runs: run9.text = ''
                                                if tf9.paragraphs and tf9.paragraphs[0].runs:
                                                    tf9.paragraphs[0].runs[0].text = _v
                    except Exception:
                        continue

                first_sc = False

            # テンプレートの元スライド7〜10（index 6〜9）を削除
            # ループで複製済みなので不要
            _sldIdLst = prs.slides._sldIdLst
            # 削除対象：セグメント処理前の元スライド（先頭から6〜9番目）
            # 複製は末尾に追加されているので、index 6〜9を削除
            _ids_to_remove = list(_sldIdLst)[6:10]
            for _sldId in _ids_to_remove:
                _rId = _sldId.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                try:
                    if _rId: prs.part.drop_rel(_rId)
                except Exception:
                    pass
                _sldIdLst.remove(_sldId)

        else:
            # セグメントなし：Slide7〜10をPRSから削除
            # 末尾から削除
            sldIdLst = prs.slides._sldIdLst
            while len(prs.slides) > 6:
                last_sldId = list(sldIdLst)[-1]
                rId = last_sldId.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if rId:
                    try:
                        prs.part.drop_rel(rId)
                    except Exception:
                        pass
                sldIdLst.remove(last_sldId)

        pptx_buf = io.BytesIO()
        prs.save(pptx_buf)
        pptx_buf.seek(0)
        import base64 as _b64
        _pp_b64 = _b64.b64encode(pptx_buf.read()).decode()
        _fn_pp = f"LTV分析_{client_name or 'report'}.pptx"
        _pp_href = f'<a href="data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{_pp_b64}" download="{_fn_pp}" class="dl-btn">.pptx</a>'
        _pp_html = _pp_href
    except ImportError:
        _pp_html = '<span class="dl-btn-err">.pptx 未対応</span>'
    except Exception as e:
        import traceback as _tb
        _tb_str = _tb.format_exc().replace('\n', ' | ')[-300:]
        _pp_html = f'<span class="dl-btn-err">.pptx エラー: {str(e)[:150]}<br><small style="font-size:0.6rem;opacity:0.7">{_tb_str}</small></span>'

# ── PDF export ────────────────────────────────────────────────
if True:
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
            ['Weibull λ（尺度パラメータ）', f'{lam + ltv_offset_days if business_type == "都度購入型" else lam:.1f}日（約{(lam + ltv_offset_days if business_type == "都度購入型" else lam)/365:.1f}年）'],
            ['R²（フィット精度）', f'{r2:.4f}  →  {" 良好（0.9以上）" if r2 >= 0.9 else "△ やや低め（0.9未満）"}'],
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
            f"Weibull k={k:.4f} / λ={lam_display:.1f}日 / R²={r2:.4f}"
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
                        km_s = _compute_km_df(df_s)
                        k_s, lam_s, r2_s, _ = _fit_weibull_df(km_s)
                        if k_s is None: continue
                        arpu_s = df_s['revenue_total'].sum() / df_s['duration'].sum() if billing_cycle == '日次（都度購入）' else df_s['arpu_daily'].mean()
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
                        f"優先獲得推奨：{best_pdf['seg']}　"
                        f"LTV∞（売上）¥{best_pdf['ltv_r']:,.0f}（全平均比+{prem:.1f}%）　"
                        f"CAC上限（粗利）¥{best_pdf['cac']:,.0f}",
                        rec_style
                    ))


                # 全セグメントの詳細分析（2グラフ横並び）
                story.append(Paragraph(f'全セグメント詳細（{sc}）', h2_style))
                for sv in sorted(seg_vals):
                    df_sv2 = df[df[sc] == sv]
                    if len(df_sv2) < 10 or df_sv2['event'].sum() < 5:
                        continue
                    try:
                        km_sv2 = _compute_km_df(df_sv2)
                        k_sv2, lam_sv2, r2_sv2, _ = _fit_weibull_df(km_sv2)
                        if k_sv2 is None: continue
                        arpu_sv2 = df_sv2['revenue_total'].sum() / df_sv2['duration'].sum() if billing_cycle == '日次（都度購入）' else df_sv2['arpu_daily'].mean()
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
        import base64 as _b64
        _pdf_b64 = _b64.b64encode(pdf_buf.read()).decode()
        _fn_pdf = f"LTV分析_{client_name or 'report'}.pdf"
        _pdf_href = f'<a href="data:application/pdf;base64,{_pdf_b64}" download="{_fn_pdf}" class="dl-btn">.pdf</a>'
        _pdf_html = _pdf_href
    except ImportError:
        _pdf_html = '<span class="dl-btn-err">.pdf 未対応</span>'
    except Exception as e:
        _pdf_html = f'<span class="dl-btn-err">.pdf エラー</span>'

# ── 3ボタンまとめて表示 ───────────────────────────────────────
st.markdown(f"""
<style>
.dl-row {{ display:flex; gap:8px; align-items:center; margin-top:4px; }}
a.dl-btn {{
    display:inline-flex; align-items:center; justify-content:center;
    width:72px; height:30px;
    background:#0d1f2d; color:#a8dadc;
    border:1.5px solid #56b4d3; border-radius:6px;
    font-size:0.78rem; font-weight:600; letter-spacing:0.04em;
    text-decoration:none;
    transition:background 0.2s, color 0.2s;
}}
a.dl-btn:hover {{ background:#56b4d3; color:#0d1f2d; }}
a.dl-btn:focus, a.dl-btn:focus-visible, a.dl-btn:active {{
    outline: none !important;
    box-shadow: none !important;
    border: 1.5px solid #56b4d3 !important;
    background: #0d1f2d;
    color: #a8dadc;
}}
.dl-btn-err {{ font-size:0.75rem; color:#888; }}
</style>
<div class="dl-row">
  {_xl_html if '_xl_html' in dir() else ''}
  {_pp_html if '_pp_html' in dir() else ''}
  {_pdf_html if '_pdf_html' in dir() else ''}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Segment Analysis (Advanced)
# ══════════════════════════════════════════════════════════════

# セグメント結果を保存（エクスポート用）
all_seg_results = {}  # {seg_col: seg_df}

if segment_cols_input.strip():
    seg_cols = [c.strip() for c in segment_cols_input.split(',') if c.strip()]
    valid_seg_cols = [c for c in seg_cols if c in df.columns]
    invalid_seg_cols = [c for c in seg_cols if c not in df.columns]

    if invalid_seg_cols:
        st.warning(f" 以下の列が見つかりませんでした: {invalid_seg_cols}")

    MAX_SEG_COLS = 5
    if len(valid_seg_cols) > MAX_SEG_COLS:
        st.warning(
            f" セグメント軸は最大{MAX_SEG_COLS}列まで指定できます（処理速度の確保のため）。"
            f"現在{len(valid_seg_cols)}列指定されています。"
            f"先頭{MAX_SEG_COLS}列のみ分析します。残りの列は別途入力してください。"
        )
        valid_seg_cols = valid_seg_cols[:MAX_SEG_COLS]

    if valid_seg_cols:
        st.markdown("<div class='section-title'>セグメント別 LTV 分析</div>", unsafe_allow_html=True)

        # セグメント一覧リンク
        _seg_links = " &nbsp;|&nbsp; ".join(
            f"<a href='#{c}_anchor' style='color:#56b4d3; font-size:0.78rem; text-decoration:none;'>{c}</a>"
            for c in valid_seg_cols
        )
        st.markdown(f"<div style='margin-bottom:12px; font-size:0.78rem; color:#888;'>{_seg_links}</div>", unsafe_allow_html=True)

        for seg_col in valid_seg_cols:
            st.markdown(f"<div id='{seg_col}_anchor' style='position:relative; top:-120px; visibility:hidden;'></div><div style='font-size:0.82rem; font-weight:600; color:#a8dadc; margin:16px 0 4px 0; letter-spacing:0.05em;'>{seg_col}</div>", unsafe_allow_html=True)

            seg_values = df[seg_col].dropna().unique()
            if len(seg_values) > 50:
                st.warning(f" `{seg_col}` のユニーク値が{len(seg_values)}個あります。50個以下にしてください。")
                continue
            elif len(seg_values) > 20:
                st.info(f"`{seg_col}` のユニーク値が{len(seg_values)}個あります。計算に少し時間がかかります。")

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
                    # ── 全体分析と完全同一のロジック ──
                    # 1. オフセット処理
                    if ltv_offset_days > 0:
                        df_seg_fit = df_seg.copy()
                        df_seg_fit['duration'] = df_seg_fit['duration'] - ltv_offset_days
                        df_seg_fit.loc[df_seg_fit['duration'] <= 0, 'event'] = 0
                        df_seg_fit.loc[df_seg_fit['duration'] <= 0, 'duration'] = 1
                        df_seg_fit['duration'] = df_seg_fit['duration'].clip(lower=1)
                        km_seg = _compute_km_df(df_seg_fit)
                    else:
                        km_seg = _compute_km_df(df_seg)

                    # 2. Weibullフィット
                    k_s, lam_s, r2_s, _ = _fit_weibull_df(km_seg)
                    if k_s is None:
                        continue

                    # 3. ARPU計算（全体と同じ分岐）
                    if business_type == '都度購入型' and 'last_purchase_date' in df_seg.columns:
                        _dorm_s = dormancy_days or 180
                        _gap_s = (df_seg['last_purchase_date'] - df_seg['start_date']).dt.days.fillna(-1)
                        _days_s = (today - df_seg['last_purchase_date']).dt.days.fillna(0)
                        _single_mask_s = (_gap_s == 0) & (_days_s >= _dorm_s)
                        _long_mask_s   = (_gap_s > 0)
                        _single_df_s   = df_seg[_single_mask_s]
                        _long_df_s     = df_seg[_long_mask_s]
                        arpu_short_s = _single_df_s['revenue_total'].sum() / len(_single_df_s) / _dorm_s if len(_single_df_s) > 0 else 0.0
                        arpu_long_s  = _long_df_s['revenue_total'].sum() / _long_df_s['duration'].sum() if len(_long_df_s) > 0 else df_seg['revenue_total'].sum() / df_seg['duration'].sum()
                        _n_s = len(_single_df_s) + len(_long_df_s)
                        _w_short_s = len(_single_df_s) / _n_s if _n_s > 0 else 0.5
                        _w_long_s  = len(_long_df_s)  / _n_s if _n_s > 0 else 0.5
                        arpu_0_dorm_s = arpu_short_s * _w_short_s + arpu_long_s * _w_long_s
                        arpu_s = arpu_long_s
                    else:
                        arpu_s = df_seg['arpu_daily'].mean()
                        arpu_long_s = arpu_s
                        arpu_0_dorm_s = arpu_s
                        _dorm_s = ltv_offset_days

                    gp_s = arpu_s * gpm

                    # 4. LTV∞計算（全体と同じ分岐）
                    if business_type == '都度購入型':
                        _ltv_short_s = _dorm_s * arpu_0_dorm_s
                        _ltv_long_s, _ = ltv_inf_offset(k_s, lam_s, arpu_long_s, 0)
                        ltv_rev_s = _ltv_short_s + _ltv_long_s
                        _gp_short_s = _dorm_s * (arpu_0_dorm_s * gpm)
                        _gp_long_s, _ = ltv_inf_offset(k_s, lam_s, arpu_long_s * gpm, 0)
                        ltv_gp_s = _gp_short_s + _gp_long_s
                    else:
                        ltv_rev_s, _ = ltv_inf_offset(k_s, lam_s, arpu_s, ltv_offset_days)
                        ltv_gp_s, _  = ltv_inf_offset(k_s, lam_s, gp_s,  ltv_offset_days)

                    cac_s          = ltv_gp_s / cac_n
                    total_ltv_s    = ltv_rev_s * len(df_seg)
                    lam_s_disp     = lam_s + ltv_offset_days
                    efficiency     = (ltv_gp_s / cac_input) if cac_known else None
                    priority_score = ltv_rev_s * len(df_seg)

                    seg_results.append({
                        'セグメント':    seg_val,
                        '顧客数':        len(df_seg),
                        'LTV∞（売上）':  ltv_rev_s,
                        'LTV∞（粗利）':  ltv_gp_s,
                        'CAC上限（粗利）': cac_s,
                        '総ポテンシャル': total_ltv_s,
                        'k':             k_s,
                        'λ（日）':       lam_s_disp,
                        'λ_raw':         lam_s,
                        'arpu_s':        arpu_s,
                        'arpu_long_s':   arpu_long_s,
                        'arpu_0_dorm_s': arpu_0_dorm_s,
                        'R²':            r2_s,
                        '獲得効率':      efficiency,
                        '優先スコア':    priority_score,
                    })
                except Exception:
                    continue

            if not seg_results:
                st.caption("分析に十分なデータがありませんでした。")
                continue

            seg_df = pd.DataFrame(seg_results).sort_values('LTV∞（売上）', ascending=False).reset_index(drop=True)
            all_seg_results[seg_col] = seg_df  # エクスポート用に保存

            progress_bar.empty()

            # Top Pick（タイトルとグラフの間）
            if seg_results:
                best_seg    = seg_df.iloc[0]
                avg_ltv     = seg_df['LTV∞（売上）'].mean()
                premium     = (best_seg['LTV∞（売上）'] - avg_ltv) / avg_ltv * 100
                cac_best    = best_seg['CAC上限（粗利）']
                cac_avg_seg = seg_df['CAC上限（粗利）'].mean()
                cac_diff    = cac_best - cac_avg_seg
                cac_str     = f"許容CAC上限 ¥{cac_best:,.0f}（全セグメント平均より¥{abs(cac_diff):,.0f}{'高く' if cac_diff >= 0 else '低く'}設定可能）"
                st.markdown(f"""
<div style='background:#0d1f2d; border:1px solid #1a3a4a; border-left:3px solid #56b4d3; border-radius:8px; padding:10px 16px; margin-bottom:8px; font-size:0.82rem; color:#ccc;'>
  <span style='font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:#7ab4c4;'>Top Pick</span>　<b style='color:#a8dadc;'>{best_seg['セグメント']}</b>　
  <span style='color:#888; margin:0 6px;'>|</span>
  LTV∞(売上): <b style='color:#a8dadc;'>¥{best_seg['LTV∞（売上）']:,.0f}</b>（全セグメント平均比 +{premium:.1f}%）
  <span style='color:#888; margin:0 6px;'>|</span>
  {cac_str}
</div>""", unsafe_allow_html=True)

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
                title_text='',
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
            display_df = display_df.drop(columns=['総ポテンシャル', '優先スコア', 'λ_raw', 'arpu_s', 'arpu_long_s', 'arpu_0_dorm_s'], errors='ignore')
            if not cac_known:
                display_df = display_df.drop(columns=['獲得効率'], errors='ignore')

            # セグメント加重平均LTV∞を計算してテーブル上に表示
            total_n = seg_df['顧客数'].sum()
            weighted_ltv_rev = (seg_df['LTV∞（売上）'] * seg_df['顧客数']).sum() / total_n
            weighted_ltv_gp  = (seg_df['LTV∞（粗利）'] * seg_df['顧客数']).sum() / total_n
            weighted_cac     = weighted_ltv_gp / cac_n
            diff_pct = (weighted_ltv_rev - ltv_rev) / ltv_rev * 100
            diff_str = f"+{diff_pct:.1f}%" if diff_pct >= 0 else f"{diff_pct:.1f}%"
            diff_color = "#a8dadc" if diff_pct >= 0 else "#e8a0a0"

            # HTMLテーブルで描画（数字右寄せ・列幅均等）
            ACCENT   = '#56b4d3'
            BG_HEAD  = '#0d1f2d'
            BG_ROW1  = '#0d1520'
            BG_ROW2  = '#0a1018'
            num_cols = ['顧客数', 'LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）', 'k', 'λ（日）', 'R²']
            if cac_known:
                num_cols.append('獲得効率')
            cols = [c for c in display_df.columns]
            n_cols = len(cols)
            seg_col_w = 20
            num_col_w = round((100 - seg_col_w) / (n_cols - 1), 1)

            # ヘッダー
            seg_html_rows = ''
            for i, row in display_df.iterrows():
                bg = BG_ROW1 if i % 2 == 0 else BG_ROW2
                seg_html_rows += f"<tr style='background:{bg};'>"
                for col in cols:
                    val = row[col]
                    # 数値フォーマット
                    if col == '顧客数':
                        val = f'{int(val):,}'
                        align = 'right'
                    elif col in ['LTV∞（売上）', 'LTV∞（粗利）', 'CAC上限（粗利）']:
                        val = f'¥{val:,.0f}'
                        align = 'right'
                    elif col == 'k':
                        val = f'{val:.3f}'
                        align = 'right'
                    elif col == 'λ（日）':
                        val = f'{val:.1f}'
                        align = 'right'
                    elif col == 'R²':
                        val = f'{val:.3f}'
                        align = 'right'
                    elif col == '獲得効率':
                        val = f'{val:.2f}x' if val else '-'
                        align = 'right'
                    else:
                        align = 'left'
                    w = f'{seg_col_w}%' if col == 'セグメント' else f'{num_col_w}%'
                    seg_html_rows += f"<td style='text-align:{align}; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; font-variant-numeric:tabular-nums; width:{w};'>{val}</td>"
                seg_html_rows += '</tr>'

            header_html = ''
            for col in cols:
                align = 'left' if col == 'セグメント' else 'right'
                w = f'{seg_col_w}%' if col == 'セグメント' else f'{num_col_w}%'
                header_html += f"<th style='text-align:{align}; padding:9px 14px; color:{ACCENT}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT}; width:{w};'>{col}</th>"

            # 加重平均行
            avg_row_vals = {
                'セグメント': '加重平均',
                '顧客数':      f'{total_n:,}',
                'LTV∞（売上）': f'¥{weighted_ltv_rev:,.0f}',
                'LTV∞（粗利）': f'¥{weighted_ltv_gp:,.0f}',
                'CAC上限（粗利）': f'¥{weighted_cac:,.0f}',
                'k': '—', 'λ（日）': '—', 'R²': '—',
            }
            avg_row_html = "<tr style='background:#0d1f2d; border-top:1px solid #1a3a4a;'>"
            for col in cols:
                align = 'left' if col == 'セグメント' else 'right'
                w = f'{seg_col_w}%' if col == 'セグメント' else f'{num_col_w}%'
                val = avg_row_vals.get(col, '—')
                avg_row_html += f"<td style='text-align:{align}; padding:8px 14px; color:#a8dadc; font-size:0.82rem; font-variant-numeric:tabular-nums; width:{w};'>{val}</td>"
            avg_row_html += '</tr>'

            seg_tbl_html = f"""
<table style='width:100%; border-collapse:collapse; margin-top:4px; table-layout:fixed;'>
  <colgroup>{''.join([f'<col style="width:{seg_col_w}%;">' if c == 'セグメント' else f'<col style="width:{num_col_w}%;">' for c in cols])}</colgroup>
  <thead><tr style='background:{BG_HEAD};'>{header_html}</tr></thead>
  <tbody>{seg_html_rows}{avg_row_html}</tbody>
</table>"""
            st.markdown(seg_tbl_html, unsafe_allow_html=True)

            # NOTEのみ（加重平均はテーブル内に移動）
            st.markdown(f"""
<div style='background:#0a1520; border:1px solid #1a3040; border-radius:8px; padding:10px 18px; margin-top:6px; font-size:0.78rem; color:#888; line-height:1.7;'>
  <span style='font-size:0.65rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; color:#3a6a7a;'>Note</span> — 加重平均行は各セグメントを個別フィット後に顧客数で重み付け平均した値です。全体LTV∞（¥{ltv_rev:,.0f}）との差（{diff_str}）は統計的に正常な現象です。広告投資にはセグメント別、全体評価には全体LTV∞を参照してください。
</div>""", unsafe_allow_html=True)



            # ── 全セグメントの詳細分析（上位N件のみブラウザ表示）──
            seg_results_display = seg_results[:seg_display_limit]
            remaining = len(seg_results) - len(seg_results_display)
            st.markdown(f"<div style='font-size:0.78rem; color:#888; margin:12px 0 4px 0;'>詳細表示 — 上位 {seg_display_limit} 件（全 {len(seg_results)} 件）</div>", unsafe_allow_html=True)
            if remaining > 0:
                st.caption(f"エクスポートされる各ファイルには全項目出力されます。サイドバーの「ブラウザ表示件数」を増やすと追加表示できます。")
            for sr in seg_results_display:
                sv           = sr['セグメント']
                k_s          = sr['k']
                lam_s        = sr['λ_raw']         # Weibull計算用生値
                lam_s_disp   = sr['λ（日）']         # 表示用（オフセット込み）
                r2_s         = sr['R²']
                n_s          = sr['顧客数']
                arpu_s       = sr['arpu_s']
                arpu_long_s  = sr['arpu_long_s']
                arpu_0_dorm_s= sr['arpu_0_dorm_s']
                gp_s         = arpu_s * gpm
                df_sv        = df[df[seg_col] == sv]
                _dorm_s      = dormancy_days or 180 if business_type == '都度購入型' else ltv_offset_days
                if business_type == '都度購入型':
                    _ltv_short_s = _dorm_s * arpu_0_dorm_s
                    _ltv_long_s, _ = ltv_inf_offset(k_s, lam_s, arpu_long_s, 0)
                    ltv_inf_s = _ltv_short_s + _ltv_long_s
                else:
                    ltv_inf_s = ltv_inf_offset(k_s, lam_s, arpu_s, ltv_offset_days)[0]
                is_best = (sv == seg_df.iloc[0]['セグメント'])

                with st.expander(
                    f"{sv}{' ·  Priority' if is_best else ''}  ·  {n_s:,} customers  ·  LTV∞ ¥{sr['LTV∞（売上）']:,.0f}  ·  k={k_s:.3f}  ·  λ={lam_s:.1f}d  ·  R²={r2_s:.3f}",
                    expanded=is_best
                ):
                    try:
                        # オフセット処理したkm_svを使用
                        if ltv_offset_days > 0:
                            df_sv_fit = df_sv.copy()
                            df_sv_fit['duration'] = df_sv_fit['duration'] - ltv_offset_days
                            df_sv_fit.loc[df_sv_fit['duration'] <= 0, 'event'] = 0
                            df_sv_fit.loc[df_sv_fit['duration'] <= 0, 'duration'] = 1
                            df_sv_fit['duration'] = df_sv_fit['duration'].clip(lower=1)
                            km_sv_fit = _compute_km_df(df_sv_fit)
                        else:
                            km_sv_fit = _compute_km_df(df_sv)
                        km_sv = _compute_km_df(df_sv)  # 表示用（オフセットなし）

                        # ── グラフ2枚（全体と同じ）──
                        st.markdown("<div style='font-size:0.75rem; color:#7ab4c4; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;'>分析モデルの信頼性</div>", unsafe_allow_html=True)
                        col_g1, col_g2 = st.columns(2)

                        with col_g1:
                            # KM曲線：km_sv_fit（オフセット後）のt軸を+ltv_offset_daysで元スケールに戻す
                            # t=0〜ltv_offset_daysはS=1.0フラット
                            _km_sv_t = [0, ltv_offset_days] + [t + ltv_offset_days for t in km_sv_fit['t'].tolist()]
                            _km_sv_s = [1.0, 1.0] + km_sv_fit['S'].tolist()
                            fig_sv1 = go.Figure()
                            fig_sv1.add_trace(go.Scatter(
                                x=_km_sv_t, y=_km_sv_s,
                                mode='lines', name='KM Curve (Observed)',
                                line=dict(color=ACCENT, width=2, shape='hv')
                            ))
                            # Weibull：t=ltv_offset_daysスタート（日割りONはt=0）
                            t_max_s = int(km_sv_fit['t'].max() + ltv_offset_days)
                            _wei_start = int(ltv_offset_days)
                            t_range_s = list(range(_wei_start, t_max_s + 30, max(1, t_max_s // 200)))
                            fig_sv1.add_trace(go.Scatter(
                                x=t_range_s,
                                y=[float(weibull_s(max(t - ltv_offset_days, 0), k_s, lam_s)) for t in t_range_s],
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
                            st.caption(f"生存曲線：実測KM曲線（実線）にWeibullフィット（破線）。k={k_s:.3f}・λ={lam_s:.1f}日")

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
                            st.caption(f"Weibull直線化プロット：R²={r2_s:.3f}（1.0に近いほど精度高い）")

                        # ── 暫定LTVテーブル（全体と同仕様）──
                        st.markdown("<div class='section-title' style='font-size:0.75rem; margin-top:16px; border-bottom:none;'>暫定 LTV — 観測期間別</div>", unsafe_allow_html=True)

                        # グラフ
                        lam_s_actual = lam_s_disp  # 表示用λ（オフセット込み済み）
                        lam_s_int = round(lam_s_actual)
                        x_max_s = max(1825, lam_s_int + 100) if lam_s_actual > 1825 else 1825
                        t_range_s = list(range(1, x_max_s + 1, max(1, x_max_s // 300)))
                        if business_type == '都度購入型':
                            rev_line_s = [ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, t, _dorm_s) for t in t_range_s]
                            gp_line_s  = [ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, t, _dorm_s) for t in t_range_s]
                        else:
                            rev_line_s = [ltv_horizon_offset(k_s, lam_s, arpu_s, t, ltv_offset_days) for t in t_range_s]
                            gp_line_s  = [ltv_horizon_offset(k_s, lam_s, arpu_s * gpm, t, ltv_offset_days) for t in t_range_s]
                        cac_line_s = [v / cac_n for v in gp_line_s]
                        if business_type == '都度購入型':
                            ltv_inf_s_offset = ltv_inf_s
                        else:
                            ltv_inf_s_offset, _ = ltv_inf_offset(k_s, lam_s, arpu_s, ltv_offset_days)

                        fig_hor_s = go.Figure()
                        fig_hor_s.add_trace(go.Scatter(x=t_range_s, y=rev_line_s, name='LTV（売上）', mode='lines', line=dict(color='#56b4d3', width=2)))
                        fig_hor_s.add_trace(go.Scatter(x=t_range_s, y=gp_line_s,  name='LTV（粗利）', mode='lines', line=dict(color='#a8dadc', width=2, dash='dash')))
                        fig_hor_s.add_trace(go.Scatter(x=t_range_s, y=cac_line_s, name='CAC上限',    mode='lines', line=dict(color='#4a7a8a', width=1.5, dash='dot')))
                        fig_hor_s.add_hline(y=ltv_inf_s_offset, line_dash='dot', line_color='#56b4d3', line_width=1, opacity=0.4,
                            annotation_text=f'LTV∞ ¥{ltv_inf_s_offset:,.0f}', annotation_position='right', annotation_font=dict(color='#56b4d3', size=10))
                        fig_hor_s.add_shape(type='line', x0=lam_s_actual, x1=lam_s_actual, y0=0, y1=1, yref='paper',
                            line=dict(color='#a8dadc', width=1.5, dash='dash'), layer='above')
                        fig_hor_s.add_annotation(x=lam_s_actual, y=0.85 if k_s < 1.0 else 0.35, yref='paper',
                            text=f'λ＝{lam_s_int}日', showarrow=False, font=dict(color='#a8dadc', size=10),
                            xanchor='center', yanchor='middle', bgcolor='#111820', borderpad=2)

                        # プロット点（全体分析と同じ分岐）
                        plot_pts_s = sorted(set([p for p in [180, 365, 730, 1095, 1460, 1825, lam_s_int] if p <= x_max_s]))
                        for arpu_v, color in [(arpu_s, '#56b4d3'), (arpu_s * gpm, '#a8dadc'), (arpu_s * gpm / cac_n, '#4a7a8a')]:
                            px_s = [p for p in plot_pts_s]
                            if business_type == '都度購入型':
                                if arpu_v == arpu_s * gpm / cac_n:
                                    py_s = [ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, p, _dorm_s) / cac_n for p in plot_pts_s]
                                elif arpu_v == arpu_s * gpm:
                                    py_s = [ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, p, _dorm_s) for p in plot_pts_s]
                                else:
                                    py_s = [ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, p, _dorm_s) for p in plot_pts_s]
                            else:
                                if arpu_v == arpu_s * gpm / cac_n:
                                    py_s = [ltv_horizon_offset(k_s, lam_s, arpu_s * gpm, p, ltv_offset_days) / cac_n for p in plot_pts_s]
                                else:
                                    py_s = [ltv_horizon_offset(k_s, lam_s, arpu_v, p, ltv_offset_days) for p in plot_pts_s]
                            fig_hor_s.add_trace(go.Scatter(x=px_s, y=py_s, mode='markers',
                                marker=dict(color=color, size=5, line=dict(color='#111820', width=1)),
                                showlegend=False, hovertemplate='%{x}日: ¥%{y:,.0f}<extra></extra>'))

                        tick_vals_s = [180, 365, 730, 1095, 1460, 1825]
                        tick_text_s = ['180日', '1年', '2年', '3年', '4年', '5年']
                        fig_hor_s.update_layout(
                            paper_bgcolor='#111820', plot_bgcolor='#111820',
                            height=280, margin=dict(t=30, b=50, l=70, r=120),
                            font=dict(color='#ccc', size=10),
                            legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
                            xaxis=dict(title='継続期間', gridcolor='#1a3040', tickvals=tick_vals_s, ticktext=tick_text_s, tickfont=dict(color='#888')),
                            yaxis=dict(title='金額（円）', gridcolor='#1a3040', tickfont=dict(color='#888'), tickformat='¥,.0f'),
                        )
                        st.plotly_chart(fig_hor_s, use_container_width=True)

                        # テーブル
                        ACCENT_S = '#56b4d3'; BG_HEAD_S = '#0d1f2d'; BG_ROW1_S = '#0d1520'; BG_ROW2_S = '#0a1018'
                        SEP_S = '#1a3a4a'
                        all_horizons_s = [180, 365, 730, 1095, 1825]
                        if business_type == '都度購入型':
                            lam_s_rev  = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, lam_s_actual, _dorm_s)
                            lam_s_gp   = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, lam_s_actual, _dorm_s)
                            rev_99_s_d = brentq(lambda h: ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, h, _dorm_s) / ltv_inf_s_offset - 0.99, 1, 365000)
                            rev_99_s   = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, rev_99_s_d, _dorm_s)
                            gp_99_s    = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, rev_99_s_d, _dorm_s)
                        else:
                            lam_s_rev  = ltv_horizon_offset(k_s, lam_s, arpu_s, lam_s_actual, ltv_offset_days)
                            lam_s_gp   = ltv_horizon_offset(k_s, lam_s, arpu_s * gpm, lam_s_actual, ltv_offset_days)
                            rev_99_s_d = brentq(lambda h: ltv_horizon_offset(k_s, lam_s, arpu_s, h, ltv_offset_days) / ltv_inf_s_offset - 0.99, 1, 365000)
                            rev_99_s   = ltv_horizon_offset(k_s, lam_s, arpu_s, rev_99_s_d, ltv_offset_days)
                            gp_99_s    = ltv_horizon_offset(k_s, lam_s, arpu_s * gpm, rev_99_s_d, ltv_offset_days)

                        hor_html_rows = ''
                        for idx_h, h in enumerate(all_horizons_s):
                            if business_type == '都度購入型':
                                lh_rev_s = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s, arpu_long_s, h, _dorm_s)
                                lh_gp_s  = ltv_horizon_spot(k_s, lam_s, arpu_0_dorm_s*gpm, arpu_long_s*gpm, h, _dorm_s)
                            else:
                                lh_rev_s = ltv_horizon_offset(k_s, lam_s, arpu_s, h, ltv_offset_days)
                                lh_gp_s  = ltv_horizon_offset(k_s, lam_s, arpu_s * gpm, h, ltv_offset_days)
                            label = f'{h}日' if h < 365 else f'{h//365}年（{h}日）'
                            bg = BG_ROW1_S if idx_h % 2 == 0 else BG_ROW2_S
                            pct = lh_rev_s / ltv_inf_s_offset * 100
                            hor_html_rows += f"<tr style='background:{bg};'><td style='text-align:left; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; width:28%;'>{label}</td><td style='text-align:right; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lh_rev_s:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lh_gp_s:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lh_gp_s/cac_n:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#c8d0d8; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>{pct:.1f}%</td></tr>"

                        # λ行
                        lam_pct_s = lam_s_rev / ltv_inf_s_offset * 100
                        hor_html_rows += f"<tr style='background:{BG_HEAD_S}; border-top:1px solid {SEP_S};'><td style='text-align:left; padding:8px 14px; color:#a8dadc; font-size:0.85rem; width:28%;'>λ {lam_s_int}日</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lam_s_rev:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lam_s_gp:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{lam_s_gp/cac_n:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>{lam_pct_s:.1f}%</td></tr>"

                        # 99%行
                        hor_html_rows += f"<tr style='background:{BG_HEAD_S}; border-top:1px solid {SEP_S};'><td style='text-align:left; padding:8px 14px; color:#a8dadc; font-size:0.85rem; width:28%;'>LTV∞到達率: 99%（{int(rev_99_s_d):,}日）</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{rev_99_s:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{gp_99_s:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>¥{gp_99_s/cac_n:,.0f}</td><td style='text-align:right; padding:8px 14px; color:#a8dadc; font-size:0.85rem; font-variant-numeric:tabular-nums; width:18%;'>99.0%</td></tr>"

                        hor_tbl_html = f"""
<table style='width:100%; border-collapse:collapse; margin-top:4px; table-layout:fixed;'>
  <colgroup><col style='width:28%;'><col style='width:18%;'><col style='width:18%;'><col style='width:18%;'><col style='width:18%;'></colgroup>
  <thead><tr style='background:{BG_HEAD_S};'>
    <th style='text-align:left; padding:9px 14px; color:{ACCENT_S}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT_S};'>ホライズン</th>
    <th style='text-align:right; padding:9px 14px; color:{ACCENT_S}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT_S};'>LTV（売上）</th>
    <th style='text-align:right; padding:9px 14px; color:{ACCENT_S}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT_S};'>LTV（粗利）</th>
    <th style='text-align:right; padding:9px 14px; color:{ACCENT_S}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT_S};'>CAC上限</th>
    <th style='text-align:right; padding:9px 14px; color:{ACCENT_S}; font-size:0.8rem; font-weight:600; border-bottom:2px solid {ACCENT_S};'>LTV∞到達率</th>
  </tr></thead>
  <tbody>{hor_html_rows}</tbody>
</table>"""
                        st.markdown(hor_tbl_html, unsafe_allow_html=True)


                    except Exception as e_sv:
                        st.caption(f"グラフ生成エラー: {e_sv}")

else:
    st.markdown("<div class='section-title'>セグメント別 LTV 分析</div>", unsafe_allow_html=True)
    st.info("サイドバーの「セグメント分析」にCSVの列名を入力してください。例：`plan, channel`")
    st.markdown("""
**使い方：**
1. CSVにセグメント列を追加（例：`plan`列に「月額」「年額」など）
2. サイドバーに列名を入力
3. セグメント別LTV∞・優先獲得推奨が自動で出力されます
    """)

# ══════════════════════════════════════════════════════════════
# Data preview
# ══════════════════════════════════════════════════════════════

with st.expander("読み込んだデータを確認"):
    st.write(f"有効データ: {len(df):,}件 ／ 解約: {df['event'].sum():,}件 ／ 継続中: {(df['event']==0).sum():,}件 ／ 平均日次ARPU: ¥{arpu_daily:,.2f}")
    st.dataframe(
        df[['customer_id','start_date','end_date','duration','event','arpu_daily']].head(30),
        hide_index=True
    )

st.markdown("---")
st.markdown("<p style='color:#333; font-size:0.82rem; text-align:center;'>LTV Analyzer — KM × Weibull Model — Built for marketing analytics professionals</p>", unsafe_allow_html=True)
