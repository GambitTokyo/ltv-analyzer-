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
  <div style='font-size: 0.78rem; color: #3a5a6a; margin-top: 8px; letter-spacing: 0.02em;'>Kaplan–Meier × Weibull — Segment-level LTV Intelligence &nbsp;·&nbsp; v200</div>
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
            "LtVwGqklJq3YpJ6nXN/ueBSPoCv4PMJoaCB37BlBL4oNv4eQGwdxYgXq82C2Jwf81hzmGYCs"
            "+/X+S79O02+DOpWQ3QRkVIAHAmS1m8Bf2Y9P96C1RqAkGsaoccU6RUUK6Sqoo2uU5zDUgSjl"
            "D9ojHNv/P0eo+IIc/3ZT67/EzX8AAAD//wMAUEsDBBQABgAIAAAAIQBzu9SrnQUAAEoVAAAi"
            "AAAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDE0LnhtbNRY3Y6cNhS+r9R3QPSyIvzY"
            "gBllNhoYiCptklVnc115wLNDC5gaz2S2VaS8Vvs4eZL6B5j9mU3ZJmmSGziAz+dzjs85/szT"
            "Z4e6MvaEdSVt5qb7xDEN0uS0KJurufn6MrOQaXQcNwWuaEPm5jXpzGdn33/3tJ11VXGOr+mO"
            "GwKj6WZ4bm45b2e23eVbUuPuCW1JI75tKKsxF4/syi4YfiOw68r2HCewa1w2Zq/PpujTzabM"
            "yZLmu5o0XIMwUmEu7O+2ZdsNaO0UtJaRTsAo7dsm7TrClsJWGRTzTHibr6rCaHAtYuC6vyS7"
            "jtPa0P6r7117yQiRUrN/ztpVe8GU2sv9BTPKQsL06qbdf+iHqcdmrwT7jvrVIOLZYcNqeRcR"
            "MA5zUyzUtbza8h05cCPXL/Pj23z76sTYfJueGG0PE9g3JpVeaePuuwMHd1ZVWRDj5a5eE2Zc"
            "VDgnW1oVQgajo4MLXXtO8986o6HCRRkR7fE4QodB3tutwa9bgS6STECLHPxjbv6+w4wTZor5"
            "D3IVenWto4Sj2X1Y+SGmxbWcey3u6iWeVR1f8euKqIdWXjZibaVTf/pRClJvkViLJAaW70No"
            "xakXWcBbAmfpAR94yVtztE143gjrJAQTcamwrBrSWK9XwuKaJxXBzRh4bROe8bP37/764f27"
            "v2XQuQq9mF9hnAQqSsaPy8fPjKOe9Fu5YB9dtYdVe3jtgmHtEtpwkf+3ls3/uGU7sVAqER5e"
            "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"
            "My5wbmeJUE5HDQoaCgAAAA1JSERSAAAECgAAAkcIBgAAABmytKQAAAA6dEVYdFNvZnR3YXJl"
            "AE1hdHBsb3RsaWIgdmVyc2lvbjMuMTAuOCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/BW3XO"
            "AAAACXBIWXMAABcSAAAXEgFnn9JSAADIKUlEQVR4nOzdd3iTVfsH8G92mqaL0UVLF7RA2XtP"
            "QUQEBQTFjSiun/q65+vWV1BBRRTFhSyRIQiyZ4GyKZuWVbo3nWma8eT3R2kgNKVtkjZt+v1c"
            "F5fmPOc5525PWnjunCHy8Q81gYiIiIiIiIgIgNjZARARERERERFRw8FEARERERERERGZMVFA"
            "RERERERERGZMFBARERERERGRGRMFRERERERERGTGRAERERERERERmTFRQERERERERERmTBQQ"
            "ERERERERkRkTBURERERERERkxkQBEREREREREZkxUUBEREREREREZkwUEBEREREREZEZEwVE"
            "REREREREZMZEARERERERERGZMVFARERERERERGZMFBARkcv44rMPsGXdCnTuFO3UOLasW4Et"
            "61Y4NYam4qP/vol/Vy+Fn29LZ4dCtfDHz/OwZd2KSuP26ovPYsu6FXho6uR6i2XUiKHYsm4F"
            "Xn3xWYtyP9+W2LJuBf74eZ5FuUIhx/I/FuCXH76GRCKptziJiOqT1NkBEBGRc/l4e+Puu+5A"
            "rx7dEBjoD7lMhsKiYuTn5yPh/EWcOHUGe2IPQqvVOjvUJsPNTYk7Rt2G3j26ISQkCJ6enjDo"
            "DcjJzcW5+PPYsXsvDh+Nc3aYTtetSyf07d0D6zZsRmZWtsW1h6ZOxsNTJ+P4ydN45c33rN7f"
            "v28vvP36S5DLZPh301bMmTsfJpPJfC8AlJWVYfJDT0Cj0VQZx+zPP0LH6PYAcMv+nOWB+ybh"
            "0QfvQ3zCBTz30htW60y65y7MePwRAMAHn87Cnn0HrNZbvuhn+Hh7Ydbsudi8bWddhdyglZXp"
            "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"
            "ERERERERqZgoICIiIiIiIiIVEwVEREREREREpGKigIiIiIiIiIhUTBQQERERERERkYqJAiIi"
            "IiIiIiJSMVFARERERERERComCoiIiIiIiIhIxUQBEREREREREamYKCAiIiIiIiIiFRMFRERE"
            "RERERKRiooCIiIiIiIiIVEwUEBEREREREZGKiQIiIiIiIiIiUjFRQEREREREREQqJgqIiIiI"
            "iIiISMVEARERERERERGpmCggIiIiIiIiIhUTBURERERERESkYqKAiIiIiIiIiFRMFBARERER"
            "ERGRSrvYFzAXolaLwh17kFW9GgZbBqKhILzNjWg7cgDh4eEZ9aUxGFG8ey8yK1dCZ7YgEvDD"
            "U1+LtqOHII2MzNMdEBERERERES0ty3ZEgaDRYM1jT6Fo163Q6PTwNFzDyJAPues2YsNnvgRD"
            "hmPafWmNJqx/6vNwbdkORZbhabgGKRyGa8sOrH/y89AajfN3I0RERERERERLyLIdUVC081bY"
            "Coow1NmOmpeehxyJAABcW3ag9I57UPGRfah58dlp9VV6570wOTPRX3cV1177NaAo6nbX5u0o"
            "uf0eNOx/bd7uhYiIiIiIiGipWJYjCgRRRP6mrQCAxnf3q0kCAOg6cwJ+dw8yiktgyc2fsi+d"
            "xYLs6jWQo1E0vbtfTRIAQMvB9xAJ+JGzeh20JnPqb4SIiIiIiIhoiVmWiQJbYTG0RiNCXg8C"
            "7p6E/f3XrgIAnBVVU/blKK2AIIrwdbQhEvDH7VMkCQONdRBEEc6yitRcPBEREREREdEStiwT"
            "BebsXADAcE9ikgAA/L3dce0mY8nJjTvmRmPnMOdM3RcRERERERHRcrcsEwUGux0AEB72Jd0f"
            "HhqKazcZvS1jtK+h5H2NnsNgz5jxdRIREREREREtN8uymKFGpwcAyJFo0v1yNFazQKPXT6Mv"
            "3WhfkaT7x7aPnXMqGz/3laTb6959D3I0AkNWwbT6WVSCEFerYbYUUUQQIhSTDph7dzRbo/Fk"
            "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"
            "IyIiSissZkhERETzbs8Tn4POaEJPYx16Gq6p28cKG47VGRivYvueSfu8PFbU8M57sXrv3ZAl"
            "CVcPvT/pMWMioSCO/epZKLKMzMIV070NIiKimwITBURERDRvbDm5uPcbf4LVt9+DSCiID37y"
            "dNx+b3cnAGDNHffGbXetXI1NH31o0r7dTQ1wNzegbMsO2HPz0HL+DALegYR2K/fchszC4oTt"
            "KzZshiCK8Hv6ZnpbREREaY1TD4iIiCgl7vzytwDElinUm0zIyHPB6SqEIIrwdnfinf/4d3ja"
            "W+OOufj2G1h16x1Yd/f9KFi1Fv1tLbA4M+GqWoXz+1+dMllw+f23cccXKgAANQfeTtqmfNsu"
            "3P3VP8RgTxf621shhcOw5eQir7wKsizh+Mu/SMHdExERpQ8mCoiIiCglVt16JwBAikYRCQXh"
            "H/Cg9sgBNJ85ieazp6AocsIxgz1d+NXf/DfsfvyzyC2vQunmbfB2deLAT3+IKwfemTJR0FFz"
            "EQAw3N+HtgvnkrY5/9arGPb0I79qFVwrV0NnMMDvHUD9iSM4/9arcDc3zO3GiYiI0ozgzC9N"
            "XMyYiIiIaBnYvO8R7Prkp3HylRdw6pUXFvtyiIiI0gJrFBAREdGypDOasP6ej0KKRFDzQfJp"
            "B0RERDRznHpAREREy0r1rXeiYNUaFKxcA4szE+f3v5a0iCERERHNDkcUEBER0bJSsGoNVt16"
            "J3RGIy6+8yaOvfjzxb4kIiKitMIaBURERERERESk4ogCIiIiIiIiIlIxUUBEREREREREKiYK"
            "iIiIiIiIiEjFRAERERERERERqZgoICIiIiIiIiIVEwVEREREREREpGKigIiIiIiIiIhUTBQQ"
            "ERERERERkYqJAiIiIiIiIiJSMVFARERERERERComCoiIiIiIiIhIxUQBEREREREREamYKCAi"
            "IiIiIiIiFRMFRERERERERKT6/wH6FsLsq+mWXgAAAABJRU5ErkJgglBLAwQKAAAAAAAAACEA"
            "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"
            "5Ocxf97cAUHBvLmzsSyLz7ftoKHxMC0trcyfO3jkwfy+oOD4l/AvPXBvolp/Z1cXBw8eIi8v"
            "l4uWLuaipYv5v8ef5rEnnz3jPn/tN36VO279As0tLdTU1FJaWsyyi5aydNFCvv29H/Lhx5vO"
            "+Jzns1966Is89MW7aW5poba+npLiIhZfsIB5lXP44z//VtKw4GyezSUXLuGPfu9rRCIR2tra"
            "qampJTU1hcmlJcy4fxrLL7+U3//GX9DV3Z20nx6Ph3/6/neomDmDmkN1HDxUS/QUxfRq6xrY"
            "tn1n0n1paWlMmzp50PYH7r2TG2+4lkAwSGtrG/urD5CZkcH8yjnMr5zDkkUL+dvv/1OifU9P"
            "L9u276Rs2hRSU1Opraunvb0jsb/pSPNJ+wjgcrn4qz//I5ZdtBSI1zJo6DrMlMml3Hzj9Vy1"
            "4nL++jvfP2ENj9SUFP71h3/HlMml1NTW0dh4mJLiIm69eSVzZlfwu3/056f8rM6EaZzZINXU"
            "1BS+9+1vMmdWORCvJxEKhZg2dTL33nUb11y1nD/95nc4cLAGGJ7PVERETp+CAhERGXZbt+3g"
            "2quWs6ByDm+sfiexvX+kQP+L2ufbd3Ll8suYXFrModp6ANxuN7PKZyTO0++6q1fw8AP30trW"
            "xr/8+D/Z8OHGxL7LL72YP/r93+KXHryPHTt3s/nTrafd17zcHG75wvX8wz/9G6vfWQvEX0B/"
            "+6u/xo3XX8M3vv7b/NpXf++0viEdD/Jyc7jrtpv4m+/+gPc3fATEP48/+YPfYfnly/j1X3mY"
            "3//jvxxwzNk+m/0Havjmt7/H5k8/HzByJD09jV/90gPcfOP1/OovPcC//Pg/k/b1issuofFw"
            "E7/+W1/nwMFDib6ezJPPvsCTz74waLvb5eIH3/0WAB98tHHAi+d7Gz5i1ep32bm7asDIgZLi"
            "Ir7x+7/FiisuZf0HH/HuuvUA7Ntfzdf/5JuJkSdPPvMCb7695qT9Ot5DX7ybZRctpbu7h29/"
            "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"
            "QERERERERESOoaBARERERERERBIUFIiIiIiIiIhIgoICEREREREREUlQUCAiIiIiIiIiCQoK"
            "RERERERERCRBQYGIiIiIiIiIJCgoEBEREREREZEEBQUiIiIiIiIikqCgQEREREREREQSFBSI"
            "iIiIiIiISIKCAhERERERERFJUFAgIiIiIiIiIgkKCkREREREREQkQUGBiIiIiIiIiCQoKBAR"
            "ERERERGRBAUFIiIiIiIiIpKgoEBEREREREREEhQUiIiIiIiIiEiCa7Q7cL7y5ZeCYeLEoqPd"
            "FREREREREZEEw3KBYxM8UntWx2tEwdkyTDBGuxOnyThfOipDpmc9cehZTwx6zhOHnvXEoWc9"
            "Meg5Txxj9VkbxN9Zz5JGFJyl/pEEoZb6Ue7JqZkeH3Y4ONrdkHNAz3ri0LOeGPScJw4964lD"
            "z3pi0HOeOMbqs/bmFg/peI0oEBEREREREZEEBQUiIiIiIiIikqCgQEREREREREQSFBSIiIiI"
            "iIiISIKCAhERERERERFJUFAgIiIiIiIiIgkKCkREREREREQkQUGBiIiIiIiIiCQoKBARERER"
            "ERGRBAUFIiIiIiIiIpKgoEBEREREREREEhQUiIiIiIiIiEiCggIRERERERERSVBQICIiIiIi"
            "IiIJCgpEREREREREJME12h0QOZmyKaUsnDebnOxMotEYdY2H+XjTVrq6e87oPKmpKSyeP5fS"
            "kkJS/D5C4TAtre18um0XDY1NiXa/8Uv3nfQ8dQ2HefXNNYnfz6mYwdTJxeRkZeLzeYlEo3R2"
            "9bCrah9V+w7gOE7S80wpLWLenArycrJxuSx6egM0NjWz4aPNRKLRM7q3sS41xc89t92Ix+Pm"
            "08938vHmrad9rMtlsWRhJdOnTcHv99HT00vVvgN8+vnOAZ9tWmoKD9x9y0nPtWvPftZt2AhA"
            "0aR8bll59Unbb9zyOVu27hiwLSc7iwsXzaewIA/TNGhubWPzZzuoazh82vckIiIiIjLWKSiQ"
            "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"
            "oAn42fJ733H/tsJbefrV177pczy+RbK+87//86nPl3XylP5yD5v53Y8e3bvqu9/+hqR/bmbY"
            "q2d3HTl6TH/7yNMGxo99VL//7Sv6l399XqvfWOdzngd69dR3v/PNe37fK9k53qIgJiZaklRa"
            "VuZ3bmlpmWJiYmQymT7x6Wvbd+zStKmT9bWvfFE7d++T0+mUJPXs0V2TMsZJqt008eNSU1vW"
            "yX4rL09f+uK3tHvv/jrzYqJjvHn8ZZSk2Bjf8xsh5IoCs82mjmMmqLIgX1cP7qv3+fo8/QW/"
            "xy/t2RtSJYTZYlb/0aM0Yf4c7XxjnXaufdNnzt4NmbJHRmr32vWquMMnIQAAAIB/SmqVGrBz"
            "mS2W+zpf/o2bAcvwcT26d1OP7nU3jT9+4pSemD5XZeXl3mP/+PPvVFnp0Fe/9Hl99UuflyS9"
            "tmiZ/rFoqSRp6YrVWrpidYPlvBcrVq/RgrmzNPKRYdq5dYO2vbNTcbGxemLKYzp/IUu9enaX"
            "2+32ed37Rz9QYqt02Ww2pXdop+e+9AWtWPw3Pf/vP9LfX1tiwEdSfyFXFLQbPkr22DidWLlY"
            "Hj8bSwSUySRzWHjDvkcAmKxhd9y10mK1avC4RzV+9gy1aNNakjR+3mzt3bxN7o/9/rkkbV25"
            "RpKaxMfdHH3SWiO0sNbNA+vcfLDWzQdrHVpMJpPMNrs8cstkC5M+/Gm1yWKr74lrz/fxw9b7"
            "PO8dzlcft8/3+roN+sLX/lWS1Dq1pb7x1S/q2Wfm69Xf/ErPfPnr3vmtuzz4ieepj7JKhyQp"
            "Ni7W7/liYmNqf2Jvtcn0CeepkTR9/uf1b998TlMey9Dnn1mgGzdz9etX/0/nsy7pr7/7H+UX"
            "Ftd5j4+udY2kc5ey9c0XfqjWbVrrpR//UJu37dT1m7W3ad8uTmITElVUXFznvWPj4yVJJeUV"
            "n/h7YjKZZLLZZJJZ5rDwO18hYTJ5/zzejyZXFHSa8LjPsYLzZ1V44ayiU1sptU9/3Tr5gUqy"
            "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"
            "BBCIE0iTu4By1m0gn9SBcRV/bbNUrqOgXPnjjjw9lave4wrOWyju2LbN1+AZpcqY1fbq3r4L"
            "N39ZeW3XmqsylK6kYv/rJk8fX2WvVV/VBo1TwcbPKVX6uI8mSG2SKbUfkaePn2l+i+KhjGWr"
            "KDhPwZvO5XnsGeseYSeOqGjLDspStZ51vmDTF1V1wCjleeTp+HhTZ8musl0GyNPbR9ls7Yq0"
            "aGe1daWN0yUsEhGfrhBAAAEEEEAAAQQQQAABBBBAwPUFiBABBBBAwAUFrkdG6OjCX7Vv1hRF"
            "XL6ki/t2Wq9PLJ6j8DOnFFK8nBW1f6as8gkM0vlt662nM6oMGqsqb45Wlip1rPNmb45NvXl6"
            "w6o0m9hYHZ77o9IXKWW7Po2yVKyp0+uWKCr0so7Z7rF+5EBFXQ01LeNL9NWrOvznT3HjWD7P"
            "antozvfW8aVDexVpG+ehOT/Etzcv0hcuoaOLZ+v02mU6YosnW9W4ce2Z/rl2fvOxrkf+/Q/t"
            "M5auovBTJ7T/pyna/8t3VpLGL22I6cZliqfLREIgCCCAAAIIIJACAtwSAQQQQAABBBBAAAEE"
            "EEAAAccSOL97q9LmLyzzlknpChZX9NUwefqlsiUpamv9h/21fsRAZatWTxX7DLMSEeZ44+hB"
            "ympLFqS54QmI8zs369qFs8rzaBP5BATpxIqFut2Xf6Ys8ksTrLLdBqrca+8oOJ/t/vbGHh7K"
            "Uau+Tq5apJiIm98VyMPLS14+vlbL8LOn5O0fKL/bJCGCcuZW2OljVtsI27hsORX5pklrHSf9"
            "Jnnu4Jk8t3Gtu2Sv9Yjs5cbI7HVmT32cEQ44mP8/2AvrwXXXg29Q8I3TG/9npJn7G0+YY3tJ"
            "/HqPhN+35sM3Duce+0mp++oex0l7+yTb16DZ2+vM3hzbizm2F3ud2dvrzN4c24s5thd7ndnb"
            "68zeHNuLObYXe53Z2+vM3hzbizm2F3ud2dvrzN4c24s5thd7ndnb68zeHNuLObYXe53Z2+vM"
            "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"
            "WFvbsH//ft4ZObLAhmHDhg2kpKTQv39/3U0lIndFjFq1ajFnzuf5XkOGDgXgScZjnj7Vngvu"
            "0MFDLF60CFd3N1avWcOSb7/FwNCQeXPnav23lHS9ihUDaNyoMdu2bXvuI0xC/JPt37ePRYu+"
            "xse3Ahs2bNDE/Px58zTJlCdPnvK4kHkYC3Pw4EFOnTpJp06d2LBhA9u2b6dTx07s2fMHs2fN"
            "0q0u/kZz5szhyqXLvDl4MNu2b2fb9u106dKF06dOMW/ePE29hPh4Zs+aTXRUFNM//pht27dT"
            "q3ZtfvnlF62V/R4/Tked+ed5y8HBAXMLS15//fV87fvs2Z/SvIC5+/wDKmJoYKi1OFBe635Z"
            "xy+//EKt2rXZtn070z/+mOioKGbPml3gXT3x11Cr1cycNZNrV68xYeJ7/LFnD127diEoKIhv"
            "lyzR1Ctqe6AbO89z9epV5s+dh5GRMbNnz2bf/v0sX7EcT09PfvrpJ62FNtavX09KcjLvTpjA"
            "H3v2sHzFctxcXfl60SKteuLlsLO1RWVry9hxY/O1CzNmfEJApaI9ElxQ3zBXxYAAnJwcuVXM"
            "Udcl9RnFX6+o5yddRW3DNm3cyFdffkmZsq4s/mYJ+/bvZ8qUKSQmJjJzxgzOn8teTK6o9cS/"
            "h1qt5osvvuDc+XOMeGekVows+rrgBUue14fJq6jXweKf4UX6n7lJuI8++ogv5s0jPT2dhQvm"
            "a00vprJV4enpxWdz5uSLhQnvvQc5yb0Zn3zC+Yvnswdh7djBhg0bqFW7Nr///luJX2cZWKhc"
            "snQLXyUqW1tmzpxJXHwcc7+Y+9wv8kX4+HgzecoH7Nix/Zknp7/aF3O/4N69B8yfN1d300uj"
            "srVlxoxPCA6+yZcLF+puFi+Bg70DMbEFr2Ar/hoTJkzA2dmF994r2ef0XxaJmZfHz8+PSZMn"
            "sXz5in/9SF6Jm5frVYgdiZmX75/QN3wREjNCXxIzojgkbv5eU6dOpV79+ty9e5c9f+xh06aN"
            "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"
            "Wpo7+o4uzgBcO3+JMf2GaEapFTRSLfc1Y3zBj008z61rN9iw8ife6T2QLz+Zw/3wuwWubOXk"
            "4oSZUpnvsQkhhMhlrbKhfvPGvP/pNIZOeId74XdZvuAbzfbHGY/JLGD+GQDPCt48Tk/nXlj+"
            "x+OtVTa4ly/HxFkfMXH2VK3XkPEj89V91mcQ/wwh14JZsfAbjIyNmDDjQ1YHbWb2d1/i5lmO"
            "rT9v0Ewc/vRxBk+fPNHsp7K3R2lhTpcBvfLFwoSZ/6Nus0YA2NiqsFbZ8Maot/LVGzdtMj4V"
            "/Yr8GcTfx9vfFzsnB80qi/ooqL1Z+eUSju47SOf+Pflh50YGDH+Taxcv69VGFCW2xMtT1DZh"
            "6dyvCL58nZ4D+7N0y88s3fIzLTu149Lp85rvX7e9uRsWzsLpnxIT/ZChE97hh50bmf3dl5Rx"
            "d2XjDz+zfsUaTd1cXv4VMDAw4E6IfklgIYQQ4lVWIgsVjPzgXYyNjdm5YSsjJr+LnaM998Pv"
            "8sUHH9OySwfNQgWlypbh7UljOX/8NDUb1cNeZ2i5QqHAyNiYjMfpZOl8qvMnT7N41jztwhdc"
            "YGDE5PHUy+mQJCY8Yuncrzl3/JRuNSH+lWSizZKTO5m4azl37oWHs2/bbg7t3quVnPes4MNb"
            "E0ez4cefClyI4EUV5TO8KImZl2/SZ9OJehDJCj0SH/80Ejcv31/d3vzVJGb+Hv/m9kZiRuhL"
            "YkYUh8SN0Je9nV3JJNWEEP88clJ4uYaMH4m1jQ0LP/60RBNdL5PEzMtVr2kj2vfuytK5X+n9"
            "+N8/icTNy/dvb28kZl6+f3t7IzEj9CUxI4pD4kboS5JqQrzC5KQg9CUxI4pD4kboS2JG6Eti"
            "RuhLYkYUh8SN0Je9nV3JzKkmhBBCCCGEEEIIIcR/iSTVhBBCCCGEEEIIIYTQkyTVhBBCCCGE"
            "EEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTVhBBCCCGEEEIIIYTQkyTV"
            "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"
            "KWV/+uPu/YfXq5//vM/Pf/zll/dv3pWPb/72x7sPX5cv8vnd73df5fhffnv/6cv9q9292Xq5"
            "P96/+fzxy8dfvg5vPv6xeqX7A8kroR6P85P85h/yfu9f6NOeL/Tp4z/eff708f14qulr/VvH"
            "9fbjm/zbr3/+8nX8lcArd1/vXn24+0Ow/vbq6+tXf/vy7vOf3v78+s6qX+6U9nznAv/vX9zr"
            "V58+f/z7+7fL3/3z+7+/+5Mc783v/Vf/txRbqnUBtC8IRpODqFMAn0okW0oshv7f6582TvBK"
            "/uqfX//tw9uPrz6/ky9v/vbl61/f/f7q7u3bv/7+9tXbd7/3//zxcfzVl9/f/sfn8Vd/kTf0"
            "l7svX999Xr6dl30Dr95+/fk1KbKgNJD+T6KF5oXWA3m5sn//+bX3Orx+dffma39RYl6/5b/8"
            "1z3z8Rfy/3+a/oa8ywcwHcOnkcKnEcKnDqZ/N4e3jbxQZmHcEJRdvm1n1PptK3Zu9cblLW2/"
            "8fv//+pN/+Pyg/KrJS/bf6xz+ebnvnx6wLIEcPdPB0AL8gtWA7MbAVhtzQaA8e3f9ZMf/O5X"
            "P/XqvfwvlmN+fjeKlfGwavUPPPLl/p/xKftp8+8eeXaEU5hyu2YBUy+MX2B4uE2Og32AaZw/"
            "DUx7LTDl0dRhUBiWN1Ne4vQw3QQmRTlb1h40IUFlxVA9V8gqs7bVOGvyATDXsu6fD3SUdZoH"
            "pZZAvfXr2/kg656P1E+QuppaSlyASkwQGyXIPjcgG2zIFjkmvFCk4wNPSh54un/gTwQxTCDK"
            "OZRpGKEl06DImaBob4GZDYfksVh3yffySR38fKSopkwVNcc5QIkuQWulAAbrhGlylBJpp3c/"
            "6ysF/8B3w+xZStF/uoGjeMG4MDRoXMK1ivSGFN3fujEB5aTGdwdmh5WzA8mHd//oduCMsISB"
            "0d9j2VQu+2PB4E0go+RnjuXyXQu5/5t///zX7ie963/2z3f//fFvX+fiL6jxoQ1m9dA6mmrn"
            "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"
            "3tIzzulUFzJn/cXSnWwRX2Zq8fz+EwAA//8DAFBLAwQUAAYACAAAACEAtM9YGbkAAAAkAQAA"
            "LAAAAHBwdC9ub3Rlc01hc3RlcnMvX3JlbHMvbm90ZXNNYXN0ZXIxLnhtbC5yZWxzjM/BCsIw"
            "DAbgu+A7lNxttx1EZO0uIuwq8wFKl3XFrS1tFff2FnZx4MFLIAn/F1I373kiLwzROMuhpAUQ"
            "tMr1xmoO9+56OAGJSdpeTs4ihwUjNGK/q284yZRDcTQ+kqzYyGFMyZ8Zi2rEWUbqPNq8GVyY"
            "Zcpt0MxL9ZAaWVUURxa+DRAbk7Q9h9D2JZBu8fiP7YbBKLw49ZzRph8nWMpZzKAMGhMHStfJ"
            "WiuaPWCiZpvfxAcAAP//AwBQSwMEFAAGAAgAAAAhAJOqfZi5AAAAJAEAADAAAABwcHQvaGFu"
            "ZG91dE1hc3RlcnMvX3JlbHMvaGFuZG91dE1hc3RlcjEueG1sLnJlbHOMz8EKwjAMBuC74DuU"
            "3G03BRFZt4sIu8p8gNJmXXFrS1vFvb2FXRx48BJIwv+FVM17GskLQzTOcihpAQStdMpYzeHe"
            "XXcnIDEJq8ToLHKYMUJTbzfVDUeRcigOxkeSFRs5DCn5M2NRDjiJSJ1Hmze9C5NIuQ2aeSEf"
            "QiPbF8WRhW8D6pVJWsUhtKoE0s0e/7Fd3xuJFyefE9r04wRLOYsZFEFj4kDpMlnqgWYPWF2x"
            "1W/1BwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAALAAAAHBwdC9zbGlkZUxheW91"
            "dHMvX3JlbHMvc2xpZGVMYXlvdXQxLnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF"
            "9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScOb"
            "GPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJ"
            "akgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//"
            "AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAABwcHQvc2xpZGVMYXlvdXRzL19yZWxz"
            "L3NsaWRlTGF5b3V0Mi54bWwucmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk"
            "5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjE"
            "XI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j"
            "/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMAUEsDBBQA"
            "BgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxh"
            "eW91dDMueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi6Nub0YKD"
            "4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyOeHCRRVE8"
            "axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7B"
            "PCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEA"
            "1dGS8bwAAAA3AQAALAAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ0Lnht"
            "bC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39N"
            "o3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgM"
            "NCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLR"
            "WTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAA"
            "NwEAACwAAABwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRlTGF5b3V0NS54bWwucmVsc4zP"
            "vQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZa"
            "ViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX"
            "0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXU"
            "U9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAA"
            "cHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDYueG1sLnJlbHOMz70KwjAQB/Bd"
            "8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne81"
            "3K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyOeHCRRVE8axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHN"
            "HXtS66raqPRtQDszxclqSCdbg7i+I/1jh65zhg7BPCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZU"
            "yxIBqm3U7N32AwAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAALAAAAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ3LnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOI"
            "NHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0"
            "FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0"
            "bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd"
            "9gMAAP//AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAABwcHQvc2xpZGVMYXlvdXRz"
            "L19yZWxzL3NsaWRlTGF5b3V0OC54bWwucmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQB"
            "juTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj2"
            "7XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpI"
            "J1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMA"
            "UEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAsAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9z"
            "bGlkZUxheW91dDkueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0AY7k2gbbJOSi"
            "6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY9u1y0VxoxFyO"
            "eHCRRVE8axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclqSCdbg7i+I/1j"
            "h65zhg7BPCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8DAFBLAwQUAAYA"
            "CAAAACEA1dGS8bwAAAA3AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlv"
            "dXQxMC54bWwucmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPj"
            "ff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxr"
            "GHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8"
            "JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMAUEsDBBQABgAIAAAAIQDK"
            "Dhnb1QAAAL4BAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDExLnht"
            "bC5yZWxzrJA9SwQxEIZ7wf8Qpjeze4WIXPYaEa6wkfMHDMlsNrj5IBPF+/dGbG7hCgvL+Xre"
            "h9kfvuKqPrlKyMnAqAdQnGx2IXkDb6fnuwdQ0ig5WnNiA2cWOEy3N/tXXqn1I1lCEdUpSQws"
            "rZVHRLELRxKdC6c+mXON1HpZPRay7+QZd8Nwj/WSAdOGqY7OQD26HajTufBf2Hmeg+WnbD8i"
            "p3YlAkPs2R1I1XMzoDVGdoF++6MuyQNe1xj/U0PW4PiFpHHdyFz0N0uj7hE/Zrj5+vQNAAD/"
            "/wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVs"
            "cy9zbGlkZUxheW91dDEyLnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG"
            "2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0FsfgScObGPbtctFc"
            "aMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0bUA7M8XJakgnW4O4"
            "viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd9gMAAP//AwBQSwME"
            "FAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAABwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3NsaWRl"
            "TGF5b3V0MTMueG1sLnJlbHOskLtqAzEQRftA/kFMH2l3CxOCtW5CwEUa43zAIM1qRVYPNEqw"
            "/94ybrzgIkXKeZ17mO3uFBbxS4V9ihp62YGgaJL10Wn4On68vILgitHikiJpOBPDbnx+2h5o"
            "wdqOePaZRaNE1jDXmt+UYjNTQJYpU2yTKZWAtZXFqYzmGx2poes2qtwzYFwxxd5qKHs7gDie"
            "M/2FnabJG3pP5idQrA8ilA8tuwGxOKoapFSBrMdbf5A5OlCPNfr/1ODFW/pErlRWMnf91VIv"
            "W8TVTK2+Pl4AAAD//wMAUEsDBBQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRl"
            "TGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE0LnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOI"
            "NHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5cs39No3hSYhe8hlpWIMibYJ3vNdyux9UWBGf0"
            "FsfgScObGPbtctFcaMRcjnhwkUVRPGsYco47pdgMNCHLEMmXSRfShLmUqVcRzR17Uuuq2qj0"
            "bUA7M8XJakgnW4O4viP9Y4euc4YOwTwm8vlHhOLRWTojZ0qFxdRT1iDld3+2VMsSAapt1Ozd"
            "9gMAAP//AwBQSwMEFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAABwcHQvc2xpZGVMYXlvdXRz"
            "L19yZWxzL3NsaWRlTGF5b3V0MTUueG1sLnJlbHOMz70KwjAQB/Bd8B3C7Satg4g0dRHBwUX0"
            "AY7k2gbbJOSi6Nub0YKD4339/lyzf02jeFJiF7yGWlYgyJtgne813K7H1RYEZ/QWx+BJw5sY"
            "9u1y0VxoxFyOeHCRRVE8axhyjjul2Aw0IcsQyZdJF9KEuZSpVxHNHXtS66raqPRtQDszxclq"
            "SCdbg7i+I/1jh65zhg7BPCby+UeE4tFZOiNnSoXF1FPWIOV3f7ZUyxIBqm3U7N32AwAA//8D"
            "AFBLAwQUAAYACAAAACEAyg4Z29UAAAC+AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMv"
            "c2xpZGVMYXlvdXQxNi54bWwucmVsc6yQPUsEMRCGe8H/EKY3s3uFiFz2GhGusJHzBwzJbDa4"
            "+SATxfv3Rmxu4QoLy/l63ofZH77iqj65SsjJwKgHUJxsdiF5A2+n57sHUNIoOVpzYgNnFjhM"
            "tzf7V16p9SNZQhHVKUkMLK2VR0SxC0cSnQunPplzjdR6WT0Wsu/kGXfDcI/1kgHThqmOzkA9"
            "uh2o07nwX9h5noPlp2w/Iqd2JQJD7NkdSNVzM6A1RnaBfvujLskDXtcY/1ND1uD4haRx3chc"
            "9DdLo+4RP2a4+fr0DQAA//8DAFBLAwQUAAYACAAAACEA1dGS8bwAAAA3AQAALQAAAHBwdC9z"
            "bGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxNy54bWwucmVsc4zPvQrCMBAH8F3wHcLt"
            "Jq2DiDR1EcHBRfQBjuTaBtsk5KLo25vRgoPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfV"
            "FgRn9BbH4EnDmxj27XLRXGjEXI54cJFFUTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lr"
            "qtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOGDsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGq"
            "bdTs3fYDAAD//wMAUEsDBBQABgAIAAAAIQDKDhnb1QAAAL4BAAAtAAAAcHB0L3NsaWRlTGF5"
            "b3V0cy9fcmVscy9zbGlkZUxheW91dDE4LnhtbC5yZWxzrJA9SwQxEIZ7wf8Qpjeze4WIXPYa"
            "Ea6wkfMHDMlsNrj5IBPF+/dGbG7hCgvL+Xreh9kfvuKqPrlKyMnAqAdQnGx2IXkDb6fnuwdQ"
            "0ig5WnNiA2cWOEy3N/tXXqn1I1lCEdUpSQwsrZVHRLELRxKdC6c+mXON1HpZPRay7+QZd8Nw"
            "j/WSAdOGqY7OQD26HajTufBf2Hmeg+WnbD8ip3YlAkPs2R1I1XMzoDVGdoF++6MuyQNe1xj/"
            "U0PW4PiFpHHdyFz0N0uj7hE/Zrj5+vQNAAD//wMAUEsDBBQABgAIAAAAIQAZV/VM1gAAAL4B"
            "AAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE5LnhtbC5yZWxzrJC7"
            "agMxEEX7QP5BTB9pdwsTgrVuQsBFGuN8wCDNakVWDzRKsP/eMm684CJFynmde5jt7hQW8UuF"
            "fYoaetmBoGiS9dFp+Dp+vLyC4IrR4pIiaTgTw258ftoeaMHajnj2mUWjRNYw15rflGIzU0CW"
            "KVNskymVgLWVxamM5hsdqaHrNqrcM2BcMcXeaih7O4A4njP9hZ2myRt6T+YnUKwPIpQPLbsB"
            "sTiqGqRUgazHW3+QOTpQjzX6/9TgxVv6RK5UVjJ3/dVSL1vE1Uytvj5eAAAA//8DAFBLAwQU"
            "AAYACAAAACEA1dGS8bwAAAA3AQAALQAAAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVM"
            "YXlvdXQyMC54bWwucmVsc4zPvQrCMBAH8F3wHcLtJq2DiDR1EcHBRfQBjuTaBtsk5KLo25vR"
            "goPjff3+XLN/TaN4UmIXvIZaViDIm2Cd7zXcrsfVFgRn9BbH4EnDmxj27XLRXGjEXI54cJFF"
            "UTxrGHKOO6XYDDQhyxDJl0kX0oS5lKlXEc0de1Lrqtqo9G1AOzPFyWpIJ1uDuL4j/WOHrnOG"
            "DsE8JvL5R4Ti0Vk6I2dKhcXUU9Yg5Xd/tlTLEgGqbdTs3fYDAAD//wMAUEsDBBQABgAIAAAA"
            "IQDV0ZLxvAAAADcBAAAtAAAAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDIx"
            "LnhtbC5yZWxzjM+9CsIwEAfwXfAdwu0mrYOINHURwcFF9AGO5NoG2yTkoujbm9GCg+N9/f5c"
            "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"
            "mzGncX4FAABdFAAAIQAAAAAAAAAAAAAAAAANmwAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxh"
            "eW91dDIueG1sUEsBAi0AFAAGAAgAAAAhAHr1i2GlBAAAew8AACEAAAAAAAAAAAAAAAAAyqAA"
            "AHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQzLnhtbFBLAQItABQABgAIAAAAIQAHSzb7"
            "PAUAALUYAAAhAAAAAAAAAAAAAAAAAK6lAABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0"
            "NC54bWxQSwECLQAUAAYACAAAACEATqf5LXoFAABHHAAAIQAAAAAAAAAAAAAAAAApqwAAcHB0"
            "L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDUueG1sUEsBAi0AFAAGAAgAAAAhAFn+i0WvBgAA"
            "by4AACEAAAAAAAAAAAAAAAAA4rAAAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQ2Lnht"
            "bFBLAQItABQABgAIAAAAIQBRImNfIwcAAMs+AAAhAAAAAAAAAAAAAAAAANC3AABwcHQvc2xp"
            "ZGVMYXlvdXRzL3NsaWRlTGF5b3V0Ny54bWxQSwECLQAUAAYACAAAACEAle8a4yUCAAA4BAAA"
            "IQAAAAAAAAAAAAAAAAAyvwAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDgueG1sUEsB"
            "Ai0AFAAGAAgAAAAhAKgJpvIkBgAAthUAACEAAAAAAAAAAAAAAAAAlsEAAHBwdC9zbGlkZUxh"
            "eW91dHMvc2xpZGVMYXlvdXQ5LnhtbFBLAQItABQABgAIAAAAIQBPB8SMggYAALYZAAAiAAAA"
            "AAAAAAAAAAAAAPnHAABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTAueG1sUEsBAi0A"
            "FAAGAAgAAAAhAGg7nT5VBgAAahUAACIAAAAAAAAAAAAAAAAAu84AAHBwdC9zbGlkZUxheW91"
            "dHMvc2xpZGVMYXlvdXQxMS54bWxQSwECLQAUAAYACAAAACEAtmToy5IEAADoDQAAIgAAAAAA"
            "AAAAAAAAAABQ1QAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDEyLnhtbFBLAQItABQA"
            "BgAIAAAAIQAkel6soQMAAFkHAAAiAAAAAAAAAAAAAAAAACLaAABwcHQvc2xpZGVMYXlvdXRz"
            "L3NsaWRlTGF5b3V0MTMueG1sUEsBAi0AFAAGAAgAAAAhAHO71KudBQAAShUAACIAAAAAAAAA"
            "AAAAAAAAA94AAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQxNC54bWxQSwECLQAUAAYA"
            "CAAAACEAoEOPW9sFAADQHgAAIgAAAAAAAAAAAAAAAADg4wAAcHB0L3NsaWRlTGF5b3V0cy9z"
            "bGlkZUxheW91dDE1LnhtbFBLAQItABQABgAIAAAAIQCJcEK9JAUAAGwNAAAiAAAAAAAAAAAA"
            "AAAAAPvpAABwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MTYueG1sUEsBAi0AFAAGAAgA"
            "AAAhAMOi3AjsBgAA+DAAACIAAAAAAAAAAAAAAAAAX+8AAHBwdC9zbGlkZUxheW91dHMvc2xp"
            "ZGVMYXlvdXQxNy54bWxQSwECLQAUAAYACAAAACEAOOvGMrkFAABdDwAAIgAAAAAAAAAAAAAA"
            "AACL9gAAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDE4LnhtbFBLAQItABQABgAIAAAA"
            "IQC/SEJxnwMAAFoHAAAiAAAAAAAAAAAAAAAAAIT8AABwcHQvc2xpZGVMYXlvdXRzL3NsaWRl"
            "TGF5b3V0MTkueG1sUEsBAi0AFAAGAAgAAAAhACbF+x6dBQAAShUAACIAAAAAAAAAAAAAAAAA"
            "YwABAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlvdXQyMC54bWxQSwECLQAUAAYACAAAACEA"
            "lM1GaNsFAADQHgAAIgAAAAAAAAAAAAAAAABABgEAcHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxh"
            "eW91dDIxLnhtbFBLAQItABQABgAIAAAAIQC3KCqNJQUAAG0NAAAiAAAAAAAAAAAAAAAAAFsM"
            "AQBwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0MjIueG1sUEsBAi0AFAAGAAgAAAAhAJPR"
            "wXfrBgAA+DAAACIAAAAAAAAAAAAAAAAAwBEBAHBwdC9zbGlkZUxheW91dHMvc2xpZGVMYXlv"
            "dXQyMy54bWxQSwECLQAUAAYACAAAACEAvLDkkrkFAABdDwAAIgAAAAAAAAAAAAAAAADrGAEA"
            "cHB0L3NsaWRlTGF5b3V0cy9zbGlkZUxheW91dDI0LnhtbFBLAQItABQABgAIAAAAIQAaE/ML"
            "7AIAAFwHAAAiAAAAAAAAAAAAAAAAAOQeAQBwcHQvc2xpZGVMYXlvdXRzL3NsaWRlTGF5b3V0"
            "MjUueG1sUEsBAi0ACgAAAAAAAAAhANyXZR64GgAAuBoAABQAAAAAAAAAAAAAAAAAECIBAHBw"
            "dC9tZWRpYS9pbWFnZTEucG5nUEsBAi0ACgAAAAAAAAAhAAXp1EzpEQAA6REAABQAAAAAAAAA"
            "AAAAAAAA+jwBAHBwdC9tZWRpYS9pbWFnZTIucG5nUEsBAi0AFAAGAAgAAAAhAAV25sPxAwAA"
            "7hAAABQAAAAAAAAAAAAAAAAAFU8BAHBwdC90aGVtZS90aGVtZTIueG1sUEsBAi0AFAAGAAgA"
            "AAAhALl/7nPtBQAAsBsAABQAAAAAAAAAAAAAAAAAOFMBAHBwdC90aGVtZS90aGVtZTMueG1s"
            "UEsBAi0ACgAAAAAAAAAhAAfzTaa9yAAAvcgAABQAAAAAAAAAAAAAAAAAV1kBAHBwdC9tZWRp"
            "YS9pbWFnZTMucG5nUEsBAi0ACgAAAAAAAAAhAMrRxx9DvAAAQ7wAABQAAAAAAAAAAAAAAAAA"
            "RiICAHBwdC9tZWRpYS9pbWFnZTQucG5nUEsBAi0ACgAAAAAAAAAhANX6HUGAZAEAgGQBABQA"
            "AAAAAAAAAAAAAAAAu94CAHBwdC9tZWRpYS9pbWFnZTUucG5nUEsBAi0ACgAAAAAAAAAhADcn"
            "UEOqaQAAqmkAABQAAAAAAAAAAAAAAAAAbUMEAHBwdC9tZWRpYS9pbWFnZTYucG5nUEsBAi0A"
            "CgAAAAAAAAAhANhD3k5kfgAAZH4AABQAAAAAAAAAAAAAAAAASa0EAHBwdC9tZWRpYS9pbWFn"
            "ZTcucG5nUEsBAi0ACgAAAAAAAAAhAOEsg2TkZgAA5GYAABQAAAAAAAAAAAAAAAAA3ysFAHBw"
            "dC9tZWRpYS9pbWFnZTgucG5nUEsBAi0ACgAAAAAAAAAhADPkvpQkfQAAJH0AABQAAAAAAAAA"
            "AAAAAAAA9ZIFAHBwdC9tZWRpYS9pbWFnZTkucG5nUEsBAi0AFAAGAAgAAAAhAN1PqvJ5MwAA"
            "NjICACEAAAAAAAAAAAAAAAAASxAGAHBwdC9jaGFuZ2VzSW5mb3MvY2hhbmdlc0luZm8xLnht"
            "bFBLAQItABQABgAIAAAAIQBfKtncFwEAAKwBAAAUAAAAAAAAAAAAAAAAAANEBgBwcHQvcmV2"
            "aXNpb25JbmZvLnhtbFBLAQItABQABgAIAAAAIQASNYk5vgEAADIDAAARAAAAAAAAAAAAAAAA"
            "AExFBgBkb2NQcm9wcy9jb3JlLnhtbFBLAQItABQABgAIAAAAIQBNP5vLJAMAAIsHAAAQAAAA"
            "AAAAAAAAAAAAADlHBgBkb2NQcm9wcy9hcHAueG1sUEsBAi0AFAAGAAgAAAAhAGAtAdUNAQAA"
            "kgEAABMAAAAAAAAAAAAAAAAAi0oGAGRvY1Byb3BzL2N1c3RvbS54bWxQSwECLQAUAAYACAAA"
            "ACEA/kXRhwABAADkAgAACwAAAAAAAAAAAAAAAADJSwYAX3JlbHMvLnJlbHNQSwECLQAUAAYA"
            "CAAAACEA2ZEb/N0BAAAHDQAAHwAAAAAAAAAAAAAAAADyTAYAcHB0L19yZWxzL3ByZXNlbnRh"
            "dGlvbi54bWwucmVsc1BLAQItABQABgAIAAAAIQB0Pzl6wgAAACgBAAAeAAAAAAAAAAAAAAAA"
            "AAxPBgBjdXN0b21YbWwvX3JlbHMvaXRlbTEueG1sLnJlbHNQSwECLQAUAAYACAAAACEAXJYn"
            "IsIAAAAoAQAAHgAAAAAAAAAAAAAAAAAKUAYAY3VzdG9tWG1sL19yZWxzL2l0ZW0yLnhtbC5y"
            "ZWxzUEsBAi0AFAAGAAgAAAAhAHvzAqPDAAAAKAEAAB4AAAAAAAAAAAAAAAAACFEGAGN1c3Rv"
            "bVhtbC9fcmVscy9pdGVtMy54bWwucmVsc1BLAQItABQABgAIAAAAIQDIB1wrZAEAAE8QAAAs"
            "AAAAAAAAAAAAAAAAAAdSBgBwcHQvc2xpZGVNYXN0ZXJzL19yZWxzL3NsaWRlTWFzdGVyMS54"
            "bWwucmVsc1BLAQItABQABgAIAAAAIQCQAuz1wQAAADgBAAAgAAAAAAAAAAAAAAAAALVTBgBw"
            "cHQvc2xpZGVzL19yZWxzL3NsaWRlMS54bWwucmVsc1BLAQItABQABgAIAAAAIQBL9T3svQAA"
            "ADcBAAAgAAAAAAAAAAAAAAAAALRUBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlMi54bWwucmVs"
            "c1BLAQItABQABgAIAAAAIQAuohua3gAAAEUCAAAgAAAAAAAAAAAAAAAAAK9VBgBwcHQvc2xp"
            "ZGVzL19yZWxzL3NsaWRlMy54bWwucmVsc1BLAQItABQABgAIAAAAIQBL9T3svQAAADcBAAAg"
            "AAAAAAAAAAAAAAAAAMtWBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlNC54bWwucmVsc1BLAQIt"
            "ABQABgAIAAAAIQDuDIdh1wAAAL4BAAAgAAAAAAAAAAAAAAAAAMZXBgBwcHQvc2xpZGVzL19y"
            "ZWxzL3NsaWRlNS54bWwucmVsc1BLAQItABQABgAIAAAAIQCQAuz1wQAAADgBAAAgAAAAAAAA"
            "AAAAAAAAANtYBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlNi54bWwucmVsc1BLAQItABQABgAI"
            "AAAAIQA9VWv21gAAAL4BAAAgAAAAAAAAAAAAAAAAANpZBgBwcHQvc2xpZGVzL19yZWxzL3Ns"
            "aWRlNy54bWwucmVsc1BLAQItABQABgAIAAAAIQCznx8y1wAAAL4BAAAgAAAAAAAAAAAAAAAA"
            "AO5aBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlOC54bWwucmVsc1BLAQItABQABgAIAAAAIQBq"
            "tNO53gAAAEUCAAAgAAAAAAAAAAAAAAAAAANcBgBwcHQvc2xpZGVzL19yZWxzL3NsaWRlOS54"
            "bWwucmVsc1BLAQItABQABgAIAAAAIQBqtNO53gAAAEUCAAAhAAAAAAAAAAAAAAAAAB9dBgBw"
            "cHQvc2xpZGVzL19yZWxzL3NsaWRlMTAueG1sLnJlbHNQSwECLQAUAAYACAAAACEAtM9YGbkA"
            "AAAkAQAALAAAAAAAAAAAAAAAAAA8XgYAcHB0L25vdGVzTWFzdGVycy9fcmVscy9ub3Rlc01h"
            "c3RlcjEueG1sLnJlbHNQSwECLQAUAAYACAAAACEAk6p9mLkAAAAkAQAAMAAAAAAAAAAAAAAA"
            "AAA/XwYAcHB0L2hhbmRvdXRNYXN0ZXJzL19yZWxzL2hhbmRvdXRNYXN0ZXIxLnhtbC5yZWxz"
            "UEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAARmAGAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXR"
            "kvG8AAAANwEAACwAAAAAAAAAAAAAAAAATGEGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xp"
            "ZGVMYXlvdXQyLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAUmIGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQzLnhtbC5yZWxz"
            "UEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAWGMGAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ0LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXR"
            "kvG8AAAANwEAACwAAAAAAAAAAAAAAAAAXmQGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xp"
            "ZGVMYXlvdXQ1LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAZGUGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ2LnhtbC5yZWxz"
            "UEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAAAAAAAAAAamYGAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ3LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXR"
            "kvG8AAAANwEAACwAAAAAAAAAAAAAAAAAcGcGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xp"
            "ZGVMYXlvdXQ4LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAACwAAAAAAAAA"
            "AAAAAAAAdmgGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQ5LnhtbC5yZWxz"
            "UEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAAAAAAAAAAAAAAAAfGkGAHBwdC9zbGlk"
            "ZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxMC54bWwucmVsc1BLAQItABQABgAIAAAAIQDK"
            "Dhnb1QAAAL4BAAAtAAAAAAAAAAAAAAAAAINqBgBwcHQvc2xpZGVMYXlvdXRzL19yZWxzL3Ns"
            "aWRlTGF5b3V0MTEueG1sLnJlbHNQSwECLQAUAAYACAAAACEA1dGS8bwAAAA3AQAALQAAAAAA"
            "AAAAAAAAAACjawYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDEyLnhtbC5y"
            "ZWxzUEsBAi0AFAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAAAAAAAAAAAAAAAAqmwGAHBwdC9z"
            "bGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxMy54bWwucmVsc1BLAQItABQABgAIAAAA"
            "IQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAMttBgBwcHQvc2xpZGVMYXlvdXRzL19yZWxz"
            "L3NsaWRlTGF5b3V0MTQueG1sLnJlbHNQSwECLQAUAAYACAAAACEA1dGS8bwAAAA3AQAALQAA"
            "AAAAAAAAAAAAAADSbgYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE1Lnht"
            "bC5yZWxzUEsBAi0AFAAGAAgAAAAhAMoOGdvVAAAAvgEAAC0AAAAAAAAAAAAAAAAA2W8GAHBw"
            "dC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxNi54bWwucmVsc1BLAQItABQABgAI"
            "AAAAIQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAPlwBgBwcHQvc2xpZGVMYXlvdXRzL19y"
            "ZWxzL3NsaWRlTGF5b3V0MTcueG1sLnJlbHNQSwECLQAUAAYACAAAACEAyg4Z29UAAAC+AQAA"
            "LQAAAAAAAAAAAAAAAAAAcgYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91dDE4"
            "LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhABlX9UzWAAAAvgEAAC0AAAAAAAAAAAAAAAAAIHMG"
            "AHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQxOS54bWwucmVsc1BLAQItABQA"
            "BgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAEF0BgBwcHQvc2xpZGVMYXlvdXRz"
            "L19yZWxzL3NsaWRlTGF5b3V0MjAueG1sLnJlbHNQSwECLQAUAAYACAAAACEA1dGS8bwAAAA3"
            "AQAALQAAAAAAAAAAAAAAAABIdQYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxheW91"
            "dDIxLnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhAMoOGdvVAAAAvgEAAC0AAAAAAAAAAAAAAAAA"
            "T3YGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyMi54bWwucmVsc1BLAQIt"
            "ABQABgAIAAAAIQDV0ZLxvAAAADcBAAAtAAAAAAAAAAAAAAAAAG93BgBwcHQvc2xpZGVMYXlv"
            "dXRzL19yZWxzL3NsaWRlTGF5b3V0MjMueG1sLnJlbHNQSwECLQAUAAYACAAAACEAyg4Z29UA"
            "AAC+AQAALQAAAAAAAAAAAAAAAAB2eAYAcHB0L3NsaWRlTGF5b3V0cy9fcmVscy9zbGlkZUxh"
            "eW91dDI0LnhtbC5yZWxzUEsBAi0AFAAGAAgAAAAhANXRkvG8AAAANwEAAC0AAAAAAAAAAAAA"
            "AAAAlnkGAHBwdC9zbGlkZUxheW91dHMvX3JlbHMvc2xpZGVMYXlvdXQyNS54bWwucmVsc1BL"
            "AQItABQABgAIAAAAIQBtED1TqwIAAIIeAAATAAAAAAAAAAAAAAAAAJ16BgBbQ29udGVudF9U"
            "eXBlc10ueG1sUEsFBgAAAABuAG4APCEAAHl9BgAAAA=="
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
            """shapeの全runをクリアして最初のrunにテキストを設定"""
            if not shape.has_text_frame: return
            tf = shape.text_frame
            for _para in tf.paragraphs:
                for _run in _para.runs:
                    _run.text = ''
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
                    from pptx.util import Pt as _Pt4, RGBColor as _RGB4
                    from pptx.oxml.ns import qn as _qn4
                    from lxml import etree as _et4
                    import copy as _copy4

                    def _make_run(para, text, bold=False, color=None, size=9):
                        """パラグラフに新しいrunを追加する"""
                        r = _et4.SubElement(para._p, _qn4('a:r'))
                        rPr = _et4.SubElement(r, _qn4('a:rPr'), attrib={'lang':'ja-JP','altLang':'en-US','dirty':'0'})
                        if bold:
                            rPr.set('b', '1')
                        if size:
                            rPr.set('sz', str(int(size * 100)))
                        if color:
                            solidFill = _et4.SubElement(rPr, _qn4('a:solidFill'))
                            srgb = _et4.SubElement(solidFill, _qn4('a:srgbClr'))
                            srgb.set('val', color.replace('#',''))
                        t = _et4.SubElement(r, _qn4('a:t'))
                        t.text = text
                        return r

                    # テキストフレームの全段落をクリア
                    for para in tf.paragraphs:
                        for r in para._p.findall(_qn4('a:r')):
                            para._p.remove(r)

                    # 段落数を確認・調整（最低7段落必要）
                    _needed = 7
                    while len(tf.paragraphs) < _needed:
                        _new_p = _et4.SubElement(tf._txBody, _qn4('a:p'))

                    _paras = tf.paragraphs

                    # 段落0: 「このテーブルの読み方」（小見出し）
                    _make_run(_paras[0], 'このテーブルの読み方', bold=True, color='#3a6a7a', size=8)

                    # 段落1: 空行（テーブルとの間隔）
                    pass  # 空のまま

                    # 段落2: λ説明
                    _make_run(_paras[2], f'・λ={round(lam_actual):,}日（約{lam_actual/365:.1f}年）は中程度の継続期間で、1〜2年継続する顧客が多いビジネスです。', size=9)

                    # 段落3: k説明
                    _make_run(_paras[3], f'・k={k:.3f}（{k_type}）: 契約直後に大量離脱するパターンです。LTV∞ は大きく見えますが少数の超長期顧客の分が含まれており、99%到達まで長期間かかります。CAC投資判断には暫定LTV（現実的な期間）を使ってください。', size=9)

                    # 段落4: LTV∞説明（1行で、年次データを色付きで）
                    _make_run(_paras[4], f'・LTV∞（¥{ltv_rev:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。', size=9)
                    _make_run(_paras[4], f'1年時点', color='#56b4d3', size=9)
                    _make_run(_paras[4], f'でLTV∞の', size=9)
                    _make_run(_paras[4], f'{all_rows_pp[1][4]:.1f}%（¥{all_rows_pp[1][1]:,.0f}）', bold=True, color='#a8dadc', size=9)
                    _make_run(_paras[4], f'、', size=9)
                    _make_run(_paras[4], f'2年時点', color='#56b4d3', size=9)
                    _make_run(_paras[4], f'で', size=9)
                    _make_run(_paras[4], f'{all_rows_pp[2][4]:.1f}%（¥{all_rows_pp[2][1]:,.0f}）', bold=True, color='#a8dadc', size=9)
                    _make_run(_paras[4], f'、', size=9)
                    _make_run(_paras[4], f'3年時点', color='#56b4d3', size=9)
                    _make_run(_paras[4], f'で', size=9)
                    _make_run(_paras[4], f'{all_rows_pp[3][4]:.1f}%（¥{all_rows_pp[3][1]:,.0f}）に到達します。', bold=True, color='#a8dadc', size=9)

                    # 段落5: CAC説明（色付き）
                    _make_run(_paras[5], f'・CAC上限（¥{cac_upper:,.0f}）の回収期間：売上ベース 約 ', size=9)
                    _make_run(_paras[5], f'{cac_recover_rev_str}', bold=True, color='#a8dadc', size=9)
                    _make_run(_paras[5], f' / 粗利ベース 約 ', size=9)
                    _make_run(_paras[5], f'{cac_recover_gp_str}', bold=True, color='#56b4d3', size=9)
                    _make_run(_paras[5], f'（契約から）', size=9)

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
        _fp5k = dict(fontproperties=_fp5) if _fp5 else {}
        _ax5.plot(t_range, rev_line, color='#56b4d3', lw=2, label='LTV（売上）')
        _ax5.plot(t_range, gp_line,  color='#a8dadc', lw=2, ls='--', label='LTV（粗利）')
        _ax5.plot(t_range, cac_line, color='#4a7a8a', lw=1.5, ls=':', label='CAC上限')
        _ax5.axhline(ltv_rev, color='#56b4d3', lw=0.8, ls=':', alpha=0.5, label=f'LTV∞ ¥{ltv_rev:,.0f}')
        _ax5.axvline(lam_actual, color='#a8dadc', lw=1.2, ls='--', alpha=0.7)
        _xtick_vals = [180, 365, 730, 1095, 1460, 1825]
        _ax5.set_xticks(_xtick_vals)
        _ax5.set_xticklabels(['180日', '1年', '2年', '3年', '4年', '5年'], **({k:v for k,v in {'fontproperties':_fp5}.items() if _fp5} if _fp5 else {}))
        _ax5.set_xlim(0, x_max + 50)
        _ax5.set_xlabel('継続期間', color='#888', fontsize=9, **_fp5k)
        _ax5.set_ylabel('金額（円）', color='#888', fontsize=9, **_fp5k)
        _ax5.tick_params(colors='#888')
        _ax5.yaxis.set_major_formatter(_plt5.FuncFormatter(lambda v, _: f'¥{v:,.0f}'))
        if _fp5:
            _ax5.legend(fontsize=8, framealpha=0.2, labelcolor='white', loc='upper left', prop=_fp5)
        else:
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
                                f"TOP PICK　{best['seg']}",
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
                        _fp_bk = dict(fontproperties=_fp_bar) if _fp_bar else {}
                        _ax.set_axisbelow(True)
                        _ax.grid(axis='y', alpha=0.2, color='#1a3040', zorder=0)
                        _bars = _ax.bar(_xs, _ltvs, color=_cols, width=0.55, zorder=2)
                        _ax.set_xticks(list(_xs))
                        _ax.set_xticklabels(_segs, fontsize=8, color='#cccccc', **_fp_bk)
                        for _b, _v in zip(_bars, _ltvs):
                            _ax.text(_b.get_x()+_b.get_width()/2, _b.get_height()+max(_ltvs)*0.01,
                                    f'¥{_v:,.0f}', ha='center', va='bottom', fontsize=9, color='#cccccc',
                                    **_fp_bk, zorder=3)
                        _ax.set_ylabel('LTV∞（¥）', color='#888', fontsize=9, **_fp_bk)
                        _ax.tick_params(axis='y', colors='#888', labelsize=8)
                        _ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda v,_: f'¥{v:,.0f}'))
                        for _sp in _ax.spines.values(): _sp.set_color('#1a3040')
                        _fig.tight_layout()
                        _buf_bar = io.BytesIO()
                        _fig.savefig(_buf_bar, format='png', dpi=130, bbox_inches='tight', facecolor='#111820')
                        _buf_bar.seek(0); _plt_bar.close()
                        _replace_image(s7, sh, _buf_bar)

                # ── Slide 8: セグメントサマリー（常に複製）──
                s8 = _copy_slide(prs, 7)

                for sh in s8.shapes:
                    if sh.name == 'テキスト プレースホルダー 4' and sh.has_text_frame:
                        _set_text(sh, f'セグメント分析結果のサマリー：{sc}')
                    elif sh.name == 'コンテンツ プレースホルダー 6':
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
                        _fp_tk = dict(fontproperties=_fp_t) if _fp_t else {}
                        for _cx, _cw, _h in zip(_col_x, _norm_w, _hdrs):
                            _al = 'left' if _cx == 0 else 'right'
                            _tx = _cx + (0.01 if _al == 'left' else _cw - 0.01)
                            _ax_t.text(_tx, 1-_hdr_h/2, _h, transform=_ax_t.transAxes,
                                      ha=_al, va='center', fontsize=7.5, color='#56B4D3', fontweight='bold', **_fp_tk)
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
                                          ha=_al2, va='center', fontsize=7.5, color=_tc, **_fp_tk)
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
                                # 全runをクリアしてから設定（タイトルはsc: sv のみ）
                                if sh.has_text_frame:
                                    for para in sh.text_frame.paragraphs:
                                        for run in para.runs: run.text = ''
                                    if sh.text_frame.paragraphs and sh.text_frame.paragraphs[0].runs:
                                        sh.text_frame.paragraphs[0].runs[0].text = f'{sc}: {str(sv)}'
                            elif sh.name == 'テキスト ボックス 17':
                                # TOP PICK ラベル（S9 Top Pickスライドのみ存在）
                                if sh.has_text_frame:
                                    for para in sh.text_frame.paragraphs:
                                        for run in para.runs: run.text = ''
                                    if sh.text_frame.paragraphs and sh.text_frame.paragraphs[0].runs:
                                        sh.text_frame.paragraphs[0].runs[0].text = 'TOP PICK　'
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
    except ImportError as _ie:
        _pp_html = f'<span class="dl-btn-err">.pptx 未対応: {str(_ie)[:80]}</span>'
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
