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
    _oc1, _oc2 = st.columns(2)
    with _oc1:
        outlier_upper_pct = st.number_input(
            "上位除外 (%)", min_value=0.0, max_value=20.0,
            value=0.0, step=0.5, format="%.1f",
            help="累計金額の上位○%を除外します。0%で除外なし。"
        )
    with _oc2:
        outlier_lower_pct = st.number_input(
            "下位除外 (%)", min_value=0.0, max_value=20.0,
            value=0.0, step=0.5, format="%.1f",
            help="累計金額の下位○%を除外します。0%で除外なし。"
        )
    outlier_removal = (outlier_upper_pct > 0) or (outlier_lower_pct > 0)
    st.caption(
        "売上分布のヒストグラムとカットラインが分析結果の手前に表示されます。"
        "分布を確認してから除外率を調整してください。"
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
        min_value=1, max_value=10, value=5,
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
    report_title = st.text_input("レポートタイトル", "", placeholder="月額SaaS顧客LTV分析など")
    client_name  = st.text_input("クライアント名", "", placeholder="会社・ブランド・商品/サービスなど")
    analyst_name = st.text_input("作成者", "", placeholder="氏名・チーム・部署・組織など")

# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 16px 0 32px 0; border-bottom: 1px solid #1a2a3a; margin-bottom: 28px;'>
  <div style='font-family: 'BIZ UDPGothic', sans-serif; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: #3a6a7a; margin-bottom: 8px;'>Analytics Tool</div>
  <div style='font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: #c8d0d8; letter-spacing: -0.03em; line-height: 1;'>LTV Analyzer <span style='color: #56b4d3;'>Advanced</span></div>
  <div style='font-size: 0.78rem; color: #3a5a6a; margin-top: 8px; letter-spacing: 0.02em;'>Kaplan–Meier × Weibull — Segment-level LTV Intelligence &nbsp;·&nbsp; v268</div>
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
    n_outlier_upper = 0
    n_outlier_lower = 0
    # 除外前の分布データを保持（ヒストグラム表示用）
    _rev_before_outlier = df['revenue_total'].copy()
    if outlier_removal:
        before = len(df)
        # 上位パーセンタイル除外
        if outlier_upper_pct > 0:
            upper_r = df['revenue_total'].quantile(1.0 - outlier_upper_pct / 100.0)
            n_outlier_upper = (df['revenue_total'] > upper_r).sum()
        else:
            upper_r = df['revenue_total'].max() + 1  # 除外なし
        # 下位パーセンタイル除外
        if outlier_lower_pct > 0:
            lower_r = df['revenue_total'].quantile(outlier_lower_pct / 100.0)
            n_outlier_lower = (df['revenue_total'] < lower_r).sum()
        else:
            lower_r = df['revenue_total'].min() - 1  # 除外なし
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
        _parts = []
        if n_outlier_upper > 0:
            _parts.append(f"上位{n_outlier_upper:,}件（上位{outlier_upper_pct:.1f}%）")
        if n_outlier_lower > 0:
            _parts.append(f"下位{n_outlier_lower:,}件（下位{outlier_lower_pct:.1f}%）")
        _pct = n_outlier / (n_outlier + len(df)) * 100
        st.info(f"{n_outlier:,}件を異常値として除外しました（{'、'.join(_parts)}）。除外率 {_pct:.1f}%、残り {len(df):,}件で分析します。")

    # ── 売上分布ヒストグラム ──
    import plotly.graph_objects as go
    _rev_data = _rev_before_outlier
    _hist_fig = go.Figure()
    # ヒストグラム（除外前の全データ）
    _hist_fig.add_trace(go.Histogram(
        x=_rev_data, nbinsx=80,
        marker_color='rgba(180, 180, 180, 0.45)',
        marker_line=dict(color='rgba(200, 200, 200, 0.6)', width=0.5),
        name='売上分布',
    ))
    # カットライン表示（st.info青系 #6CB4EE に統一）
    _cut_color = '#6CB4EE'
    _cutline_shapes = []
    _cutline_annots = []
    if outlier_upper_pct > 0:
        _upper_val = _rev_data.quantile(1.0 - outlier_upper_pct / 100.0)
        _cutline_shapes.append(dict(
            type='line', x0=_upper_val, x1=_upper_val, y0=0, y1=1,
            yref='paper', line=dict(color=_cut_color, width=2, dash='dash'),
        ))
        _cutline_annots.append(dict(
            x=_upper_val, y=1.02, yref='paper', text=f'上位{outlier_upper_pct:.1f}%<br>¥{_upper_val:,.0f}',
            showarrow=False, font=dict(color=_cut_color, size=11), xanchor='left',
        ))
    if outlier_lower_pct > 0:
        _lower_val = _rev_data.quantile(outlier_lower_pct / 100.0)
        # 下位カットラインがY軸と重ならないよう、値が極小の場合はスキップ
        _x_range = _rev_data.max() - _rev_data.min()
        if _lower_val > _rev_data.min() + _x_range * 0.005:
            _cutline_shapes.append(dict(
                type='line', x0=_lower_val, x1=_lower_val, y0=0, y1=1,
                yref='paper', line=dict(color=_cut_color, width=2, dash='dash'),
            ))
            _cutline_annots.append(dict(
                x=_lower_val, y=1.02, yref='paper', text=f'下位{outlier_lower_pct:.1f}%<br>¥{_lower_val:,.0f}',
                showarrow=False, font=dict(color=_cut_color, size=11), xanchor='left',
            ))
    _hist_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text='累計売上の分布', font=dict(size=14)),
        xaxis=dict(
            title='累計売上 (¥)', tickformat=',',
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=False,
        ),
        yaxis=dict(
            title='顧客数',
            gridcolor='rgba(255,255,255,0.06)',
            zeroline=False,
            layer='below traces',
        ),
        shapes=_cutline_shapes,
        annotations=_cutline_annots,
        margin=dict(l=50, r=30, t=60, b=40),
        height=300,
        showlegend=False,
    )
    # サマリー指標
    _sc1, _sc2, _sc3, _sc4 = st.columns(4)
    _sc1.metric("全顧客数", f"{len(_rev_data):,}")
    _sc2.metric("除外件数", f"{n_outlier:,}")
    _sc3.metric("除外率", f"{n_outlier / (n_outlier + len(df)) * 100 if (n_outlier + len(df)) > 0 else 0:.1f}%")
    _sc4.metric("分析対象", f"{len(df):,}")
    st.plotly_chart(_hist_fig, use_container_width=True)
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

st.markdown("<div class='section-title'>暫定 LTV — 観測期間別</div>", unsafe_allow_html=True)

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
    # 線グラフ（マーカーなし）
    fig_ltv.add_trace(go.Scatter(x=t_range, y=rev_line, name='LTV（売上）', mode='lines', line=dict(color='#56b4d3', width=2), showlegend=True))
    fig_ltv.add_trace(go.Scatter(x=t_range, y=gp_line,  name='LTV（粗利）', mode='lines', line=dict(color='#a8dadc', width=2, dash='dash'), showlegend=True))
    fig_ltv.add_trace(go.Scatter(x=t_range, y=cac_line, name='CAC上限',    mode='lines', line=dict(color='#4a7a8a', width=1.5, dash='dot'), showlegend=True))

    # 特定ポイントにマーカー（180日,1年,2年,3年,4年,5年,λ）
    _marker_days = sorted(set([180, 365, 730, 1095, 1460, 1825, round(lam_actual)]))
    for _trace_data, _color, _name in [(rev_line, '#56b4d3', 'LTV（売上）'), (gp_line, '#a8dadc', 'LTV（粗利）'), (cac_line, '#4a7a8a', 'CAC上限')]:
        _mx = [d for d in _marker_days if d <= max(t_range)]
        _my = []
        for d in _mx:
            # t_rangeから最も近いインデックスを探す
            _closest = min(range(len(t_range)), key=lambda i: abs(t_range[i] - d))
            _my.append(_trace_data[_closest])
        fig_ltv.add_trace(go.Scatter(x=_mx, y=_my, mode='markers', marker=dict(color=_color, size=6, symbol='circle'), showlegend=False, hoverinfo='skip'))

    fig_ltv.add_hline(y=ltv_rev, line_dash='dot', line_color='#56b4d3', line_width=1, opacity=0.4,
        annotation_text=f'LTV∞ ¥{ltv_rev:,.0f}', annotation_position='right',
        annotation_font=dict(color='#56b4d3', size=10))
    fig_ltv.add_shape(type='line', x0=lam_actual, x1=lam_actual, y0=0, y1=ltv_rev,
        line=dict(color='#a8dadc', width=1.5, dash='dash'), layer='above')
    fig_ltv.add_annotation(x=lam_actual, y=0.85 if k < 1.0 else 0.35, yref='paper',
        text=f'λ＝{lam_int}日', showarrow=False,
        font=dict(color='#a8dadc', size=10), xanchor='center', yanchor='middle',
        bgcolor='#111820', borderpad=2)
    tick_vals = [180, 365, 730, 1095, 1460, 1825]
    tick_text = ['180日', '1年', '2年', '3年', '4年', '5年']
    fig_ltv.update_layout(
        paper_bgcolor='#111820', plot_bgcolor='#111820',
        height=300, margin=dict(t=40, b=50, l=70, r=120),
        font=dict(color='#ccc', size=10),
        legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title='継続期間', gridcolor='#1a3040', tickvals=tick_vals, ticktext=tick_text, tickfont=dict(color='#888'), range=[0, x_max + 50]),
        yaxis=dict(title='金額（円）', gridcolor='#1a3040', tickfont=dict(color='#888'), tickformat=',', tickprefix=''),
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

# matplotlibにフォントを明示登録（Streamlit Cloud対応）
try:
    import matplotlib as _mpl_init
    import matplotlib.font_manager as _fm_init
    _IPA_TTF = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    import os as _os_init
    if _os_init.path.exists(_IPA_TTF):
        _fm_init.fontManager.addfont(_IPA_TTF)
        _JP_FONT_PATH = _IPA_TTF
    elif _JP_FONT_PATH:
        _fm_init.fontManager.addfont(_JP_FONT_PATH)
    _JP_FONT_NAME = _fm_init.FontProperties(fname=_JP_FONT_PATH).get_name() if _JP_FONT_PATH else None
except Exception:
    _JP_FONT_NAME = None



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
  <div style='margin-top:8px;'>・CAC上限（¥{cac_upper:,.0f}）の回収期間：売上ベース 約 <b style='color:#a8dadc;'>{cac_recover_rev_str}</b> / 粗利ベース 約 <b style='color:#56b4d3;'>{cac_recover_gp_str}</b></div>
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
・Daily ARPU（売上）: ¥{arpu_daily:,.2f} / Daily GP（粗利）: ¥{gp_daily:,.2f} / GPM: {gpm:.1%}
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
            ('Daily ARPU（売上ベース・¥）', round(arpu_daily, 2)),
            ('粗利率（GPM）', f'{gpm:.1%}'),
            ('Daily GP（粗利ベース・¥）', round(gp_daily, 2)),
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
        import os as _os, sys as _sys
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _sys.path.insert(0, _here)
        from pptx_export import generate_pptx
        from pptx import Presentation

        _TMPL_PATH = _os.path.join(_here, 'LTV-analyzer.pptx')

        _billing_disp = billing_cycle_display if 'billing_cycle_display' in dir() else billing_cycle.split('←')[0].strip()

        # S4ガイドデータを組み立て
        _lam_display = lam + ltv_offset_days if business_type == "都度購入型" else lam
        _s4_guide = {
            'lam_desc': lam_desc,
            'k_desc': k_desc,
            'ltv_rev': ltv_rev,
            'pct_1y': pct_1y, 'ltv_1y': ltv_1y,
            'pct_2y': pct_2y, 'ltv_2y': ltv_2y,
            'pct_3y': pct_3y, 'ltv_3y': ltv_3y,
            'cac_upper': cac_upper,
            'cac_recover_rev_str': cac_recover_rev_str,
            'cac_recover_gp_str': cac_recover_gp_str,
            'lam_actual_round': round(_lam_display),
            'lam_years': _lam_display / 365,
            'lam_gp': lam_gp,
            'lam_meaning': "リピート顧客の63.2%が離脱するまでの期間（初回購入起点）" if business_type == "都度購入型" else "多くの顧客が離脱するまでの期間の目安",
        }

        # 異常値処理の表示文字列
        _outlier_parts = []
        if outlier_upper_pct > 0:
            _outlier_parts.append(f"上位{outlier_upper_pct:.1f}%")
        if outlier_lower_pct > 0:
            _outlier_parts.append(f"下位{outlier_lower_pct:.1f}%")
        _outlier_label = '、'.join(_outlier_parts) if _outlier_parts else "除外なし"

        pptx_buf = generate_pptx(
            tmpl_path=_TMPL_PATH,
            k=k, lam=lam, lam_actual=lam_actual, r2=r2,
            arpu_daily=arpu_daily, gpm=gpm, gp_daily=gp_daily,
            cac_n=cac_n, cac_upper=cac_upper,
            ltv_rev=ltv_rev, lam_rev=lam_rev, lam_gp=lam_gp,
            rev_99=rev_99, gp_99=gp_99, days_99=days_99,
            t_range=t_range, rev_line=rev_line, gp_line=gp_line,
            cac_line=cac_line, x_max=x_max,
            tbl_rows=tbl_rows, horizons=horizons,
            ltv_offset_days=ltv_offset_days,
            ltv_horizon_offset=ltv_horizon_offset,
            ltv_horizon_spot=ltv_horizon_spot,
            business_type=business_type,
            dormancy_days=dormancy_days if 'dormancy_days' in dir() else None,
            buf1=buf1, buf2=buf2,
            df=df, client_name=client_name, analyst_name=analyst_name,
            billing_cycle_display=_billing_disp,
            k_summary=k_summary, r2_summary=r2_summary,
            cac_label=cac_label,
            cac_recover_rev_str=cac_recover_rev_str,
            cac_recover_gp_str=cac_recover_gp_str,
            segment_cols_input=segment_cols_input,
            _compute_km_df=_compute_km_df,
            _fit_weibull_df=_fit_weibull_df,
            ltv_inf=ltv_inf,
            fmt_horizon=fmt_horizon,
            s4_guide_data=_s4_guide,
            report_title=report_title,
            arpu_0_dorm=arpu_0_dorm if 'arpu_0_dorm' in dir() else arpu_daily,
            arpu_long=arpu_long if 'arpu_long' in dir() else arpu_daily,
            outlier_label=_outlier_label,
        )

        import base64 as _b64
        _pp_b64 = _b64.b64encode(pptx_buf.read()).decode()
        _fn_pp  = f"LTV分析_{client_name or 'report'}.pptx"
        _pp_href = (f'<a href="data:application/vnd.openxmlformats-officedocument'
                    f'.presentationml.presentation;base64,{_pp_b64}" '
                    f'download="{_fn_pp}" class="dl-btn">.pptx</a>')
        _pp_html = _pp_href
    except ImportError as _ie:
        _pp_html = f'<span class="dl-btn-err">.pptx 未対応: {str(_ie)[:80]}</span>'
    except Exception as e:
        import traceback as _tb
        _tb_str = _tb.format_exc().replace('\n', ' | ')[-300:]
        _pp_html = (f'<span class="dl-btn-err">.pptx エラー: {str(e)[:150]}'
                    f'<br><small style="font-size:0.6rem;opacity:0.7">{_tb_str}</small></span>')

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
            ['Daily ARPU（売上）', f'¥{arpu_daily:,.2f}'],
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
            f"Daily ARPU（売上）: ¥{arpu_daily:,.2f} / GPM: {gpm:.0%}\n"
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
                cac_avg_seg = (seg_df['CAC上限（粗利）'] * seg_df['顧客数']).sum() / seg_df['顧客数'].sum()
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

            # Plotlyで横棒グラフ（PPTXと統一）
            n_seg = len(seg_df)
            bar_colors = [ACCENT if i == 0 else ACCENT2 for i in range(n_seg)]
            fig_height = max(300, n_seg * 35 + 100)

            fig_plotly = go.Figure()
            seg_names = seg_df['セグメント'].astype(str).tolist()[::-1]
            seg_vals  = seg_df['LTV∞（売上）'].tolist()[::-1]
            bar_colors_rev = bar_colors[::-1]
            fig_plotly.add_trace(go.Bar(
                y=seg_names,
                x=seg_vals,
                orientation='h',
                marker=dict(color=bar_colors_rev),
                text=[f'¥{v:,.0f}' for v in seg_vals],
                textposition='inside',
                insidetextanchor='end',
                textfont=dict(color='white', size=10),
                hovertemplate='%{y}: ¥%{x:,.0f}<extra></extra>',
            ))
            fig_plotly.update_yaxes(
                tickfont=dict(color='#aaa', size=10 if n_seg <= 20 else 8),
                gridcolor='#1a3040',
                linecolor='#1a3040',
            )
            fig_plotly.update_xaxes(
                title_text='LTV∞（¥）',
                tickfont=dict(color='#888'),
                gridcolor='#1a3040',
                tickprefix='¥',
                tickformat=',',
                showgrid=True,
                side='top',
            )
            fig_plotly.update_layout(
                title_text='',
                paper_bgcolor='#111820',
                plot_bgcolor='#111820',
                height=fig_height,
                showlegend=False,
                margin=dict(t=30, b=40, l=120, r=30),
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
                        st.markdown("<div style='font-size:0.75rem; color:#7ab4c4; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-top:16px; margin-bottom:6px;'>暫定 LTV — 観測期間別</div>", unsafe_allow_html=True)

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
                        fig_hor_s.add_shape(type='line', x0=lam_s_actual, x1=lam_s_actual, y0=0, y1=ltv_inf_s_offset,
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
                            height=280, margin=dict(t=40, b=50, l=70, r=120),
                            font=dict(color='#ccc', size=10),
                            legend=dict(orientation='h', y=1.08, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
                            xaxis=dict(title='継続期間', gridcolor='#1a3040', tickvals=tick_vals_s, ticktext=tick_text_s, tickfont=dict(color='#888')),
                            yaxis=dict(title='金額（円）', gridcolor='#1a3040', tickfont=dict(color='#888'), tickformat=',', tickprefix=''),
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
    st.write(f"有効データ: {len(df):,}件 ／ 解約: {df['event'].sum():,}件 ／ 継続中: {(df['event']==0).sum():,}件 ／ Daily ARPU: ¥{arpu_daily:,.2f}")
    st.dataframe(
        df[['customer_id','start_date','end_date','duration','event','arpu_daily']].head(30),
        hide_index=True
    )

st.markdown("---")
st.markdown("<p style='color:#333; font-size:0.82rem; text-align:center;'>LTV Analyzer — KM × Weibull Model — Built for marketing analytics professionals</p>", unsafe_allow_html=True)
