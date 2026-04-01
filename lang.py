# lang.py — LTV Analyzer 言語/通貨設定
# ═══════════════════════════════════════════════════════════════

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
