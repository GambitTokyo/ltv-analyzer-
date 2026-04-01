# lang.py — LTV Analyzer 言語/通貨設定
# ═══════════════════════════════════════════════════════════════

CURRENCIES = {
    'JPY': {'symbol': '¥',  'prefix': True,  'decimal': 0},
    'USD': {'symbol': '$',  'prefix': True,  'decimal': 2},
    'EUR': {'symbol': '€',  'prefix': True,  'decimal': 2},
    'GBP': {'symbol': '£',  'prefix': True,  'decimal': 2},
    'KRW': {'symbol': '₩',  'prefix': True,  'decimal': 0},
    'CNY': {'symbol': '¥',  'prefix': True,  'decimal': 2},
    'TWD': {'symbol': 'NT$','prefix': True,  'decimal': 0},
    'INR': {'symbol': '₹',  'prefix': True,  'decimal': 2},
    'none':{'symbol': '',   'prefix': True,  'decimal': 0},
}

LANG_DEFAULTS = {
    'ja': 'JPY',
    'en': 'USD',
}

# ── 通貨フォーマッター ────────────────────────────────────────

def fmt_c(val, cur='JPY'):
    """通貨フォーマット — fmt_c(12345, 'JPY') → '¥12,345'"""
    c = CURRENCIES.get(cur, CURRENCIES['JPY'])
    d = c['decimal']
    s = c['symbol']
    if d == 0:
        num = f'{val:,.0f}'
    else:
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
