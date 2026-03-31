"""PPTX Export Module v245"""
import io, copy, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from lxml import etree
from datetime import date

_JP_FP = None
def _init():
    global _JP_FP
    _here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_here, 'ipag.ttf'),
        '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf',
        '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
        '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p); _JP_FP = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = _JP_FP.get_name()
            plt.rcParams['axes.unicode_minus'] = False; return
    for p in fm.findSystemFonts():
        if any(k in p.lower() for k in ['cjk','ipag','japanese','gothic']):
            fm.fontManager.addfont(p); _JP_FP = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = _JP_FP.get_name()
            plt.rcParams['axes.unicode_minus'] = False; return
_init()
BG = '#111820'
def _fp(): return _JP_FP or fm.FontProperties()
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

def _set_text(sh, txt):
    if not sh.has_text_frame: return
    tf = sh.text_frame
    for p in tf.paragraphs:
        for r in p.runs: r.text = ''
    if tf.paragraphs and tf.paragraphs[0].runs:
        tf.paragraphs[0].runs[0].text = txt
    elif tf.paragraphs and txt:
        # runがない場合: endParaRPrの書式を引き継いでrunを新規作成
        p0 = tf.paragraphs[0]._p
        epr = p0.find(f'{{{A}}}endParaRPr')
        import copy as _cp
        r = etree.SubElement(p0, f'{{{A}}}r')
        if epr is not None:
            rPr = _cp.deepcopy(epr)
            rPr.tag = f'{{{A}}}rPr'
            r.insert(0, rPr)
        t = etree.SubElement(r, f'{{{A}}}t')
        t.text = txt
        # runをendParaRPrの前に移動
        if epr is not None:
            p0.remove(r)
            p0.insert(list(p0).index(epr), r)

def _replace_image(sld, sh, buf):
    blip = sh._element.find('.//' + qn('a:blip'))
    if blip is None: return
    buf.seek(0); ip, rId = sld.part.get_or_add_image_part(buf); blip.set(qn('r:embed'), rId)

def _write_table(tbl, hdr, rows, footer=None):
    data = rows + ([footer] if footer else [])
    need = max(2, 1 + len(data))
    tr = copy.deepcopy(tbl.rows[1]._tr) if len(tbl.rows) > 1 else None
    while len(tbl.rows) < need and tr: tbl._tbl.append(copy.deepcopy(tr))
    while len(tbl.rows) > need and len(tbl.rows) > 1: tbl._tbl.remove(tbl.rows[len(tbl.rows)-1]._tr)
    for ri, rd in enumerate(data):
        if ri+1 >= len(tbl.rows): break
        row = tbl.rows[ri+1]
        for ci, v in enumerate(rd):
            if ci >= len(row.cells): break
            tf = row.cells[ci].text_frame
            for p in tf.paragraphs:
                for r in p.runs: r.text = ''
            if tf.paragraphs and tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text = str(v)

def _write_table_styled(tbl, hdr, rows, footer=None):
    _write_table(tbl, hdr, rows, footer)
    data = rows + ([footer] if footer else [])
    for ri in range(len(data)):
        if ri+1 >= len(tbl.rows): break
        row = tbl.rows[ri+1]; is_ft = footer and ri == len(data)-1
        bg = '142030' if is_ft else ('0D1520' if ri%2==0 else '0A1018')
        fc = 'A8DADC' if is_ft else 'C8D0D8'
        for ci in range(len(row.cells)):
            cell = row.cells[ci]
            tcPr = cell._tc.get_or_add_tcPr()
            for sf in tcPr.findall(f'{{{A}}}solidFill'): tcPr.remove(sf)
            sf = etree.SubElement(tcPr, f'{{{A}}}solidFill')
            c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', bg)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    rPr = r._r.find(f'{{{A}}}rPr')
                    if rPr is None: rPr = etree.SubElement(r._r, f'{{{A}}}rPr'); r._r.insert(0, rPr)
                    for sf2 in rPr.findall(f'{{{A}}}solidFill'): rPr.remove(sf2)
                    sf2 = etree.Element(f'{{{A}}}solidFill')
                    c2 = etree.SubElement(sf2, f'{{{A}}}srgbClr'); c2.set('val', fc)
                    rPr.insert(0, sf2)

def _copy_slide(prs, idx):
    src = prs.slides[idx]; new = prs.slides.add_slide(src.slide_layout)
    for el in list(new.shapes._spTree): new.shapes._spTree.remove(el)
    for el in list(src.shapes._spTree): new.shapes._spTree.append(copy.deepcopy(el))
    for rId, rel in src.part.rels.items():
        if rel.reltype == RT.IMAGE: new.part.rels._rels[rId] = rel
    # コピーされたスライドに明示的にダーク背景を設定
    P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
    cSld = new._element.find(f'{{{P}}}cSld')
    if cSld is not None:
        # 既存のbgを削除
        for bg in cSld.findall(f'{{{P}}}bg'): cSld.remove(bg)
        # ダーク背景を追加（#0A0E14）
        bg = etree.SubElement(cSld, f'{{{P}}}bg')
        bgPr = etree.SubElement(bg, f'{{{P}}}bgPr')
        sf = etree.SubElement(bgPr, f'{{{A}}}solidFill')
        c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', '0A0E14')
        etree.SubElement(bgPr, f'{{{A}}}effectLst')
        # bgをcSldの最初の子要素に移動（PPTXの仕様上bgはspTreeの前にあるべき）
        cSld.remove(bg)
        cSld.insert(0, bg)
    return new

def _set_note_text(sh, note_text):
    """NOTE: テンプレートrunの書式を保持しつつテキスト差し替え+色修正"""
    if not sh.has_text_frame: return
    p0 = sh.text_frame.paragraphs[0] if sh.text_frame.paragraphs else None
    if p0 is None: return
    runs = list(p0.runs)
    if ' — ' in note_text:
        prefix, body = note_text.split(' — ', 1)
        body = '\xa0— ' + body
    else:
        prefix, body = note_text, ''
    if len(runs) >= 2:
        runs[0].text = prefix
        rPr0 = runs[0]._r.find(f'{{{A}}}rPr')
        if rPr0 is not None:
            for sf in rPr0.findall(f'{{{A}}}solidFill'): rPr0.remove(sf)
            sf = etree.Element(f'{{{A}}}solidFill')
            c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', '3A6A7A')
            rPr0.insert(0, sf)
        runs[1].text = body
        rPr1 = runs[1]._r.find(f'{{{A}}}rPr')
        if rPr1 is not None:
            rPr1.set('b', '0')
            if 'cap' in rPr1.attrib: del rPr1.attrib['cap']
            rPr1.set('lang', 'ja-JP')
            for sf in rPr1.findall(f'{{{A}}}solidFill'): rPr1.remove(sf)
            sf = etree.Element(f'{{{A}}}solidFill')
            c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', '888888')
            rPr1.insert(0, sf)
            for latin in rPr1.findall(f'{{{A}}}latin'): rPr1.remove(latin)
            for ea in rPr1.findall(f'{{{A}}}ea'): rPr1.remove(ea)
            for eff in rPr1.findall(f'{{{A}}}effectLst'): rPr1.remove(eff)
        # 3つ目以降の空runをXMLから完全削除
        p0_xml = p0._p
        all_runs = p0_xml.findall(f'{{{A}}}r')
        for r_el in all_runs[2:]:
            p0_xml.remove(r_el)
    elif len(runs) == 1:
        runs[0].text = prefix + body

def _set_s4_guide(sh, g):
    if not sh.has_text_frame: return
    txBody = sh.text_frame._txBody
    for p in txBody.findall(f'{{{A}}}p'): txBody.remove(p)
    def _para(text, color='C8D0D8', bold=False, sz=1000, indent=0):
        p = etree.SubElement(txBody, f'{{{A}}}p')
        if indent > 0:
            pPr = etree.SubElement(p, f'{{{A}}}pPr'); pPr.set('marL', str(indent)); pPr.set('indent', str(-indent))
            bc = etree.SubElement(pPr, f'{{{A}}}buClr'); c = etree.SubElement(bc, f'{{{A}}}srgbClr'); c.set('val', color)
            bs = etree.SubElement(pPr, f'{{{A}}}buSzPct'); bs.set('val', '100000')
            bu = etree.SubElement(pPr, f'{{{A}}}buChar'); bu.set('char', '•')
        r = etree.SubElement(p, f'{{{A}}}r'); rPr = etree.SubElement(r, f'{{{A}}}rPr')
        rPr.set('lang', 'ja-JP'); rPr.set('b', '1' if bold else '0'); rPr.set('sz', str(sz))
        sf = etree.SubElement(rPr, f'{{{A}}}solidFill'); c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
        t = etree.SubElement(r, f'{{{A}}}t'); t.text = text
    def _mpara(segs, sz=1000, indent=0):
        p = etree.SubElement(txBody, f'{{{A}}}p')
        if indent > 0:
            pPr = etree.SubElement(p, f'{{{A}}}pPr'); pPr.set('marL', str(indent)); pPr.set('indent', str(-indent))
            bc = etree.SubElement(pPr, f'{{{A}}}buClr'); c = etree.SubElement(bc, f'{{{A}}}srgbClr'); c.set('val', segs[0][1] if segs else 'C8D0D8')
            bs = etree.SubElement(pPr, f'{{{A}}}buSzPct'); bs.set('val', '100000')
            bu = etree.SubElement(pPr, f'{{{A}}}buChar'); bu.set('char', '•')
        for text, color, bold in segs:
            r = etree.SubElement(p, f'{{{A}}}r'); rPr = etree.SubElement(r, f'{{{A}}}rPr')
            rPr.set('lang', 'ja-JP'); rPr.set('b', '1' if bold else '0'); rPr.set('sz', str(sz))
            sf = etree.SubElement(rPr, f'{{{A}}}solidFill'); c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
            t = etree.SubElement(r, f'{{{A}}}t'); t.text = text
    def _empty(): etree.SubElement(txBody, f'{{{A}}}p')
    sz = 1000; ind = 228600
    _para('このテーブルの読み方', '3A6A7A', True, sz)
    _para(g['lam_desc'], 'C8D0D8', False, sz, ind)
    _para(g['k_desc'], 'C8D0D8', False, sz, ind)
    _mpara([(f'LTV∞（¥{g["ltv_rev"]:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。', 'C8D0D8', False)], sz, ind)
    p2 = etree.SubElement(txBody, f'{{{A}}}p')
    pPr2 = etree.SubElement(p2, f'{{{A}}}pPr'); pPr2.set('marL', str(ind)); pPr2.set('indent', '0')
    etree.SubElement(pPr2, f'{{{A}}}buNone')
    for text, color, bold in [
        ('1年時点', '56B4D3', False), ('でLTV∞の', 'C8D0D8', False),
        (f'{g["pct_1y"]:.1f}%', 'A8DADC', True), (f'（¥{g["ltv_1y"]:,.0f}）、 ', 'C8D0D8', False),
        ('2年時点', '56B4D3', False), ('で', 'C8D0D8', False),
        (f'{g["pct_2y"]:.1f}%', 'A8DADC', True), (f'（¥{g["ltv_2y"]:,.0f}）、 ', 'C8D0D8', False),
        ('3年時点', '56B4D3', False), ('で', 'C8D0D8', False),
        (f'{g["pct_3y"]:.1f}%', 'A8DADC', True), (f'（¥{g["ltv_3y"]:,.0f}）に到達します。', 'C8D0D8', False),
    ]:
        r = etree.SubElement(p2, f'{{{A}}}r'); rPr = etree.SubElement(r, f'{{{A}}}rPr')
        rPr.set('lang', 'ja-JP'); rPr.set('b', '1' if bold else '0'); rPr.set('sz', str(sz))
        sf = etree.SubElement(rPr, f'{{{A}}}solidFill'); c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
        t = etree.SubElement(r, f'{{{A}}}t'); t.text = text
    _mpara([(f'CAC上限（¥{g["cac_upper"]:,.0f}）の回収期間：売上ベース 約 ', 'C8D0D8', False),
        (g['cac_recover_rev_str'], 'A8DADC', True), (' / 粗利ベース 約 ', 'C8D0D8', False),
        (g['cac_recover_gp_str'], '56B4D3', True)], sz, ind)
    _empty()
    _mpara([('CAC設計の目安', '56B4D3', True), ('：回収期間に迷ったら、', 'C8D0D8', False),
        (f'λ={g["lam_actual_round"]:,}日（約{g["lam_years"]:.1f}年）時点の暫定LTV（粗利）¥{g["lam_gp"]:,.0f}', 'A8DADC', True),
        (f' を用いてCAC上限を算出してください。λは{g["lam_meaning"]}をデータが示した答えです。', 'C8D0D8', False)], sz)

# ── グラフ（S5: 日本語に戻す） ──
def _make_ltv_graph(t_range, rev_line, gp_line, cac_line, ltv_rev, lam_actual, x_max):
    fp = _fp()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    marker_days = sorted(set(d for d in [180, 365, 1095, 1825, round(lam_actual)] if d <= max(t_range)))
    def _fy(line, day):
        idx = min(range(len(t_range)), key=lambda i: abs(t_range[i]-day))
        return line[idx]
    ax.plot(t_range, rev_line, color='#56b4d3', lw=1.8)
    ax.plot(t_range, gp_line, color='#a8dadc', lw=1.8, ls='--')
    ax.plot(t_range, cac_line, color='#4a7a8a', lw=1.4, ls=':')
    for line, clr in [(rev_line,'#56b4d3'),(gp_line,'#a8dadc'),(cac_line,'#4a7a8a')]:
        ax.plot(marker_days, [_fy(line,d) for d in marker_days], 'o', color=clr, ms=4, zorder=5)
    ax.axhline(ltv_rev, color='#56b4d3', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(lam_actual, color='#a8dadc', lw=1.0, ls='--', alpha=0.7, ymax=0.85)
    ax.set_xlim(0, x_max+30); ax.set_ylim(0, ltv_rev*1.15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f'{int(v):,}'))
    ax.set_xticks([180,365,1095,1825])
    ax.set_xticklabels(['180日','1年','3年','5年'], fontproperties=fp, color='#888', fontsize=8)
    ax.tick_params(colors='#888', labelsize=8, length=0)
    for l in ax.get_yticklabels(): l.set_color('#888')
    ax.grid(True, alpha=0.15, color='#1a3040')
    for s in ax.spines.values(): s.set_color('#1a3040')
    ax.set_xlabel('継続期間', color='#888', fontsize=9, fontproperties=fp)
    ax.set_ylabel('金額（円）', color='#888', fontsize=9, fontproperties=fp)
    leg = ax.legend(
        [Line2D([0],[0],color='#56b4d3',lw=1.2,marker='o',ms=2),
         Line2D([0],[0],color='#a8dadc',lw=1.2,ls='--',marker='o',ms=2),
         Line2D([0],[0],color='#4a7a8a',lw=1.0,ls=':')],
        ['LTV（売上）','LTV（粗利）','CAC上限'],
        loc='upper left', frameon=False, fontsize=8, labelcolor='white', ncol=3,
        bbox_to_anchor=(0.02, 0.98), borderaxespad=0)
    for t in leg.get_texts(): t.set_fontproperties(fp)
    ax.annotate(f'λ={int(lam_actual)}日', xy=(lam_actual, ltv_rev*0.92), color='#a8dadc', fontsize=8, fontproperties=fp, ha='center', annotation_clip=False, bbox=dict(boxstyle='square,pad=0.15', facecolor=BG, edgecolor='none'))
    ax.annotate(f'LTV∞ ¥{ltv_rev:,.0f}', xy=(x_max+32, ltv_rev), color='#56b4d3', fontsize=8, fontproperties=fp, va='center', annotation_clip=False, clip_on=False)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.14)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, facecolor=BG); buf.seek(0); plt.close(fig)
    return buf

# ── S7/S13 棒グラフ：数値ラベル+X軸数値+加重平均ライン数値 ──
def _make_bar_graph(pp_rows, best, avg_ltv):
    fp = _fp()
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pp_rows)*0.45)), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    segs = [r['seg'] for r in pp_rows][::-1]
    vals = [r['ltv_r'] for r in pp_rows][::-1]
    colors = ['#56b4d3' if pp_rows[::-1][i]['seg']==best['seg'] else '#2a4a5a' for i in range(len(pp_rows))]
    # 加重平均ラインを先に描画（背面）
    ax.axvline(avg_ltv, color='#a8dadc', lw=1, ls='--', alpha=0.7, zorder=1)
    ax.text(avg_ltv, len(segs)-0.3, f'Avg ¥{avg_ltv:,.0f}', color='#a8dadc', fontsize=7, ha='center', va='bottom', fontproperties=fp, zorder=5)
    # 棒グラフを後に描画（最前面）
    bars = ax.barh(segs, vals, color=colors, height=0.6, zorder=3)
    # 棒内に数値ラベル（右寄せ）
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() - max(vals)*0.02, bar.get_y() + bar.get_height()/2,
                f'¥{val:,.0f}', va='center', ha='right', color='white', fontsize=7, fontproperties=fp, zorder=4)
    ax.tick_params(colors='#888', labelsize=8)
    for l in ax.get_yticklabels(): l.set_fontproperties(fp)
    # X軸: 数値表記（60,000形式）
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_:f'{int(v):,}'))
    for l in ax.get_xticklabels(): l.set_fontproperties(fp); l.set_fontsize(7)
    for s in ax.spines.values(): s.set_color('#1a3040')
    ax.grid(True, alpha=0.15, color='#1a3040', axis='x')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, facecolor=BG); buf.seek(0); plt.close(fig)
    return buf

# ════════════════════════════════════════════════════════════════
def generate_pptx(
    tmpl_path, k, lam, lam_actual, r2, arpu_daily, gpm, gp_daily, cac_n, cac_upper,
    ltv_rev, lam_rev, lam_gp, rev_99, gp_99, days_99,
    t_range, rev_line, gp_line, cac_line, x_max,
    tbl_rows, horizons, ltv_offset_days,
    ltv_horizon_offset, ltv_horizon_spot, business_type, dormancy_days,
    buf1, buf2, df, client_name, analyst_name, billing_cycle_display,
    k_summary, r2_summary, cac_label, cac_recover_rev_str, cac_recover_gp_str,
    segment_cols_input, _compute_km_df, _fit_weibull_df, ltv_inf, fmt_horizon,
    s4_guide_data=None,
    report_title=None,
    arpu_0_dorm=None,
    arpu_long=None,
    outlier_label='除外なし',
):
    if k<0.7: ki=f"k={k:.3f}（強い初期集中型）: 利用開始直後の体験品質が生死を分ける構造。30日以内の離脱防止施策が最重要。"
    elif k<1.0: ki=f"k={k:.3f}（緩やかな初期集中型）: 離脱率は一定に近いが初期にやや多め。オンボーディング改善とリテンション施策を並行実施。"
    elif k<1.5: ki=f"k={k:.3f}（逓増型・中程度）: 継続期間が長いほど離脱リスクが増す。1年超の顧客へのエンゲージメント強化が鍵。"
    else: ki=f"k={k:.3f}（強い逓増型）: 長期顧客ほど急速に離脱。VIP施策・継続特典による長期繋ぎ止めが急務。"
    if r2>=0.95: rc=f"R²={r2:.3f}: 非常に高精度。LTV∞推定値の信頼性は高い。"
    elif r2>=0.85: rc=f"R²={r2:.3f}: 許容範囲内。推定値に±15%程度の幅を見込んで意思決定を。"
    else: rc=f"R²={r2:.3f}: やや低め。データ件数不足または複数の離脱パターンが混在している可能性あり。"
    prs = Presentation(tmpl_path)
    pd_mod = __import__('pandas')
    _dc = [df['start_date']]
    if 'end_date' in df.columns and df['end_date'].notna().any(): _dc.append(df['end_date'])
    _ds = pd_mod.concat(_dc).dropna().min().strftime('%Y/%m/%d')
    _de = pd_mod.concat(_dc).dropna().max().strftime('%Y/%m/%d')
    for sh in prs.slides[0].shapes:
        if sh.name=='TextBox 4': _set_text(sh, report_title or 'Kaplan–Meier × Weibull Model')
        elif sh.name=='TextBox 5': _set_text(sh, client_name or '')
        elif sh.name=='TextBox 6': _set_text(sh, date.today().strftime('%Y年%m月%d日'))
        elif sh.name=='TextBox 7': _set_text(sh, analyst_name or '')
    s2=prs.slides[1]
    for sh in s2.shapes:
        if sh.name=='テキスト プレースホルダー 6' and sh.has_text_frame:
            tf=sh.text_frame
            # 1行目：データ期間 | 顧客数 | 解約済み | 継続中 | 異常値の処理
            i1=f"データ期間: {_ds} – {_de}　|　顧客数: {len(df):,}件　|　解約済み: {df['event'].sum():,}件　|　継続中: {(df['event']==0).sum():,}件　|　異常値の処理：{outlier_label}"
            # 2行目：ビジネスタイプ | 請求サイクル/休眠判定 | 日割り(サブスクのみ) | Daily ARPU | GPM
            if business_type == '都度購入型':
                _dorm_disp = f'{dormancy_days}日' if dormancy_days else '180日'
                i2=f"{business_type}　|　休眠判定: {_dorm_disp}　|　Daily ARPU: ¥{arpu_daily:,.2f}　|　GPM: {gpm:.0%}"
            else:
                i2=f"{business_type}　|　{billing_cycle_display}　|　解約時の日割り計算：{'ON' if ltv_offset_days==0 else 'OFF'}　|　Daily ARPU: ¥{arpu_daily:,.2f}　|　GPM: {gpm:.0%}"
            if len(tf.paragraphs)>=1:
                for r in tf.paragraphs[0].runs: r.text=''
                if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text=i1
            if len(tf.paragraphs)>=2:
                for r in tf.paragraphs[1].runs: r.text=''
                if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text=i2
        elif sh.name=='グループ化 26':
            kpi={'TextBox 6':f'¥{ltv_rev:,.0f}','TextBox 10':f'¥{cac_upper:,.0f}','TextBox 11':f'CAC上限（{cac_label}）','TextBox 14':f'{k:.3f}','TextBox 18':f'{lam_actual:.0f}日','TextBox 22':f'{r2:.3f}'}
            for g in sh.shapes:
                if g.name in kpi: _set_text(g, kpi[g.name])
        elif sh.name=='テキスト ボックス 49' and sh.has_text_frame:
            tf=sh.text_frame
            if len(tf.paragraphs)>=2:
                for r in tf.paragraphs[0].runs: r.text=''
                if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text='結論'
                for r in tf.paragraphs[1].runs: r.text=''
                if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text=k_summary+r2_summary
    s3=prs.slides[2]; ld=lam+ltv_offset_days if business_type=='都度購入型' else lam
    for sh in s3.shapes:
        if sh.name=='Picture 3': buf1.seek(0); _replace_image(s3,sh,buf1)
        elif sh.name=='Picture 4': buf2.seek(0); _replace_image(s3,sh,buf2)
        elif sh.name=='TextBox 7': _set_text(sh, f"k（形状パラメータ） = {k:.3f}\n→ 左グラフ（Survival Curve）の曲線の急峻さを決める値。k=1で指数分布（一定離脱率）\n→ {ki}")
        elif sh.name=='TextBox 8': _set_text(sh, f"λ（尺度パラメータ） = {ld:.1f}日（約{ld/365:.1f}年）\n→ 大きいほどLTV∞到達が長期化する。λ日時点での暫定LTV到達率はk値により異なる（k=1のとき63.2%）\n→ {rc}")
    s4=prs.slides[3]; rows_s4=[]
    for h in horizons:
        if business_type=='都度購入型':
            dp=dormancy_days or 180
            _a0 = arpu_0_dorm if arpu_0_dorm is not None else arpu_daily
            _al = arpu_long if arpu_long is not None else arpu_daily
            lr=ltv_horizon_spot(k,lam,_a0,_al,h,dp); lg=ltv_horizon_spot(k,lam,_a0*gpm,_al*gpm,h,dp)
        else:
            lr=ltv_horizon_offset(k,lam,arpu_daily,h,ltv_offset_days); lg=ltv_horizon_offset(k,lam,gp_daily,h,ltv_offset_days)
        label=f'{h}日' if h<365 else f'{h//365}年（{h:,}日）'
        rows_s4.append([label,f'¥{lr:,.0f}',f'¥{lg:,.0f}',f'¥{lg/cac_n:,.0f}',f'{lr/ltv_rev*100:.1f}%'])
    rows_s4.append([f'λ {round(lam_actual):,}日',f'¥{lam_rev:,.0f}',f'¥{lam_gp:,.0f}',f'¥{lam_gp/cac_n:,.0f}',f'{lam_rev/ltv_rev*100:.1f}%'])
    rows_s4.append([f'LTV∞到達率: 99%（{int(days_99):,}日）',f'¥{rev_99:,.0f}',f'¥{gp_99:,.0f}',f'¥{gp_99/cac_n:,.0f}','99.0%'])
    for sh in s4.shapes:
        if sh.shape_type==19: _write_table(sh.table,None,rows_s4[:-1],rows_s4[-1])
        elif sh.name=='テキスト ボックス 3' and sh.has_text_frame:
            sh.top=sh.top+182880
            if s4_guide_data: _set_s4_guide(sh, s4_guide_data)
    buf_s5=_make_ltv_graph(t_range,rev_line,gp_line,cac_line,ltv_rev,lam_actual,x_max)
    for sh in prs.slides[4].shapes:
        if sh.name=='コンテンツ プレースホルダー 7':
            sh.left=int(0.5*914400); sh.top=int(1.6*914400); sh.width=int(12.2*914400); sh.height=int(4.3*914400)
            _replace_image(prs.slides[4],sh,buf_s5)
    if not segment_cols_input.strip():
        sldIdLst=prs.slides._sldIdLst
        for idx in [9,8,7,6,5]:
            if idx<len(sldIdLst):
                e=sldIdLst[idx]; rId=e.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if rId:
                    try: prs.part.drop_rel(rId)
                    except: pass
                sldIdLst.remove(e)
    else:
        seg_cols=[c.strip() for c in segment_cols_input.split(',') if c.strip() and c.strip() in df.columns]
        for sh in prs.slides[5].shapes:
            if sh.name=='TextBox 4': _set_text(sh,'  |  '.join(seg_cols))
        for sc in seg_cols:
            pp_rows=[]
            for sv in sorted(df[sc].dropna().unique()):
                df_s=df[df[sc]==sv]
                if len(df_s)<10 or df_s['event'].sum()<5: continue
                try:
                    km_s=_compute_km_df(df_s); k_s,lam_s,r2_s,_=_fit_weibull_df(km_s)
                    if k_s is None: continue
                    arpu_s=df_s['arpu_daily'].mean(); gp_s=arpu_s*gpm
                    ltv_r,_=ltv_inf(k_s,lam_s,arpu_s); ltv_g,_=ltv_inf(k_s,lam_s,gp_s)
                    pp_rows.append({'seg':str(sv),'n':len(df_s),'ltv_r':ltv_r,'ltv_g':ltv_g,'cac':ltv_g/cac_n,'k':k_s,'lam':lam_s,'r2':r2_s})
                except: continue
            if not pp_rows: continue
            pp_rows.sort(key=lambda x:x['ltv_r'],reverse=True)
            best=pp_rows[0]; n_total=sum(r['n'] for r in pp_rows)
            avg_ltv=sum(r['ltv_r']*r['n'] for r in pp_rows)/n_total
            w_ltv_g=sum(r['ltv_g']*r['n'] for r in pp_rows)/n_total
            w_cac=sum(r['cac']*r['n'] for r in pp_rows)/n_total
            avg_cac=sum(r['cac']*r['n'] for r in pp_rows)/n_total
            premium=(best['ltv_r']-avg_ltv)/avg_ltv*100
            cac_diff=best['cac']-avg_cac
            cac_diff_str=f"+¥{cac_diff:,.0f}高く設定可能" if cac_diff>=0 else f"¥{abs(cac_diff):,.0f}低め"
            s7=_copy_slide(prs,6)
            for sh in s7.shapes:
                if sh.name=='タイトル 4': _set_text(sh,f'{sc}: LTV∞')
                elif sh.name=='テキスト プレースホルダー 5' and sh.has_text_frame:
                    tf=sh.text_frame
                    if len(tf.paragraphs)>=2:
                        for r in tf.paragraphs[0].runs: r.text=''
                        if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text=f'TOP PICK　{best["seg"]}'
                        for r in tf.paragraphs[1].runs: r.text=''
                        if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text=f'LTV∞(売上): ¥{best["ltv_r"]:,.0f}（全セグメント平均比 +{premium:.1f}%）　|　許容CAC上限 ¥{best["cac"]:,.0f}（全セグメント平均より{cac_diff_str}）'
                elif sh.name=='コンテンツ プレースホルダー 8':
                    buf7=_make_bar_graph(pp_rows[:10],best,avg_ltv); _replace_image(s7,sh,buf7)
            s8=_copy_slide(prs,7); top10=pp_rows[:10]
            dr8=[[r['seg'],f'{r["n"]:,}',f'¥{r["ltv_r"]:,.0f}',f'¥{r["ltv_g"]:,.0f}',f'¥{r["cac"]:,.0f}',f'{r["k"]:.3f}',f'{r["lam"]:.1f}',f'{r["r2"]:.3f}'] for r in top10]
            ft8=['加重平均',f'{n_total:,}',f'¥{avg_ltv:,.0f}',f'¥{w_ltv_g:,.0f}',f'¥{w_cac:,.0f}','—','—','—']
            for sh in s8.shapes:
                if sh.name=='タイトル 2': _set_text(sh,f'{sc}: 分析結果のサマリー')
                elif sh.shape_type==19: _write_table_styled(sh.table,None,dr8,ft8)
                elif sh.name=='テキスト ボックス 9' and sh.has_text_frame:
                    diff_pct=(avg_ltv-ltv_rev)/ltv_rev*100
                    note=f'NOTE — テーブルは最大上位10項目を表示。加重平均行は全{len(pp_rows)}項目を顧客数で重み付けした値です。全体LTV∞との差（{diff_pct:+.1f}%）は統計的に正常な現象です。詳細スライドには全セグメントを掲載しています。'
                    _set_note_text(sh,note)
            for ri,row in enumerate(pp_rows):
                tmpl_idx=8 if ri==0 else 9
                sx=_copy_slide(prs,tmpl_idx)
                buf_km=buf_wb=None
                try:
                    df_s=df[df[sc]==row['seg']]; km_s=_compute_km_df(df_s); arpu_s=df_s['arpu_daily'].mean()
                    fig_km,ax_km=plt.subplots(figsize=(4,2.8),dpi=120); fig_km.patch.set_facecolor(BG); ax_km.set_facecolor(BG)
                    ax_km.step(km_s['time'],km_s['survival'],where='post',color='#56b4d3',lw=1.5)
                    t_fit=list(range(int(km_s['time'].max()))); s_fit=[math.exp(-(t/row['lam'])**row['k']) for t in t_fit]
                    ax_km.plot(t_fit,s_fit,color='#a8dadc',lw=1.2,ls='--')
                    ax_km.tick_params(colors='#888',labelsize=7)
                    for s in ax_km.spines.values(): s.set_color('#1a3040')
                    ax_km.grid(True,alpha=0.15,color='#1a3040'); fig_km.tight_layout()
                    buf_km=io.BytesIO(); fig_km.savefig(buf_km,format='png',dpi=120,facecolor=BG); buf_km.seek(0); plt.close(fig_km)
                    x_s=list(range(1,int(max(1825,row['lam']*2)),max(1,int(row['lam']*2)//200)))
                    y_s=[ltv_horizon_offset(row['k'],row['lam'],arpu_s,t,ltv_offset_days) for t in x_s]
                    fig_wb,ax_wb=plt.subplots(figsize=(4,2.8),dpi=120); fig_wb.patch.set_facecolor(BG); ax_wb.set_facecolor(BG)
                    ax_wb.plot(x_s,y_s,color='#56b4d3',lw=1.5)
                    ax_wb.axvline(row['lam'],color='#a8dadc',lw=1,ls='--',alpha=0.7)
                    ax_wb.axhline(row['ltv_r'],color='#56b4d3',lw=0.8,ls=':',alpha=0.5)
                    ax_wb.tick_params(colors='#888',labelsize=7)
                    for s in ax_wb.spines.values(): s.set_color('#1a3040')
                    ax_wb.grid(True,alpha=0.15,color='#1a3040'); fig_wb.tight_layout()
                    buf_wb=io.BytesIO(); fig_wb.savefig(buf_wb,format='png',dpi=120,facecolor=BG); buf_wb.seek(0); plt.close(fig_wb)
                except: pass
                rows_sx=[]
                for h in horizons:
                    lr_s=ltv_horizon_offset(row['k'],row['lam'],df[df[sc]==row['seg']]['arpu_daily'].mean(),h,ltv_offset_days)
                    lg_s=lr_s*gpm
                    rows_sx.append([fmt_horizon(h),f'¥{lr_s:,.0f}',f'¥{lg_s:,.0f}',f'¥{lg_s/cac_n:,.0f}',f'{lr_s/row["ltv_r"]*100:.1f}%'])
                lam_r_s=ltv_horizon_offset(row['k'],row['lam'],df[df[sc]==row['seg']]['arpu_daily'].mean(),row['lam'],ltv_offset_days)
                rows_sx.append([f'λ {round(row["lam"]):,}日',f'¥{lam_r_s:,.0f}',f'¥{lam_r_s*gpm:,.0f}',f'¥{lam_r_s*gpm/cac_n:,.0f}',f'{lam_r_s/row["ltv_r"]*100:.1f}%'])
                try:
                    from scipy.optimize import brentq as _bq
                    _arpu=df[df[sc]==row['seg']]['arpu_daily'].mean()
                    d99s=_bq(lambda h:ltv_horizon_offset(row['k'],row['lam'],_arpu,h,ltv_offset_days)/row['ltv_r']-0.99,1,100000)
                    r99s=ltv_horizon_offset(row['k'],row['lam'],_arpu,d99s,ltv_offset_days)
                    rows_sx.append([f'LTV∞到達率: 99%（{int(d99s):,}日）',f'¥{r99s:,.0f}',f'¥{r99s*gpm:,.0f}',f'¥{r99s*gpm/cac_n:,.0f}','99.0%'])
                except: pass
                for sh in sx.shapes:
                    if sh.name=='タイトル 8': _set_text(sh,f'{sc}: {row["seg"]}')
                    elif sh.name=='Picture 6' and buf_km: buf_km.seek(0); _replace_image(sx,sh,buf_km)
                    elif sh.name=='Picture 7' and buf_wb: buf_wb.seek(0); _replace_image(sx,sh,buf_wb)
                    elif sh.shape_type==19: _write_table(sh.table,None,rows_sx[:-1],rows_sx[-1])
                    elif sh.name=='テキスト ボックス 17' and sh.has_text_frame:
                        _set_text(sh,'TOP PICK　' if ri==0 else '')
                        bp=sh.text_frame._txBody.find(f'{{{A}}}bodyPr')
                        if bp is None: bp=etree.SubElement(sh.text_frame._txBody,f'{{{A}}}bodyPr')
                        bp.set('anchor','ctr')
        sldIdLst=prs.slides._sldIdLst
        for idx in [9,8,7,6]:
            if idx<len(sldIdLst):
                e=sldIdLst[idx]; rId=e.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if rId:
                    try: prs.part.drop_rel(rId)
                    except: pass
                sldIdLst.remove(e)
    buf_out=io.BytesIO(); prs.save(buf_out); buf_out.seek(0)
    return buf_out
