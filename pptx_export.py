"""
PPTX Export Module v237
"""
import io, copy, os
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

# ── 日本語フォント（項目名用のみ） ──
_JP_FP = None
def _init_jp():
    global _JP_FP
    for p in ['/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
              '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(p):
            fm.fontManager.addfont(p); _JP_FP = fm.FontProperties(fname=p); return
    for p in fm.findSystemFonts():
        if any(k in p.lower() for k in ['cjk','ipag','japanese','gothic']):
            fm.fontManager.addfont(p); _JP_FP = fm.FontProperties(fname=p); return
_init_jp()
BG = '#111820'
def _fp(): return _JP_FP or fm.FontProperties()

A = 'http://schemas.openxmlformats.org/drawingml/2006/main'

# ── ヘルパー ──
def _set_text(shape, text):
    if not shape.has_text_frame: return
    tf = shape.text_frame
    for p in tf.paragraphs:
        for r in p.runs: r.text = ''
    if tf.paragraphs and tf.paragraphs[0].runs:
        tf.paragraphs[0].runs[0].text = text

def _replace_image(slide, shape, buf):
    blip = shape._element.find('.//' + qn('a:blip'))
    if blip is None: return
    buf.seek(0)
    img_part, rId = slide.part.get_or_add_image_part(buf)
    blip.set(qn('r:embed'), rId)

def _write_table(tbl, header, rows, footer=None):
    data = rows + ([footer] if footer else [])
    needed = 1 + len(data)
    tr = copy.deepcopy(tbl.rows[1]._tr) if len(tbl.rows) > 1 else None
    while len(tbl.rows) < needed and tr: tbl._tbl.append(copy.deepcopy(tr))
    while len(tbl.rows) > needed: tbl._tbl.remove(tbl.rows[-1]._tr)
    for ri, rd in enumerate(data):
        row = tbl.rows[ri+1]
        for ci, val in enumerate(rd):
            if ci >= len(row.cells): break
            tf = row.cells[ci].text_frame
            for p in tf.paragraphs:
                for r in p.runs: r.text = ''
            if tf.paragraphs and tf.paragraphs[0].runs:
                tf.paragraphs[0].runs[0].text = str(val)

def _write_table_styled(tbl, header, rows, footer=None):
    _write_table(tbl, header, rows, footer)
    data = rows + ([footer] if footer else [])
    for ri in range(len(data)):
        row = tbl.rows[ri+1]
        is_footer = footer and ri == len(data)-1
        bg = '142030' if is_footer else ('0D1520' if ri%2==0 else '0A1018')
        # フォント色: 加重平均行はA8DADC、通常はC8D0D8
        font_color = 'A8DADC' if is_footer else 'C8D0D8'
        for ci in range(len(row.cells)):
            cell = row.cells[ci]
            # 背景色
            tcPr = cell._tc.get_or_add_tcPr()
            for sf in tcPr.findall(f'{{{A}}}solidFill'): tcPr.remove(sf)
            sf = etree.SubElement(tcPr, f'{{{A}}}solidFill')
            c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', bg)
            # フォント色
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    rPr = r._r.find(f'{{{A}}}rPr')
                    if rPr is None: rPr = etree.SubElement(r._r, f'{{{A}}}rPr', nsmap={None: A})
                    for sf2 in rPr.findall(f'{{{A}}}solidFill'): rPr.remove(sf2)
                    sf2 = etree.SubElement(rPr, f'{{{A}}}solidFill')
                    c2 = etree.SubElement(sf2, f'{{{A}}}srgbClr'); c2.set('val', font_color)

def _copy_slide(prs, idx):
    src = prs.slides[idx]
    new = prs.slides.add_slide(src.slide_layout)
    for el in list(new.shapes._spTree): new.shapes._spTree.remove(el)
    for el in list(src.shapes._spTree): new.shapes._spTree.append(copy.deepcopy(el))
    for rId, rel in src.part.rels.items():
        if rel.reltype == RT.IMAGE: new.part.rels._rels[rId] = rel
    return new

def _set_note_text(shape, note_text):
    """NOTE = 青, — 以降 = グレー。テンプレートのrPrを保持しつつ新runを作成"""
    if not shape.has_text_frame: return
    tf = shape.text_frame
    # テンプレートの書式を取得
    tmpl_rPr = None
    if tf.paragraphs and tf.paragraphs[0].runs:
        orig = tf.paragraphs[0].runs[0]._r.find(f'{{{A}}}rPr')
        if orig is not None: tmpl_rPr = copy.deepcopy(orig)

    # 全runクリア
    for p in tf.paragraphs:
        for r in p.runs: r.text = ''

    # 最初のrunにNOTE部、2番目以降に本文を入れる
    if ' — ' in note_text:
        prefix, body = note_text.split(' — ', 1)
        body = ' — ' + body
    else:
        prefix, body = note_text, ''

    # 既存runを活用（最初のparagraphに書く）
    p0 = tf.paragraphs[0]
    # 既存runをXMLから削除
    p0_xml = p0._p
    for r_el in p0_xml.findall(f'{{{A}}}r'): p0_xml.remove(r_el)

    def _add_run(para_xml, text, color):
        r = etree.SubElement(para_xml, f'{{{A}}}r')
        if tmpl_rPr is not None:
            rPr = copy.deepcopy(tmpl_rPr)
        else:
            rPr = etree.SubElement(r, f'{{{A}}}rPr')
            rPr.set('lang', 'ja-JP'); rPr.set('b', '1'); rPr.set('sz', '1000')
        # 色を上書き
        for sf in rPr.findall(f'{{{A}}}solidFill'): rPr.remove(sf)
        sf = etree.SubElement(rPr, f'{{{A}}}solidFill')
        c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
        r.insert(0, rPr)
        t = etree.SubElement(r, f'{{{A}}}t'); t.text = text

    _add_run(p0_xml, prefix, '56B4D3')
    if body: _add_run(p0_xml, body, '3A6A7A')

def _set_s4_guide(shape, g):
    if not shape.has_text_frame: return
    txBody = shape.text_frame._txBody
    for p in txBody.findall(f'{{{A}}}p'): txBody.remove(p)

    def _para(text, color='C8D0D8', bold=False, sz=1000, indent=0):
        p = etree.SubElement(txBody, f'{{{A}}}p')
        if indent > 0:
            pPr = etree.SubElement(p, f'{{{A}}}pPr')
            pPr.set('marL', str(indent))  # EMU
            pPr.set('indent', str(-indent))
            # 箇条書き設定（buChar）
            buClr = etree.SubElement(pPr, f'{{{A}}}buClr')
            c = etree.SubElement(buClr, f'{{{A}}}srgbClr'); c.set('val', color)
            buSz = etree.SubElement(pPr, f'{{{A}}}buSzPct'); buSz.set('val', '100000')
            buChar = etree.SubElement(pPr, f'{{{A}}}buChar'); buChar.set('char', '•')
        r = etree.SubElement(p, f'{{{A}}}r')
        rPr = etree.SubElement(r, f'{{{A}}}rPr')
        rPr.set('lang','ja-JP'); rPr.set('b','1' if bold else '0'); rPr.set('sz',str(sz))
        sf = etree.SubElement(rPr, f'{{{A}}}solidFill')
        c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
        t = etree.SubElement(r, f'{{{A}}}t'); t.text = text

    def _mpara(segs, sz=1000, indent=0):
        p = etree.SubElement(txBody, f'{{{A}}}p')
        if indent > 0:
            pPr = etree.SubElement(p, f'{{{A}}}pPr')
            pPr.set('marL', str(indent)); pPr.set('indent', str(-indent))
            buClr = etree.SubElement(pPr, f'{{{A}}}buClr')
            c = etree.SubElement(buClr, f'{{{A}}}srgbClr'); c.set('val', segs[0][1] if segs else 'C8D0D8')
            buSz = etree.SubElement(pPr, f'{{{A}}}buSzPct'); buSz.set('val', '100000')
            buChar = etree.SubElement(pPr, f'{{{A}}}buChar'); buChar.set('char', '•')
        for text, color, bold in segs:
            r = etree.SubElement(p, f'{{{A}}}r')
            rPr = etree.SubElement(r, f'{{{A}}}rPr')
            rPr.set('lang','ja-JP'); rPr.set('b','1' if bold else '0'); rPr.set('sz',str(sz))
            sf = etree.SubElement(rPr, f'{{{A}}}solidFill')
            c = etree.SubElement(sf, f'{{{A}}}srgbClr'); c.set('val', color)
            t = etree.SubElement(r, f'{{{A}}}t'); t.text = text

    def _empty_para():
        etree.SubElement(txBody, f'{{{A}}}p')

    sz = 1000; ind = 228600  # 0.25 inch indent
    _para('このテーブルの読み方', '3A6A7A', True, sz)
    _para(g['lam_desc'], 'C8D0D8', False, sz, ind)
    _para(g['k_desc'], 'C8D0D8', False, sz, ind)
    _mpara([
        (f'LTV\u221e（\u00a5{g["ltv_rev"]:,.0f}）は理論上の上限値で、実際にはこの金額に向かって時間をかけて積み上がります。\n', 'C8D0D8', False),
        ('1年時点', '56B4D3', False), (f'でLTV\u221eの', 'C8D0D8', False),
        (f'{g["pct_1y"]:.1f}%', 'A8DADC', True), (f'（\u00a5{g["ltv_1y"]:,.0f}）、 ', 'C8D0D8', False),
        ('2年時点', '56B4D3', False), ('で', 'C8D0D8', False),
        (f'{g["pct_2y"]:.1f}%', 'A8DADC', True), (f'（\u00a5{g["ltv_2y"]:,.0f}）、 ', 'C8D0D8', False),
        ('3年時点', '56B4D3', False), ('で', 'C8D0D8', False),
        (f'{g["pct_3y"]:.1f}%', 'A8DADC', True), (f'（\u00a5{g["ltv_3y"]:,.0f}）に到達します。', 'C8D0D8', False),
    ], sz, ind)
    _mpara([
        (f'CAC上限（\u00a5{g["cac_upper"]:,.0f}）の回収期間：売上ベース 約 ', 'C8D0D8', False),
        (g['cac_recover_rev_str'], 'A8DADC', True),
        (' / 粗利ベース 約 ', 'C8D0D8', False),
        (g['cac_recover_gp_str'], '56B4D3', True),
    ], sz, ind)
    _empty_para()
    _mpara([
        ('CAC設計の目安', '56B4D3', True),
        ('：回収期間に迷ったら、', 'C8D0D8', False),
        (f'\u03bb={g["lam_actual_round"]:,}日（約{g["lam_years"]:.1f}年）時点の暫定LTV（粗利）\u00a5{g["lam_gp"]:,.0f}', 'A8DADC', True),
        (f' を用いてCAC上限を算出してください。\u03bbは{g["lam_meaning"]}をデータが示した答えです。', 'C8D0D8', False),
    ], sz)


# ── グラフ（全て英数記号、日本語なし） ──
def _make_ltv_graph(t_range, rev_line, gp_line, cac_line, ltv_rev, lam_actual, x_max):
    fig, ax = plt.subplots(figsize=(898/150, 280/150), dpi=150)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(t_range, rev_line, color='#56b4d3', lw=1.8)
    ax.plot(t_range, gp_line, color='#a8dadc', lw=1.8, ls='--')
    ax.plot(t_range, cac_line, color='#4a7a8a', lw=1.4, ls=':')
    ax.axhline(ltv_rev, color='#56b4d3', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(lam_actual, color='#a8dadc', lw=1.0, ls='--', alpha=0.7)
    ax.set_xlim(0, x_max+30); ax.set_ylim(0, ltv_rev*1.08)
    # 縦軸: カンマ付き数値
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{int(v):,}'))
    # 横軸: 英語ラベル
    ax.set_xticks([180,365,730,1095,1460,1825])
    ax.set_xticklabels(['180d','1yr','2yr','3yr','4yr','5yr'], color='#888', fontsize=7)
    ax.tick_params(colors='#888', labelsize=7, length=0)
    for l in ax.get_yticklabels(): l.set_color('#888')
    ax.grid(True, alpha=0.15, color='#1a3040')
    for s in ax.spines.values(): s.set_color('#1a3040')
    ax.set_xlabel('Duration', color='#888', fontsize=8)
    ax.set_ylabel('Amount (JPY)', color='#888', fontsize=8)
    ax.legend(
        [Line2D([0],[0],color='#56b4d3',lw=1.8), Line2D([0],[0],color='#a8dadc',lw=1.8,ls='--'), Line2D([0],[0],color='#4a7a8a',lw=1.4,ls=':')],
        ['LTV (Revenue)','LTV (GP)','CAC Limit'],
        loc='upper left', frameon=False, fontsize=7, labelcolor='white', ncol=3, bbox_to_anchor=(0.0,1.18))
    ax.annotate(f'lambda={int(lam_actual)}d', xy=(lam_actual, ltv_rev*0.92), color='white', fontsize=7, annotation_clip=False)
    ax.annotate(f'LTV-inf {ltv_rev:,.0f}', xy=(x_max+32, ltv_rev), color='#56b4d3', fontsize=7, va='center', annotation_clip=False, clip_on=False)
    fig.subplots_adjust(left=0.12, right=0.84, top=0.78, bottom=0.20)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150, facecolor=BG, bbox_inches='tight'); buf.seek(0); plt.close(fig)
    return buf

def _make_bar_graph(pp_rows, best, avg_ltv):
    """横棒グラフ（項目名は日本語対応）"""
    fp = _fp()
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pp_rows)*0.4)), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    segs = [r['seg'] for r in pp_rows]; vals = [r['ltv_r'] for r in pp_rows]
    colors = ['#56b4d3' if r['seg']==best['seg'] else '#2a4a5a' for r in pp_rows]
    ax.barh(segs[::-1], vals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(avg_ltv, color='#a8dadc', lw=1, ls='--', alpha=0.7)
    ax.tick_params(colors='#888', labelsize=8)
    for l in ax.get_yticklabels(): l.set_fontproperties(fp)  # 項目名は日本語
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v/10000:.0f}'))
    ax.set_xlabel('LTV-inf (x10,000 JPY)', color='#888', fontsize=8)
    for s in ax.spines.values(): s.set_color('#1a3040')
    ax.grid(True, alpha=0.15, color='#1a3040', axis='x')
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120, facecolor=BG); buf.seek(0); plt.close(fig)
    return buf


# ════════════════════════════════════════════════════════════════
def generate_pptx(
    tmpl_path,
    k, lam, lam_actual, r2, arpu_daily, gpm, gp_daily, cac_n, cac_upper,
    ltv_rev, lam_rev, lam_gp, rev_99, gp_99, days_99,
    t_range, rev_line, gp_line, cac_line, x_max,
    tbl_rows, horizons, ltv_offset_days,
    ltv_horizon_offset, ltv_horizon_spot, business_type, dormancy_days,
    buf1, buf2,
    df, client_name, analyst_name, billing_cycle_display,
    k_summary, r2_summary, cac_label,
    cac_recover_rev_str, cac_recover_gp_str,
    segment_cols_input,
    _compute_km_df, _fit_weibull_df, ltv_inf, fmt_horizon,
    s4_guide_data=None,
):
    if k<0.7:    ki=f"k={k:.3f}: Strong early churn. 30-day retention is critical."
    elif k<1.0:  ki=f"k={k:.3f}: Mild early churn. Improve onboarding + retention."
    elif k<1.5:  ki=f"k={k:.3f}: Increasing hazard. Focus on 1yr+ customer engagement."
    else:        ki=f"k={k:.3f}: Strong increasing hazard. VIP retention programs needed."
    if r2>=0.95:   rc=f"R2={r2:.3f}: High accuracy."
    elif r2>=0.85: rc=f"R2={r2:.3f}: Acceptable. Allow +/-15% margin."
    else:          rc=f"R2={r2:.3f}: Low fit. Check data quality."

    prs = Presentation(tmpl_path)
    _dc = [df['start_date']]
    if 'end_date' in df.columns and df['end_date'].notna().any(): _dc.append(df['end_date'])
    pd = __import__('pandas')
    _ds = pd.concat(_dc).dropna().min().strftime('%Y/%m/%d')
    _de = pd.concat(_dc).dropna().max().strftime('%Y/%m/%d')

    # S1
    for sh in prs.slides[0].shapes:
        if sh.name=='TextBox 5': _set_text(sh, client_name or '')
        elif sh.name=='TextBox 6': _set_text(sh, date.today().strftime('%Y年%m月%d日'))
        elif sh.name=='TextBox 7': _set_text(sh, analyst_name or '')

    # S2
    s2=prs.slides[1]
    for sh in s2.shapes:
        if sh.name=='テキスト プレースホルダー 6' and sh.has_text_frame:
            tf=sh.text_frame
            i1=f"データ期間: {_ds} – {_de}　|　顧客数: {len(df):,}件　|　解約済み: {df['event'].sum():,}件　|　継続中: {(df['event']==0).sum():,}件　|　平均日次 ARPU: ¥{arpu_daily:,.2f}　|　GPM: {gpm:.0%}"
            i2=f"異常値の処理：除外なし　|　{business_type}　|　{billing_cycle_display}　|　解約時の日割り計算：{'ON' if ltv_offset_days==0 else 'OFF'}"
            if len(tf.paragraphs)>=1:
                for r in tf.paragraphs[0].runs: r.text=''
                if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text=i1
            if len(tf.paragraphs)>=2:
                for r in tf.paragraphs[1].runs: r.text=''
                if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text=i2
        elif sh.name=='グループ化 26':
            kpi={'TextBox 6':f'¥{ltv_rev:,.0f}','TextBox 10':f'¥{cac_upper:,.0f}','TextBox 11':f'CAC上限（{cac_label}）','TextBox 14':f'{k:.3f}','TextBox 18':f'{lam_actual:.0f}日','TextBox 22':f'{r2:.3f}'}
            for g in sh.shapes:
                if g.name in kpi: _set_text(g,kpi[g.name])
        elif sh.name=='テキスト ボックス 49' and sh.has_text_frame:
            tf=sh.text_frame
            if len(tf.paragraphs)>=2:
                for r in tf.paragraphs[0].runs: r.text=''
                if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text='結論'
                for r in tf.paragraphs[1].runs: r.text=''
                if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text=k_summary+r2_summary

    # S3
    s3=prs.slides[2]; ld=lam+ltv_offset_days if business_type=='都度購入型' else lam
    for sh in s3.shapes:
        if sh.name=='Picture 3': buf1.seek(0); _replace_image(s3,sh,buf1)
        elif sh.name=='Picture 4': buf2.seek(0); _replace_image(s3,sh,buf2)
        elif sh.name=='TextBox 7': _set_text(sh, f"k = {k:.3f}\n→ {ki}")
        elif sh.name=='TextBox 8': _set_text(sh, f"lambda = {ld:.1f}d (~{ld/365:.1f}yr)\n→ {rc}")

    # S4
    s4=prs.slides[3]; rows_s4=[]
    for h in horizons:
        if business_type=='都度購入型':
            dp=dormancy_days or 180; lr=ltv_horizon_spot(k,lam,arpu_daily,h,dp); lg=ltv_horizon_spot(k,lam,arpu_daily*gpm,h,dp)
        else:
            lr=ltv_horizon_offset(k,lam,arpu_daily,h,ltv_offset_days); lg=ltv_horizon_offset(k,lam,gp_daily,h,ltv_offset_days)
        label=f'{h}日' if h<365 else f'{h//365}年（{h:,}日）'
        rows_s4.append([label,f'¥{lr:,.0f}',f'¥{lg:,.0f}',f'¥{lg/cac_n:,.0f}',f'{lr/ltv_rev*100:.1f}%'])
    rows_s4.append([f'λ {round(lam_actual):,}日',f'¥{lam_rev:,.0f}',f'¥{lam_gp:,.0f}',f'¥{lam_gp/cac_n:,.0f}',f'{lam_rev/ltv_rev*100:.1f}%'])
    rows_s4.append([f'LTV∞到達率: 99%（{int(days_99):,}日）',f'¥{rev_99:,.0f}',f'¥{gp_99:,.0f}',f'¥{gp_99/cac_n:,.0f}','99.0%'])
    for sh in s4.shapes:
        if sh.shape_type==19: _write_table(sh.table,None,rows_s4[:-1],rows_s4[-1])
        elif sh.name=='テキスト ボックス 3' and sh.has_text_frame:
            # 1行分下にオフセット
            sh.top = sh.top + 182880  # ~0.2inch down
            if s4_guide_data: _set_s4_guide(sh, s4_guide_data)

    # S5
    buf_s5=_make_ltv_graph(t_range,rev_line,gp_line,cac_line,ltv_rev,lam_actual,x_max)
    for sh in prs.slides[4].shapes:
        if sh.name=='コンテンツ プレースホルダー 7':
            sh.left=int(0.629*914400); sh.top=int(1.869*914400); sh.width=int(12.075*914400); sh.height=int(3.765*914400)
            _replace_image(prs.slides[4],sh,buf_s5)

    # S6~
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

        import math
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
            premium=(best['ltv_r']-avg_ltv)/avg_ltv*100
            avg_cac=sum(r['cac']*r['n'] for r in pp_rows)/n_total
            cac_diff=best['cac']-avg_cac
            cac_diff_str=f"+¥{cac_diff:,.0f}高く設定可能" if cac_diff>=0 else f"¥{abs(cac_diff):,.0f}低め"

            # S7
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
                    buf7=_make_bar_graph(pp_rows,best,avg_ltv); _replace_image(s7,sh,buf7)

            # S8
            s8=_copy_slide(prs,7); top10=pp_rows[:10]
            dr8=[[r['seg'],f'{r["n"]:,}',f'¥{r["ltv_r"]:,.0f}',f'¥{r["ltv_g"]:,.0f}',f'¥{r["cac"]:,.0f}',f'{r["k"]:.3f}',f'{r["lam"]:.1f}',f'{r["r2"]:.3f}'] for r in top10]
            ft8=['加重平均',f'{n_total:,}',f'¥{avg_ltv:,.0f}',f'¥{w_ltv_g:,.0f}',f'¥{w_cac:,.0f}','—','—','—']
            for sh in s8.shapes:
                if sh.name=='タイトル 2': _set_text(sh,f'{sc}: 分析結果のサマリー')
                elif sh.shape_type==19: _write_table_styled(sh.table,None,dr8,ft8)
                elif sh.name=='テキスト ボックス 9' and sh.has_text_frame:
                    diff_pct=(avg_ltv-ltv_rev)/ltv_rev*100
                    note=f'NOTE — テーブルは最大上位10項目を表示。加重平均行は全{len(pp_rows)}項目を顧客数で重み付けした値です。全体LTV∞との差（{diff_pct:+.1f}%）は統計的に正常な現象です。詳細スライドには全セグメントを掲載しています。'
                    _set_note_text(sh, note)

            # S9/S10
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
                rows_sx.append([f'lambda {round(row["lam"]):,}d',f'¥{lam_r_s:,.0f}',f'¥{lam_r_s*gpm:,.0f}',f'¥{lam_r_s*gpm/cac_n:,.0f}',f'{lam_r_s/row["ltv_r"]*100:.1f}%'])
                try:
                    from scipy.optimize import brentq as _bq
                    _arpu=df[df[sc]==row['seg']]['arpu_daily'].mean()
                    d99s=_bq(lambda h:ltv_horizon_offset(row['k'],row['lam'],_arpu,h,ltv_offset_days)/row['ltv_r']-0.99,1,100000)
                    r99s=ltv_horizon_offset(row['k'],row['lam'],_arpu,d99s,ltv_offset_days)
                    rows_sx.append([f'99% ({int(d99s):,}d)',f'¥{r99s:,.0f}',f'¥{r99s*gpm:,.0f}',f'¥{r99s*gpm/cac_n:,.0f}','99.0%'])
                except: pass

                for sh in sx.shapes:
                    if sh.name=='タイトル 8': _set_text(sh,f'{sc}: {row["seg"]}')
                    elif sh.name=='Picture 6' and buf_km: buf_km.seek(0); _replace_image(sx,sh,buf_km)
                    elif sh.name=='Picture 7' and buf_wb: buf_wb.seek(0); _replace_image(sx,sh,buf_wb)
                    elif sh.shape_type==19: _write_table(sh.table,None,rows_sx[:-1],rows_sx[-1])
                    elif sh.name=='テキスト ボックス 17' and sh.has_text_frame:
                        _set_text(sh,'TOP PICK　' if ri==0 else '')
                        # 垂直中央揃え
                        bodyPr = sh.text_frame._txBody.find(f'{{{A}}}bodyPr')
                        if bodyPr is None:
                            bodyPr = etree.SubElement(sh.text_frame._txBody, f'{{{A}}}bodyPr')
                        bodyPr.set('anchor', 'ctr')

        # テンプレートS7-S10削除
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
