"""
PPTX Export Module - ゼロベース書き直し版
テンプレートLTV-analyzer.pptxに数値・グラフを差し込む
"""
import io, copy
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


# ════════════════════════════════════════════════════════════════
# 日本語フォント設定（パスハードコードなし）
# ════════════════════════════════════════════════════════════════
def _setup_jp_font():
    for fp in fm.findSystemFonts():
        if any(x in fp.lower() for x in ['gothic','noto','cjk','ipa','japanese','unicode']):
            try:
                name = fm.FontProperties(fname=fp).get_name()
                plt.rcParams['font.family'] = name
                return
            except:
                pass

_setup_jp_font()
BG = '#111820'


# ════════════════════════════════════════════════════════════════
# ヘルパー関数
# ════════════════════════════════════════════════════════════════
def _set_text(shape, text):
    """シェイプの先頭runにテキストをセット（書式保持）"""
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    for para in tf.paragraphs:
        for r in para.runs:
            r.text = ''
    if tf.paragraphs and tf.paragraphs[0].runs:
        tf.paragraphs[0].runs[0].text = text


def _replace_image(slide, shape, buf):
    """シェイプの画像をbufで差し替え"""
    blip = shape._element.find('.//' + qn('a:blip'))
    if blip is None:
        return
    buf.seek(0)
    img_part, new_rId = slide.part.get_or_add_image_part(buf)
    blip.set(qn('r:embed'), new_rId)


def _write_table(tbl, header, rows, footer=None):
    """
    テーブルにデータを書き込む。行数はデータに合わせて動的に調整。
    header: 使わない（テンプレートのヘッダー行を保持）
    rows: [[col0, col1, ...], ...]
    footer: 最終行（加重平均など）
    """
    # 必要な行数 = ヘッダー1 + データ行 + フッター1
    data_rows = rows + ([footer] if footer else [])
    needed = 1 + len(data_rows)

    # 行が足りなければテンプレートのデータ行(row[1])をコピーして追加
    template_data_row = copy.deepcopy(tbl.rows[1]._tr) if len(tbl.rows) > 1 else None
    while len(tbl.rows) < needed and template_data_row is not None:
        tbl._tbl.append(copy.deepcopy(template_data_row))

    # 多すぎる行を削除
    while len(tbl.rows) > needed:
        tbl._tbl.remove(tbl.rows[-1]._tr)

    # データを書き込む
    for ri, row_data in enumerate(data_rows):
        row = tbl.rows[ri + 1]
        for ci, val in enumerate(row_data):
            if ci >= len(row.cells):
                break
            tf = row.cells[ci].text_frame
            for para in tf.paragraphs:
                for r in para.runs:
                    r.text = ''
            if tf.paragraphs and tf.paragraphs[0].runs:
                tf.paragraphs[0].runs[0].text = str(val)


def _copy_slide(prs, from_idx):
    """スライドを末尾に複製"""
    src = prs.slides[from_idx]
    new = prs.slides.add_slide(src.slide_layout)
    src_sp = src.shapes._spTree
    new_sp = new.shapes._spTree
    for el in list(new_sp):
        new_sp.remove(el)
    for el in list(src_sp):
        new_sp.append(copy.deepcopy(el))
    for rId, rel in src.part.rels.items():
        if rel.reltype == RT.IMAGE:
            new.part.rels._rels[rId] = rel
    return new


def _make_ltv_graph(t_range, rev_line, gp_line, cac_line, ltv_rev, lam_actual, x_max):
    """S5: LTV推移グラフ画像を生成"""
    fig, ax = plt.subplots(figsize=(898/150, 280/150), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(t_range, rev_line, color='#56b4d3', lw=1.8, ls='-')
    ax.plot(t_range, gp_line,  color='#a8dadc', lw=1.8, ls='--')
    ax.plot(t_range, cac_line, color='#4a7a8a', lw=1.4, ls=':')
    ax.axhline(ltv_rev,    color='#56b4d3', lw=0.8, ls=':', alpha=0.5)
    ax.axvline(lam_actual, color='#a8dadc', lw=1.0, ls='--', alpha=0.7)

    ax.set_xlim(0, x_max + 30)
    ax.set_ylim(0, ltv_rev * 1.08)
    ax.set_yticks([0, 50000, 100000, 150000])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{int(v):,}'.replace(',', '')))
    ax.set_xticks([180, 365, 730, 1095, 1460, 1825])
    ax.set_xticklabels(['180日','1年','2年','3年','4年','5年'],
                       color='#888888', fontsize=7)
    ax.tick_params(colors='#888888', labelsize=7, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color('#888888')
    ax.grid(True, alpha=0.15, color='#1a3040')
    for sp in ax.spines.values():
        sp.set_color('#1a3040')
    ax.set_xlabel('継続期間', color='#888888', fontsize=8)
    ax.set_ylabel('金額（円）', color='#888888', fontsize=8)
    ax.legend(
        [Line2D([0],[0],color='#56b4d3',lw=1.8),
         Line2D([0],[0],color='#a8dadc',lw=1.8,ls='--'),
         Line2D([0],[0],color='#4a7a8a',lw=1.4,ls=':')],
        ['LTV（売上）','LTV（粗利）','CAC上限'],
        loc='upper left', frameon=False, fontsize=7, labelcolor='white',
        ncol=3, bbox_to_anchor=(0.0, 1.18))
    ax.annotate(f'λ={int(lam_actual)}日',
                xy=(lam_actual, ltv_rev * 0.92),
                color='white', fontsize=7, annotation_clip=False)
    ax.annotate(f'LTV∞ ¥{ltv_rev:,.0f}',
                xy=(x_max + 32, ltv_rev),
                color='#56b4d3', fontsize=7,
                va='center', annotation_clip=False, clip_on=False)
    fig.subplots_adjust(left=0.11, right=0.84, top=0.78, bottom=0.20)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=BG, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def _make_bar_graph(pp_rows, best, avg_ltv):
    """S7: セグメントLTV∞棒グラフ"""
    fig, ax = plt.subplots(figsize=(6, max(2.5, len(pp_rows)*0.4)), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    segs  = [r['seg'] for r in pp_rows]
    vals  = [r['ltv_r'] for r in pp_rows]
    colors = ['#56b4d3' if r['seg'] == best['seg'] else '#2a4a5a' for r in pp_rows]
    ax.barh(segs[::-1], vals[::-1], color=colors[::-1], height=0.6)
    ax.axvline(avg_ltv, color='#a8dadc', lw=1, ls='--', alpha=0.7)
    ax.tick_params(colors='#888888', labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'¥{v/10000:.0f}万'))
    for sp in ax.spines.values(): sp.set_color('#1a3040')
    ax.grid(True, alpha=0.15, color='#1a3040', axis='x')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf


# ════════════════════════════════════════════════════════════════
# メイン生成関数
# ════════════════════════════════════════════════════════════════
def generate_pptx(
    tmpl_path,
    # 分析パラメータ
    k, lam, lam_actual, r2, arpu_daily, gpm, gp_daily, cac_n, cac_upper,
    ltv_rev, lam_rev, lam_gp, rev_99, gp_99, days_99,
    t_range, rev_line, gp_line, cac_line, x_max,
    # テーブルデータ
    tbl_rows, horizons, ltv_offset_days,
    ltv_horizon_offset, ltv_horizon_spot,
    business_type, dormancy_days,
    # グラフバッファ（S3）
    buf1, buf2,
    # メタ情報
    df, client_name, analyst_name, billing_cycle_display,
    k_summary, r2_summary, cac_label,
    cac_recover_rev_str, cac_recover_gp_str,
    # セグメント
    segment_cols_input,
    _compute_km_df, _fit_weibull_df, ltv_inf, fmt_horizon,
):
    # k_insight / r2_comment を内部計算
    if k < 0.7:
        k_insight = f"k={k:.3f}（強い初期集中型）: 利用開始直後の体験品質が生死を分ける構造。30日以内の離脱防止施策が最重要。"
    elif k < 1.0:
        k_insight = f"k={k:.3f}（緩やかな初期集中型）: 離脱率は一定に近いが初期にやや多め。オンボーディング改善とリテンション施策を並行実施。"
    elif k < 1.5:
        k_insight = f"k={k:.3f}（逓増型・中程度）: 継続期間が長いほど離脱リスクが増す。1年超の顧客へのエンゲージメント強化が鍵。"
    else:
        k_insight = f"k={k:.3f}（強い逓増型）: 長期顧客ほど急速に離脱。VIP施策・継続特典による長期繋ぎ止めが急務。"
    if r2 >= 0.95:
        r2_comment = f"R²={r2:.3f}: 非常に高精度。LTV∞推定値の信頼性は高い。"
    elif r2 >= 0.85:
        r2_comment = f"R²={r2:.3f}: 許容範囲内。推定値に±15%程度の幅を見込んで意思決定を。"
    else:
        r2_comment = f"R²={r2:.3f}: やや低め。データ件数不足または複数の離脱パターンが混在している可能性あり。"
    prs = Presentation(tmpl_path)

    # ── データ期間 ──
    _dcols = [df['start_date']]
    if 'end_date' in df.columns and df['end_date'].notna().any():
        _dcols.append(df['end_date'])
    _data_start = __import__('pandas').concat(_dcols).dropna().min().strftime('%Y/%m/%d')
    _data_end   = __import__('pandas').concat(_dcols).dropna().max().strftime('%Y/%m/%d')

    # ════════════════════════════════════════
    # S1: タイトル
    # ════════════════════════════════════════
    s1 = prs.slides[0]
    for sh in s1.shapes:
        if sh.name == 'TextBox 5': _set_text(sh, client_name or '')
        elif sh.name == 'TextBox 6': _set_text(sh, date.today().strftime('%Y年%m月%d日'))
        elif sh.name == 'TextBox 7': _set_text(sh, analyst_name or '')

    # ════════════════════════════════════════
    # S2: 分析結果サマリー
    # ════════════════════════════════════════
    s2 = prs.slides[1]
    for sh in s2.shapes:
        if sh.name == 'テキスト プレースホルダー 6' and sh.has_text_frame:
            tf = sh.text_frame
            info1 = (f"データ期間: {_data_start} – {_data_end}　|　顧客数: {len(df):,}件　|　"
                     f"解約済み: {df['event'].sum():,}件　|　継続中: {(df['event']==0).sum():,}件　|　"
                     f"平均日次 ARPU: ¥{arpu_daily:,.2f}　|　GPM: {gpm:.0%}")
            info2 = (f"異常値の処理：除外なし　|　{business_type}　|　"
                     f"{billing_cycle_display}　|　解約時の日割り計算：{'ON' if ltv_offset_days==0 else 'OFF'}")
            if len(tf.paragraphs) >= 1:
                p = tf.paragraphs[0]
                for r in p.runs: r.text = ''
                if p.runs: p.runs[0].text = info1
            if len(tf.paragraphs) >= 2:
                p = tf.paragraphs[1]
                for r in p.runs: r.text = ''
                if p.runs: p.runs[0].text = info2
        elif sh.name == 'グループ化 26':
            kpi = {'TextBox 6': f'¥{ltv_rev:,.0f}',
                   'TextBox 10': f'¥{cac_upper:,.0f}',
                   'TextBox 11': f'CAC上限（{cac_label}）',
                   'TextBox 14': f'{k:.3f}',
                   'TextBox 18': f'{lam_actual:.0f}日',
                   'TextBox 22': f'{r2:.3f}'}
            for g in sh.shapes:
                if g.name in kpi: _set_text(g, kpi[g.name])
        elif sh.name == 'テキスト ボックス 49' and sh.has_text_frame:
            tf = sh.text_frame
            if len(tf.paragraphs) >= 2:
                for r in tf.paragraphs[0].runs: r.text = ''
                if tf.paragraphs[0].runs: tf.paragraphs[0].runs[0].text = '結論'
                for r in tf.paragraphs[1].runs: r.text = ''
                if tf.paragraphs[1].runs: tf.paragraphs[1].runs[0].text = k_summary + r2_summary

    # ════════════════════════════════════════
    # S3: 分析の信頼性
    # ════════════════════════════════════════
    s3 = prs.slides[2]
    lam_disp = lam + ltv_offset_days if business_type == '都度購入型' else lam
    for sh in s3.shapes:
        if sh.name == 'Picture 3':
            buf1.seek(0); _replace_image(s3, sh, buf1)
        elif sh.name == 'Picture 4':
            buf2.seek(0); _replace_image(s3, sh, buf2)
        elif sh.name == 'TextBox 7':
            _set_text(sh, f"k（形状パラメータ） = {k:.3f}\n"
                          f"→ 左グラフ（Survival Curve）の曲線の急峻さを決める値。k=1で指数分布（一定離脱率）\n"
                          f"→ {k_insight}")
        elif sh.name == 'TextBox 8':
            _set_text(sh, f"λ（尺度パラメータ） = {lam_disp:.1f}日（約{lam_disp/365:.1f}年）\n"
                          f"→ 大きいほどLTV∞到達が長期化する。λ日時点での暫定LTV到達率はk値により異なる（k=1のとき63.2%）\n"
                          f"→ {r2_comment}")

    # ════════════════════════════════════════
    # S4: 暫定LTVテーブル
    # ════════════════════════════════════════
    s4 = prs.slides[3]
    rows_s4 = []
    for h in horizons:
        if business_type == '都度購入型':
            dp = dormancy_days or 180
            lr = ltv_horizon_spot(k, lam, arpu_daily, h, dp)
            lg = ltv_horizon_spot(k, lam, arpu_daily*gpm, h, dp)
        else:
            lr = ltv_horizon_offset(k, lam, arpu_daily, h, ltv_offset_days)
            lg = ltv_horizon_offset(k, lam, gp_daily, h, ltv_offset_days)
        label = f'{h}日' if h < 365 else f'{h//365}年（{h:,}日）'
        rows_s4.append([label, f'¥{lr:,.0f}', f'¥{lg:,.0f}', f'¥{lg/cac_n:,.0f}', f'{lr/ltv_rev*100:.1f}%'])
    rows_s4.append([f'λ {round(lam_actual):,}日',
                    f'¥{lam_rev:,.0f}', f'¥{lam_gp:,.0f}',
                    f'¥{lam_gp/cac_n:,.0f}', f'{lam_rev/ltv_rev*100:.1f}%'])
    rows_s4.append([f'LTV∞到達率: 99%（{int(days_99):,}日）',
                    f'¥{rev_99:,.0f}', f'¥{gp_99:,.0f}',
                    f'¥{gp_99/cac_n:,.0f}', '99.0%'])

    # S4テーブル読み方テキストを動的生成
    lam_years = lam_actual / 365
    if k < 0.7:
        k_desc = f"k={k:.3f}（初期離脱型）: 契約直後に大量離脱するパターンです。LTV∞は大きく見えますが到達に時間がかかります。"
    elif k < 1.0:
        k_desc = f"k={k:.3f}（緩やかな初期集中型）: 離脱率は一定に近いですが初期にやや多めです。"
    elif k < 1.5:
        k_desc = f"k={k:.3f}（逓増型）: 継続期間が長いほど離脱リスクが増すパターンです。"
    else:
        k_desc = f"k={k:.3f}（強い逓増型）: 長期顧客ほど急速に離脱するパターンです。"

    if lam_actual < 180:
        lam_desc = "短い継続期間"
    elif lam_actual < 365:
        lam_desc = "半年〜1年程度の継続期間"
    elif lam_actual < 730:
        lam_desc = "中程度の継続期間で、1〜2年継続する顧客が多いビジネス"
    else:
        lam_desc = "長期継続型のビジネス"

    s4_guide = (f"このテーブルの読み方\n"
                f"λ={round(lam_actual):,}日（約{lam_years:.1f}年）は{lam_desc}です。\n"
                f"{k_desc}\n"
                f"LTV∞到達率の列でCAC回収に必要な期間を確認してください。")

    for sh in s4.shapes:
        if sh.shape_type == 19:
            _write_table(sh.table, None, rows_s4[:-1], rows_s4[-1])
        elif sh.name == 'テキスト ボックス 3' and sh.has_text_frame:
            # 余分なparagraphを削除してからテキストをセット
            tf = sh.text_frame
            txBody = tf._txBody
            a_ns = 'http://schemas.openxmlformats.org/drawingml/2006/main'
            paras = txBody.findall(f'{{{a_ns}}}p')
            for p in paras[1:]:
                txBody.remove(p)
            if tf.paragraphs and tf.paragraphs[0].runs:
                tf.paragraphs[0].runs[0].text = s4_guide

    # ════════════════════════════════════════
    # S5: LTV推移グラフ
    # ════════════════════════════════════════
    s5 = prs.slides[4]
    buf_s5 = _make_ltv_graph(t_range, rev_line, gp_line, cac_line, ltv_rev, lam_actual, x_max)
    for sh in s5.shapes:
        if sh.name == 'コンテンツ プレースホルダー 7':
            sh.left=int(0.629*914400); sh.top=int(1.869*914400)
            sh.width=int(12.075*914400); sh.height=int(3.765*914400)
            _replace_image(s5, sh, buf_s5)

    # ════════════════════════════════════════
    # S6〜: セグメント別
    # ════════════════════════════════════════
    if not segment_cols_input.strip():
        # セグメントなし → S6〜S10を削除
        from pptx.oxml.ns import qn as _qn
        for _ in range(min(5, len(prs.slides) - 5)):
            sldIdLst = prs.slides._sldIdLst
            last = sldIdLst[-1]
            rId = last.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            if rId:
                try: prs.part.drop_rel(rId)
                except: pass
            sldIdLst.remove(last)
    else:
        seg_cols = [c.strip() for c in segment_cols_input.split(',')
                    if c.strip() and c.strip() in df.columns]

        # S6: 扉
        s6 = prs.slides[5]
        for sh in s6.shapes:
            if sh.name == 'TextBox 4':
                _set_text(sh, '  |  '.join(seg_cols))

        import math
        for sc in seg_cols:
            # セグメント計算
            pp_rows = []
            for sv in sorted(df[sc].dropna().unique()):
                df_s = df[df[sc] == sv]
                if len(df_s) < 10 or df_s['event'].sum() < 5: continue
                try:
                    km_s = _compute_km_df(df_s)
                    k_s, lam_s, r2_s, _ = _fit_weibull_df(km_s)
                    if k_s is None: continue
                    arpu_s = df_s['arpu_daily'].mean()
                    gp_s   = arpu_s * gpm
                    ltv_r, _ = ltv_inf(k_s, lam_s, arpu_s)
                    ltv_g, _ = ltv_inf(k_s, lam_s, gp_s)
                    pp_rows.append({'seg': str(sv), 'n': len(df_s),
                                    'ltv_r': ltv_r, 'ltv_g': ltv_g,
                                    'cac': ltv_g/cac_n, 'k': k_s,
                                    'lam': lam_s, 'r2': r2_s})
                except: continue
            if not pp_rows: continue

            pp_rows.sort(key=lambda x: x['ltv_r'], reverse=True)
            best    = pp_rows[0]
            n_total = sum(r['n'] for r in pp_rows)
            avg_ltv = sum(r['ltv_r']*r['n'] for r in pp_rows) / n_total
            avg_cac = sum(r['cac']*r['n']   for r in pp_rows) / n_total
            w_ltv_g = sum(r['ltv_g']*r['n'] for r in pp_rows) / n_total
            w_cac   = sum(r['cac']*r['n']   for r in pp_rows) / n_total
            premium  = (best['ltv_r'] - avg_ltv) / avg_ltv * 100
            cac_diff = best['cac'] - avg_cac
            cac_diff_str = f"+¥{cac_diff:,.0f}高く設定可能" if cac_diff >= 0 else f"¥{abs(cac_diff):,.0f}低め"

            # S7: 棒グラフ
            s7 = _copy_slide(prs, 6)
            for sh in s7.shapes:
                if sh.name == 'タイトル 4':
                    _set_text(sh, f'{sc}: LTV∞')
                elif sh.name == 'テキスト プレースホルダー 5' and sh.has_text_frame:
                    tf = sh.text_frame
                    if len(tf.paragraphs) >= 2:
                        for r in tf.paragraphs[0].runs: r.text = ''
                        if tf.paragraphs[0].runs:
                            tf.paragraphs[0].runs[0].text = f'TOP PICK　{best["seg"]}'
                        for r in tf.paragraphs[1].runs: r.text = ''
                        if tf.paragraphs[1].runs:
                            tf.paragraphs[1].runs[0].text = (
                                f'LTV∞(売上): ¥{best["ltv_r"]:,.0f}'
                                f'（全セグメント平均比 +{premium:.1f}%）　|　'
                                f'許容CAC上限 ¥{best["cac"]:,.0f}'
                                f'（全セグメント平均より{cac_diff_str}）')
                elif sh.name == 'コンテンツ プレースホルダー 8':
                    buf7 = _make_bar_graph(pp_rows, best, avg_ltv)
                    _replace_image(s7, sh, buf7)

            # S8: サマリーテーブル
            s8 = _copy_slide(prs, 7)
            top10 = pp_rows[:10]
            data_rows_s8 = [[r['seg'], f'{r["n"]:,}',
                             f'¥{r["ltv_r"]:,.0f}', f'¥{r["ltv_g"]:,.0f}',
                             f'¥{r["cac"]:,.0f}', f'{r["k"]:.3f}',
                             f'{r["lam"]:.1f}', f'{r["r2"]:.3f}']
                            for r in top10]
            footer_s8 = ['加重平均', f'{n_total:,}',
                         f'¥{avg_ltv:,.0f}', f'¥{w_ltv_g:,.0f}',
                         f'¥{w_cac:,.0f}', '—', '—', '—']
            for sh in s8.shapes:
                if sh.name == 'タイトル 2':
                    _set_text(sh, f'{sc}: 分析結果のサマリー')
                elif sh.shape_type == 19:
                    _write_table(sh.table, None, data_rows_s8, footer_s8)
                elif sh.name == 'テキスト ボックス 9' and sh.has_text_frame:
                    tf = sh.text_frame
                    diff_pct = (avg_ltv - ltv_rev) / ltv_rev * 100
                    note = (f'NOTE — テーブルは最大上位10項目を表示。'
                            f'加重平均行は全{len(pp_rows)}項目を顧客数で重み付けした値です。'
                            f'全体LTV∞との差（{diff_pct:+.1f}%）は統計的に正常な現象です。'
                            f'詳細スライドには全セグメントを掲載しています。')
                    for para in tf.paragraphs:
                        for r in para.runs: r.text = ''
                    if tf.paragraphs and tf.paragraphs[0].runs:
                        tf.paragraphs[0].runs[0].text = note

            # S9(TOP PICK) / S10(通常): セグメント詳細
            for ri, row in enumerate(pp_rows):
                tmpl_idx = 8 if ri == 0 else 9
                sx = _copy_slide(prs, tmpl_idx)

                # KM・Weibullグラフ
                buf_km = buf_wb = None
                try:
                    df_s  = df[df[sc] == row['seg']]
                    km_s  = _compute_km_df(df_s)
                    arpu_s = df_s['arpu_daily'].mean()

                    fig_km, ax_km = plt.subplots(figsize=(4, 2.8), dpi=120)
                    fig_km.patch.set_facecolor(BG); ax_km.set_facecolor(BG)
                    ax_km.step(km_s['time'], km_s['survival'],
                               where='post', color='#56b4d3', lw=1.5)
                    t_fit = list(range(int(km_s['time'].max())))
                    s_fit = [math.exp(-(t/row['lam'])**row['k']) for t in t_fit]
                    ax_km.plot(t_fit, s_fit, color='#a8dadc', lw=1.2, ls='--')
                    ax_km.tick_params(colors='#888888', labelsize=7)
                    for sp in ax_km.spines.values(): sp.set_color('#1a3040')
                    ax_km.grid(True, alpha=0.15, color='#1a3040')
                    fig_km.tight_layout()
                    buf_km = io.BytesIO()
                    fig_km.savefig(buf_km, format='png', dpi=120, facecolor=BG)
                    buf_km.seek(0); plt.close(fig_km)

                    x_s = list(range(1, int(max(1825, row['lam']*2)), max(1, int(row['lam']*2)//200)))
                    y_s = [ltv_horizon_offset(row['k'], row['lam'], arpu_s, t, ltv_offset_days) for t in x_s]
                    fig_wb, ax_wb = plt.subplots(figsize=(4, 2.8), dpi=120)
                    fig_wb.patch.set_facecolor(BG); ax_wb.set_facecolor(BG)
                    ax_wb.plot(x_s, y_s, color='#56b4d3', lw=1.5)
                    ax_wb.axvline(row['lam'], color='#a8dadc', lw=1, ls='--', alpha=0.7)
                    ax_wb.axhline(row['ltv_r'], color='#56b4d3', lw=0.8, ls=':', alpha=0.5)
                    ax_wb.tick_params(colors='#888888', labelsize=7)
                    for sp in ax_wb.spines.values(): sp.set_color('#1a3040')
                    ax_wb.grid(True, alpha=0.15, color='#1a3040')
                    fig_wb.tight_layout()
                    buf_wb = io.BytesIO()
                    fig_wb.savefig(buf_wb, format='png', dpi=120, facecolor=BG)
                    buf_wb.seek(0); plt.close(fig_wb)
                except: pass

                # 暫定LTVテーブル
                rows_sx = []
                for h in horizons:
                    lr_s = ltv_horizon_offset(row['k'], row['lam'],
                                              df[df[sc]==row['seg']]['arpu_daily'].mean(),
                                              h, ltv_offset_days)
                    lg_s = lr_s * gpm
                    rows_sx.append([fmt_horizon(h),
                                    f'¥{lr_s:,.0f}', f'¥{lg_s:,.0f}',
                                    f'¥{lg_s/cac_n:,.0f}',
                                    f'{lr_s/row["ltv_r"]*100:.1f}%'])
                lam_r_s = ltv_horizon_offset(row['k'], row['lam'],
                                             df[df[sc]==row['seg']]['arpu_daily'].mean(),
                                             row['lam'], ltv_offset_days)
                rows_sx.append([f'λ {round(row["lam"]):,}日',
                                f'¥{lam_r_s:,.0f}', f'¥{lam_r_s*gpm:,.0f}',
                                f'¥{lam_r_s*gpm/cac_n:,.0f}',
                                f'{lam_r_s/row["ltv_r"]*100:.1f}%'])
                try:
                    from scipy.optimize import brentq as _bq
                    _arpu = df[df[sc]==row['seg']]['arpu_daily'].mean()
                    d99s  = _bq(lambda h: ltv_horizon_offset(row['k'],row['lam'],_arpu,h,ltv_offset_days)/row['ltv_r']-0.99, 1, 100000)
                    r99s  = ltv_horizon_offset(row['k'],row['lam'],_arpu,d99s,ltv_offset_days)
                    rows_sx.append([f'LTV∞到達率: 99%（{int(d99s):,}日）',
                                    f'¥{r99s:,.0f}', f'¥{r99s*gpm:,.0f}',
                                    f'¥{r99s*gpm/cac_n:,.0f}', '99.0%'])
                except: pass

                for sh in sx.shapes:
                    if sh.name == 'タイトル 8':
                        _set_text(sh, f'{sc}: {row["seg"]}')
                    elif sh.name == 'Picture 6' and buf_km:
                        buf_km.seek(0); _replace_image(sx, sh, buf_km)
                    elif sh.name == 'Picture 7' and buf_wb:
                        buf_wb.seek(0); _replace_image(sx, sh, buf_wb)
                    elif sh.shape_type == 19:
                        _write_table(sh.table, None, rows_sx[:-1], rows_sx[-1])
                    elif sh.name == 'テキスト ボックス 17' and sh.has_text_frame:
                        # TOP PICKラベル: 先頭セグメントのみ表示、それ以外は消去
                        if ri == 0:
                            _set_text(sh, f'TOP PICK　{row["seg"]}')
                        else:
                            _set_text(sh, '')

        # テンプレートのS7〜S10（idx 6-9）を削除（複製済み）
        # 逆順で削除しないとインデックスがずれる
        sldIdLst = prs.slides._sldIdLst
        for idx in [9, 8, 7, 6]:
            if idx < len(sldIdLst):
                entry = sldIdLst[idx]
                rId = entry.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if rId:
                    try: prs.part.drop_rel(rId)
                    except: pass
                sldIdLst.remove(entry)

    # 保存
    buf_out = io.BytesIO()
    prs.save(buf_out)
    buf_out.seek(0)
    return buf_out
