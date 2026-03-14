"""TOB スクリーニング ダッシュボード (v2 - Supabase 読み取り版)

データは tob_updater.py (Cron Job) が Supabase に書き込み済み。
このアプリは読み取り・表示のみ。
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from supabase import create_client

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL") or st.secrets["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or st.secrets["SUPABASE_KEY"]

st.set_page_config(page_title="TOBスクリーニング ダッシュボード", layout="wide")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Supabase からデータ読み込み（1000件制限をページネーションで回避）
# ---------------------------------------------------------------------------
def _fetch_all(table: str, select: str = "*") -> list[dict]:
    """Supabase の1000件制限を回避して全行取得。"""
    all_data = []
    page_size = 1000
    offset = 0
    while True:
        resp = supabase.table(table).select(select).range(offset, offset + page_size - 1).execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return all_data


@st.cache_data(ttl=3600)
def load_stocks() -> pd.DataFrame:
    """tob_stocks テーブルから全銘柄データを読み込む。"""
    data = _fetch_all("tob_stocks")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["pbr"] = pd.to_numeric(df["pbr"], errors="coerce")
    df["net_cash_ratio"] = pd.to_numeric(df["net_cash_ratio"], errors="coerce")
    df["free_float_ratio"] = pd.to_numeric(df["free_float_ratio"], errors="coerce")
    df["volume_ratio"] = pd.to_numeric(df["volume_ratio"], errors="coerce")
    df["price_drop_pct"] = pd.to_numeric(df["price_drop_pct"], errors="coerce")
    df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
    if "eps" in df.columns:
        df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    if "equity_ratio" in df.columns:
        df["equity_ratio"] = pd.to_numeric(df["equity_ratio"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_parent_subsidiary() -> pd.DataFrame:
    data = _fetch_all("parent_subsidiary")
    if not data:
        return pd.DataFrame(columns=["parent_code", "parent_name", "child_code", "child_name", "holding_pct"])
    df = pd.DataFrame(data)
    df["holding_pct"] = pd.to_numeric(df["holding_pct"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_parent_extra() -> pd.DataFrame:
    data = _fetch_all("parent_extra")
    if not data:
        return pd.DataFrame(columns=["parent_code", "parent_pbr", "activist_in_parent", "activist_names"])
    df = pd.DataFrame(data)
    df["parent_pbr"] = pd.to_numeric(df["parent_pbr"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_edinet_shareholders() -> pd.DataFrame:
    """edinet_shareholders テーブルから大株主データを読み込む。"""
    data = _fetch_all("edinet_shareholders")
    if not data:
        return pd.DataFrame(columns=[
            "code", "rank", "shareholder_name", "holding_ratio",
            "shares_held", "report_date", "is_activist",
        ])
    df = pd.DataFrame(data)
    df["holding_ratio"] = pd.to_numeric(df["holding_ratio"], errors="coerce")
    df["shares_held"] = pd.to_numeric(df["shares_held"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_edinet_holders() -> pd.DataFrame:
    """edinet_holders テーブルから大量保有報告書データを読み込む。"""
    data = _fetch_all("edinet_holders")
    if not data:
        return pd.DataFrame(columns=[
            "doc_id", "code", "filer_name", "holding_ratio",
            "purpose", "report_date", "doc_type_code", "is_activist",
        ])
    df = pd.DataFrame(data)
    df["holding_ratio"] = pd.to_numeric(df["holding_ratio"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# データ結合
# ---------------------------------------------------------------------------
def merge_all(stocks: pd.DataFrame, ps: pd.DataFrame, pe: pd.DataFrame,
              edinet: pd.DataFrame = None) -> pd.DataFrame:
    df = stocks.copy()

    # 親子上場マージ
    if not ps.empty:
        ps_dedup = ps.drop_duplicates(subset=["child_code"], keep="first")
        df = df.merge(
            ps_dedup[["child_code", "parent_code", "parent_name", "holding_pct"]],
            left_on="code", right_on="child_code", how="left",
        )
        df = df.rename(columns={
            "parent_code": "top_sh_code",
            "parent_name": "top_sh_name",
            "holding_pct": "top_sh_pct",
        })
        df = df.drop(columns=["child_code"], errors="ignore")
    else:
        df["top_sh_code"] = None
        df["top_sh_name"] = None
        df["top_sh_pct"] = None

    # 親会社追加情報マージ
    if not pe.empty:
        pe_map = pe.set_index("parent_code").to_dict("index")
        df["parent_pbr"] = df["top_sh_code"].map(
            lambda x: pe_map.get(x, {}).get("parent_pbr") if pd.notna(x) else None
        )
        df["activist_in_parent"] = df["top_sh_code"].map(
            lambda x: pe_map.get(x, {}).get("activist_in_parent", False) if pd.notna(x) else False
        )
        df["activist_names"] = df["top_sh_code"].map(
            lambda x: pe_map.get(x, {}).get("activist_names", "") if pd.notna(x) else ""
        )
    else:
        df["parent_pbr"] = None
        df["activist_in_parent"] = False
        df["activist_names"] = ""

    # EDINET 大量保有報告書マージ
    df["edinet_activist"] = False
    df["edinet_activist_names"] = ""
    df["edinet_max_ratio"] = None
    df["edinet_holder_count"] = 0

    if edinet is not None and not edinet.empty:
        # 各銘柄ごとにファイラー別の最新報告のみ残す
        edinet_sorted = edinet.sort_values("report_date", ascending=False)
        latest_per_filer = edinet_sorted.drop_duplicates(
            subset=["code", "filer_name"], keep="first"
        )

        for code in df["code"].unique():
            code_holders = latest_per_filer[latest_per_filer["code"] == code]
            if code_holders.empty:
                continue

            mask = df["code"] == code
            df.loc[mask, "edinet_holder_count"] = len(code_holders)

            # 最大保有割合
            max_ratio = code_holders["holding_ratio"].max()
            if pd.notna(max_ratio):
                df.loc[mask, "edinet_max_ratio"] = max_ratio

            # アクティビスト判定
            activist_rows = code_holders[code_holders["is_activist"] == True]
            if not activist_rows.empty:
                df.loc[mask, "edinet_activist"] = True
                names = activist_rows["filer_name"].unique().tolist()
                df.loc[mask, "edinet_activist_names"] = ", ".join(names)

                # EDINET のアクティビスト情報で既存フラグも補完
                if not df.loc[mask, "activist_in_parent"].any():
                    df.loc[mask, "activist_in_parent"] = True
                    existing_names = df.loc[mask, "activist_names"].iloc[0]
                    combined = ", ".join(filter(None, [existing_names, ", ".join(names)]))
                    df.loc[mask, "activist_names"] = combined

    return df


# ---------------------------------------------------------------------------
# TOBスコア算出
# ---------------------------------------------------------------------------
def calculate_tob_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = df.copy()

    # PBR Score: PBR 2.0以下で高スコア（バックテスト: TOBターゲット中央値1.11）
    pbr = out["pbr"].clip(lower=0, upper=5)
    out["score_pbr"] = np.where(pbr.isna(), 0, np.clip((2.0 - pbr) / 2.0, 0, 1) * 100)

    nc = out["net_cash_ratio"].fillna(-1)
    out["score_nc"] = np.clip((nc + 0.5) / 1.0, 0, 1) * 100

    # Small Cap Score: 500億円以下で高スコア（バックテスト: 中央値1399億円）
    mc = out["market_cap"].fillna(1e12)
    mc_billion = mc / 1e9
    out["score_smallcap"] = np.clip((500 - mc_billion) / 450, 0, 1) * 100

    vr = out["volume_ratio"].fillna(1.0)
    out["score_volume"] = np.clip((vr - 1.0) / 2.0, 0, 1) * 100

    # Price Drop Score: 40%下落で満点（バックテスト: 最も有効なファクター）
    pd_pct = out["price_drop_pct"].fillna(0)
    out["score_pricedrop"] = np.clip(pd_pct / 0.4, 0, 1) * 100

    has_parent = out["top_sh_code"].notna()
    top_pct = out["top_sh_pct"].fillna(0)
    out["score_top_sh"] = np.where(
        ~has_parent, 0,
        np.where(top_pct > 0, np.clip(top_pct / 50, 0, 1) * 100, 50),
    )

    # Activist Score: EDINET 大量保有報告書ベースのアクティビスト検出
    # アクティビスト介入あり = 100点、大量保有報告書あり = 30点、なし = 0点
    out["score_activist"] = np.where(
        out["edinet_activist"] | out["activist_in_parent"], 100,
        np.where(out["edinet_holder_count"] > 0, 30, 0),
    )

    w_total = sum(weights.values())
    out["tob_score"] = (
        out["score_pbr"] * weights["pbr"]
        + out["score_nc"] * weights["nc"]
        + out["score_smallcap"] * weights["smallcap"]
        + out["score_volume"] * weights["volume"]
        + out["score_pricedrop"] * weights["pricedrop"]
        + out["score_top_sh"] * weights["top_sh"]
        + out["score_activist"] * weights.get("activist", 0)
    ) / w_total

    return out


# ---------------------------------------------------------------------------
# レーダーチャート
# ---------------------------------------------------------------------------
def make_radar_chart(row: pd.Series) -> go.Figure:
    categories = ["PBR割安", "ネットキャッシュ", "低時価総額", "出来高急増", "株価下落", "親子上場", "アクティビスト"]
    values = [
        row["score_pbr"], row["score_nc"], row["score_smallcap"],
        row["score_volume"], row["score_pricedrop"], row["score_top_sh"],
        row.get("score_activist", 0),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=categories_closed, fill="toself",
        line=dict(color="#1f77b4"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=350, margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 株価チャート (個別銘柄のみ yfinance)
# ---------------------------------------------------------------------------
def make_price_chart(code: str) -> go.Figure | None:
    try:
        tk = yf.Ticker(f"{code}.T")
        hist = tk.history(period="6mo")
        if hist.empty:
            return None
    except Exception:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.7, 0.3], subplot_titles=("株価", "出来高"),
    )
    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"], name="株価",
        ),
        row=1, col=1,
    )
    colors = ["#ef5350" if c < o else "#26a69a" for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist["Volume"], marker_color=colors, name="出来高", showlegend=False),
        row=2, col=1,
    )
    fig.update_layout(
        height=400, xaxis_rangeslider_visible=False,
        template="plotly_white", showlegend=False,
    )
    fig.update_yaxes(title_text="価格 (JPY)", row=1, col=1)
    fig.update_yaxes(title_text="出来高", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# グループマップ構築
# ---------------------------------------------------------------------------
def build_group_map(scored_df: pd.DataFrame, ps_df: pd.DataFrame, pe_df: pd.DataFrame) -> pd.DataFrame:
    if ps_df.empty:
        return pd.DataFrame()

    pe_map = pe_df.set_index("parent_code").to_dict("index") if not pe_df.empty else {}

    child_data = scored_df[["code", "name", "market", "market_cap", "pbr", "tob_score"]].copy()
    group = ps_df.merge(child_data, left_on="child_code", right_on="code", how="inner")

    rows = []
    for _, r in group.iterrows():
        pe = pe_map.get(r["parent_code"], {})
        rows.append({
            "holder_code": r["parent_code"],
            "holder_name": r["parent_name"],
            "holder_pbr": pe.get("parent_pbr"),
            "holder_activist": pe.get("activist_in_parent", False),
            "holder_activist_names": pe.get("activist_names", ""),
            "target_code": r["child_code"],
            "target_name": r["child_name"],
            "target_market": r["market"],
            "holding_pct": r["holding_pct"],
            "target_market_cap": r["market_cap"],
            "target_pbr": r["pbr"],
            "target_tob_score": r["tob_score"],
        })

    if not rows:
        return pd.DataFrame()

    group_df = pd.DataFrame(rows)
    group_df = group_df.sort_values(["holder_code", "holding_pct"], ascending=[True, False], na_position="last")
    return group_df


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    st.title("TOBスクリーニング ダッシュボード")

    # --- データ読み込み ---
    stocks = load_stocks()
    if stocks.empty:
        st.error("銘柄データがありません。tob_updater.py を実行してデータを投入してください。")
        return

    ps_df = load_parent_subsidiary()
    pe_df = load_parent_extra()
    edinet_df = load_edinet_holders()
    edinet_sh_df = load_edinet_shareholders()
    df = merge_all(stocks, ps_df, pe_df, edinet_df)

    updated_at = stocks["updated_at"].max() if "updated_at" in stocks.columns else "不明"
    st.caption(f"最終更新: {updated_at}　|　取得銘柄数: {len(df)}　|　親子上場: {len(ps_df)} ペア登録")

    # --- サイドバー ---
    st.sidebar.header("フィルター")

    markets = sorted(df["market"].unique())
    selected_markets = st.sidebar.multiselect("市場区分", markets, default=markets)
    min_score = st.sidebar.slider("TOBスコア 下限", 0, 100, 30, step=5)
    only_with_parent = st.sidebar.checkbox("親子上場のみ表示")

    st.sidebar.header("指標ウェイト")
    w_pbr = st.sidebar.slider("PBR割安", 0, 50, 25)
    w_nc = st.sidebar.slider("ネットキャッシュ比率", 0, 50, 20)
    w_smallcap = st.sidebar.slider("低時価総額", 0, 50, 20)
    w_volume = st.sidebar.slider("出来高急増", 0, 50, 5)
    w_pricedrop = st.sidebar.slider("株価下落(52週高値比)", 0, 50, 25)
    w_top_sh = st.sidebar.slider("親子上場", 0, 50, 20)
    w_activist = st.sidebar.slider("アクティビスト(EDINET)", 0, 50, 15)

    weights = {
        "pbr": w_pbr, "nc": w_nc, "smallcap": w_smallcap,
        "volume": w_volume, "pricedrop": w_pricedrop, "top_sh": w_top_sh,
        "activist": w_activist,
    }

    if st.sidebar.button("キャッシュクリア"):
        st.cache_data.clear()
        st.rerun()

    # --- スコア算出 & フィルタ ---
    scored = calculate_tob_score(df, weights)
    view = scored[scored["market"].isin(selected_markets)].copy()
    view = view[view["tob_score"] >= min_score]
    if only_with_parent:
        view = view[view["top_sh_code"].notna()]
    view = view.sort_values("tob_score", ascending=False).reset_index(drop=True)

    # =====================================================================
    # タブ構成
    # =====================================================================
    tab_screening, tab_group = st.tabs(["TOBスクリーニング", "グループマップ（逆引き）"])

    # =================================================================
    # タブ1: TOBスクリーニング
    # =================================================================
    with tab_screening:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("対象銘柄数", f"{len(view)}")
        col2.metric("スコア70以上", f"{(view['tob_score'] >= 70).sum()}")
        col3.metric("スコア50以上", f"{(view['tob_score'] >= 50).sum()}")
        col4.metric("親子上場", f"{view['top_sh_code'].notna().sum()}")
        col5.metric("親アクティビスト", f"{view['activist_in_parent'].sum()}")
        col6.metric("EDINET大量保有", f"{(view['edinet_holder_count'] > 0).sum()}")

        display = view[["code", "name", "market", "tob_score", "pbr",
                         "net_cash_ratio", "market_cap", "volume_ratio",
                         "price_drop_pct", "top_sh_name", "top_sh_pct",
                         "parent_pbr", "free_float_ratio", "activist_in_parent",
                         "activist_names", "edinet_holder_count",
                         "edinet_max_ratio"]].copy()
        display.insert(0, "順位", range(1, len(display) + 1))
        display["tob_score"] = display["tob_score"].round(1)
        display["pbr"] = display["pbr"].round(2)
        display["net_cash_ratio"] = (display["net_cash_ratio"] * 100).round(1)
        display["market_cap"] = (display["market_cap"] / 1e8).round(0)
        display["volume_ratio"] = display["volume_ratio"].round(2)
        display["price_drop_pct"] = (display["price_drop_pct"] * 100).round(1)
        display["top_sh_name"] = display["top_sh_name"].fillna("")
        display["top_sh_pct"] = display["top_sh_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else ""
        )
        display["parent_pbr"] = display["parent_pbr"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )
        display["free_float_ratio"] = display["free_float_ratio"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
        )
        display["activist_in_parent"] = display["activist_in_parent"].map(
            {True: "○", False: ""}
        ).fillna("")
        display["activist_names"] = display["activist_names"].fillna("")
        display["edinet_max_ratio"] = display["edinet_max_ratio"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else ""
        )

        display.columns = [
            "順位", "コード", "銘柄名", "市場区分", "TOBスコア", "PBR",
            "NC比率(%)", "時価総額(億円)", "出来高倍率", "52週高値乖離(%)",
            "親会社", "支配株主比率", "親会社PBR", "流動株比率",
            "アクティビスト", "アクティビスト名", "大量保有報告数",
            "最大保有割合",
        ]

        st.subheader("TOBスコア ランキング")
        st.dataframe(display, use_container_width=True, hide_index=True, height=500)

        st.subheader("銘柄詳細")
        if view.empty:
            st.info("条件に該当する銘柄がありません。")
        else:
            options = [f"{r['code']} - {r['name']}" for _, r in view.head(100).iterrows()]
            selected = st.selectbox("銘柄を選択", options)

            if selected:
                sel_code = selected.split(" - ")[0]
                sel_row = view[view["code"] == sel_code].iloc[0]

                st.markdown(f"### {sel_row['name']}（{sel_code}）  TOBスコア: **{sel_row['tob_score']:.1f}**")

                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    radar_fig = make_radar_chart(sel_row)
                    st.plotly_chart(radar_fig, use_container_width=True)

                with detail_col2:
                    st.markdown("**指標詳細**")
                    st.markdown(f"- PBR: **{sel_row['pbr']:.2f}**" if pd.notna(sel_row['pbr']) else "- PBR: N/A")
                    st.markdown(f"- ネットキャッシュ比率: **{sel_row['net_cash_ratio']*100:.1f}%**" if pd.notna(sel_row['net_cash_ratio']) else "- ネットキャッシュ比率: N/A")
                    st.markdown(f"- 時価総額: **{sel_row['market_cap']/1e8:.0f}億円**")
                    st.markdown(f"- 出来高倍率: **{sel_row['volume_ratio']:.2f}倍**" if pd.notna(sel_row['volume_ratio']) else "- 出来高倍率: N/A")
                    st.markdown(f"- 52週高値からの下落率: **{sel_row['price_drop_pct']*100:.1f}%**" if pd.notna(sel_row['price_drop_pct']) else "- 52週高値からの下落率: N/A")
                    st.markdown(f"- 流動株比率: **{sel_row['free_float_ratio']*100:.1f}%**" if pd.notna(sel_row.get('free_float_ratio')) else "- 流動株比率: N/A")
                    if pd.notna(sel_row.get("top_sh_name")):
                        pct_str = f" 支配株主比率 **{sel_row['top_sh_pct']:.1f}%**" if pd.notna(sel_row.get("top_sh_pct")) else ""
                        st.markdown(f"- 親会社: **{sel_row['top_sh_name']}**（{sel_row['top_sh_code']}）{pct_str}")
                        parent_pbr_str = f"**{sel_row['parent_pbr']:.2f}**" if pd.notna(sel_row.get("parent_pbr")) else "N/A"
                        st.markdown(f"- 親会社PBR: {parent_pbr_str}")
                        if sel_row.get("activist_in_parent"):
                            st.markdown(f"- 🚨 親会社にアクティビスト: **{sel_row['activist_names']}**")
                        else:
                            st.markdown("- 親会社にアクティビスト: なし")
                    else:
                        st.markdown("- 親子上場: 該当なし")

                    # EDINET 大量保有報告書情報
                    edinet_count = sel_row.get("edinet_holder_count", 0)
                    if edinet_count > 0:
                        st.markdown(f"- 大量保有報告: **{edinet_count}件**")
                        max_ratio = sel_row.get("edinet_max_ratio")
                        if pd.notna(max_ratio):
                            st.markdown(f"- 最大保有割合: **{max_ratio*100:.1f}%**")
                        if sel_row.get("edinet_activist"):
                            st.markdown(f"- EDINET アクティビスト: **{sel_row['edinet_activist_names']}**")

                # 大株主テーブル（EDINET 有価証券報告書）
                sel_shareholders = edinet_sh_df[edinet_sh_df["code"] == sel_code].sort_values("rank")
                if not sel_shareholders.empty:
                    st.markdown("**大株主の状況（有価証券報告書）**")
                    sh_display = sel_shareholders[["rank", "shareholder_name", "holding_ratio", "is_activist"]].copy()
                    sh_display["holding_ratio"] = sh_display["holding_ratio"].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""
                    )
                    sh_display["is_activist"] = sh_display["is_activist"].map({True: "★", False: ""}).fillna("")
                    sh_display.columns = ["順位", "株主名", "保有比率", "アクティビスト"]
                    st.dataframe(sh_display, use_container_width=True, hide_index=True)

                price_fig = make_price_chart(sel_code)
                if price_fig:
                    st.plotly_chart(price_fig, use_container_width=True)
                else:
                    st.warning("株価チャートを取得できませんでした。")

    # =================================================================
    # タブ2: グループマップ（逆引き）
    # =================================================================
    with tab_group:
        st.subheader("上場企業グループマップ")
        st.caption("親会社が保有する上場子会社を逆引きで一覧表示します。TOB（完全子会社化）候補の発見に活用できます。")

        group_df = build_group_map(scored, ps_df, pe_df)

        if group_df.empty:
            st.info("親子上場データがありません。")
        else:
            gm_col1, gm_col2 = st.columns(2)
            with gm_col1:
                min_holding = st.slider(
                    "保有比率の下限 (%)", 0, 100, 0, step=5,
                    key="gm_min_holding",
                )
            with gm_col2:
                sort_by = st.selectbox(
                    "並び順",
                    ["保有先数（多い順）", "最大保有比率（高い順）"],
                    key="gm_sort",
                )

            if min_holding > 0:
                filtered = group_df[group_df["holding_pct"].fillna(0) >= min_holding]
            else:
                filtered = group_df

            if filtered.empty:
                st.info(f"保有比率 {min_holding}% 以上の該当がありません。")
            else:
                holder_stats = filtered.groupby("holder_code").agg(
                    holder_name=("holder_name", "first"),
                    count=("target_code", "count"),
                    max_pct=("holding_pct", "max"),
                ).reset_index()

                if sort_by == "保有先数（多い順）":
                    holder_stats = holder_stats.sort_values("count", ascending=False)
                else:
                    holder_stats = holder_stats.sort_values("max_pct", ascending=False, na_position="last")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("親会社数", f"{len(holder_stats)}")
                kpi2.metric("上場子会社数", f"{len(filtered)}")
                has_pct = filtered["holding_pct"].notna()
                kpi3.metric("保有比率30%超", f"{(filtered.loc[has_pct, 'holding_pct'] >= 30).sum()}")
                kpi4.metric("親にアクティビスト", f"{filtered['holder_activist'].sum()}")

                st.markdown("---")

                for _, hs in holder_stats.iterrows():
                    h_code = hs["holder_code"]
                    h_name = hs["holder_name"]
                    h_count = hs["count"]

                    h_targets = filtered[filtered["holder_code"] == h_code]
                    h_pbr = h_targets["holder_pbr"].iloc[0] if not h_targets.empty else None
                    h_activist = h_targets["holder_activist"].iloc[0] if not h_targets.empty else False
                    h_activist_names = h_targets["holder_activist_names"].iloc[0] if not h_targets.empty else ""

                    pbr_label = f" PBR:{h_pbr:.2f}" if pd.notna(h_pbr) else ""
                    activist_label = f" 🚨アクティビスト:{h_activist_names}" if h_activist else ""
                    with st.expander(f"**{h_name}**（{h_code}）{pbr_label}{activist_label} — 上場子会社 {h_count} 社", expanded=(h_count >= 3)):
                        targets = filtered[filtered["holder_code"] == h_code].sort_values(
                            "holding_pct", ascending=False, na_position="last",
                        )
                        t_display = targets[["target_code", "target_name", "target_market",
                                             "holding_pct", "target_market_cap",
                                             "target_pbr", "target_tob_score"]].copy()
                        t_display["holding_pct"] = t_display["holding_pct"].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else "-"
                        )
                        t_display["target_market_cap"] = (t_display["target_market_cap"] / 1e8).round(0)
                        t_display["target_pbr"] = t_display["target_pbr"].round(2)
                        t_display["target_tob_score"] = t_display["target_tob_score"].round(1)

                        t_display.columns = [
                            "コード", "銘柄名", "市場区分", "保有比率(%)",
                            "時価総額(億円)", "PBR", "TOBスコア",
                        ]
                        st.dataframe(t_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
