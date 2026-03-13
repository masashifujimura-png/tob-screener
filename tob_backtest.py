"""TOB スコアリングモデル バックテスト

過去TOB事例のデータを使い、スコアリングモデルの予測力を検証する。
データ取得には yfinance を使用（全期間対応、レート制限が緩い）。

手法:
1. 各TOB事例の発表30日前のデータを取得
2. TOBスコアの各因子を算出
3. 全銘柄のスコア分布と比較
4. 結果を出力
"""

import time
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# データ取得 (yfinance)
# ---------------------------------------------------------------------------
def fetch_stock_data(code: str, ann_date: str) -> dict | None:
    """TOB発表前のデータを取得。ann_date は 'YYYY-MM-DD' 形式。"""
    ticker = f"{code}.T"
    try:
        tk = yf.Ticker(ticker)

        # 発表日の1年前〜発表前日の株価
        ann_dt = pd.Timestamp(ann_date)
        start = ann_dt - pd.DateOffset(years=1, days=45)  # 余裕を持って取得
        end = ann_dt - pd.DateOffset(days=1)

        hist = tk.history(start=start, end=end)
        if hist.empty or len(hist) < 10:
            return None

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()

        if close.empty:
            return None

        current_price = float(close.iloc[-1])

        # Volume ratio (5d / 60d)
        volume_ratio = None
        if len(volume) >= 10:
            avg_5d = volume.tail(5).mean()
            avg_60d = volume.tail(60).mean() if len(volume) >= 60 else volume.mean()
            if avg_60d > 0:
                volume_ratio = float(avg_5d / avg_60d)

        # Price drop from 52-week high
        price_drop_pct = None
        high_52w = float(close.max())
        if high_52w > 0:
            price_drop_pct = (high_52w - current_price) / high_52w

        # 財務データ
        info = tk.info
        shares = info.get("sharesOutstanding")
        bps = info.get("bookValue")
        market_cap = info.get("marketCap")

        pbr = None
        if bps and bps > 0:
            pbr = current_price / bps

        if market_cap is None and shares and current_price:
            market_cap = current_price * shares

        # Net cash ratio
        net_cash_ratio = None
        bs = tk.balance_sheet
        if bs is not None and not bs.empty and market_cap:
            # 発表前の最新BS
            bs_dates = [c for c in bs.columns if c <= ann_dt]
            if bs_dates:
                latest_bs = bs[max(bs_dates)]
                cash = latest_bs.get("Cash And Cash Equivalents", 0) or 0
                short_inv = latest_bs.get("Other Short Term Investments", 0) or 0
                long_inv = latest_bs.get("Long Term Equity Investment", 0) or 0
                securities = short_inv + long_inv
                total_liab = latest_bs.get("Total Liabilities Net Minority Interest", 0) or 0
                net_cash = cash + securities * 0.7 - total_liab
                net_cash_ratio = net_cash / market_cap

        # Free float
        free_float_ratio = None
        float_shares = info.get("floatShares")
        if float_shares and shares and shares > 0:
            free_float_ratio = float_shares / shares

        return {
            "current_price": current_price,
            "pbr": safe_float(pbr),
            "net_cash_ratio": safe_float(net_cash_ratio),
            "market_cap": safe_float(market_cap),
            "volume_ratio": safe_float(volume_ratio),
            "price_drop_pct": safe_float(price_drop_pct),
            "free_float_ratio": safe_float(free_float_ratio),
            "shares": shares,
            "bps": safe_float(bps),
        }
    except Exception as e:
        log.warning(f"  {code} データ取得エラー: {e}")
        return None


# ---------------------------------------------------------------------------
# スコア算出
# ---------------------------------------------------------------------------
def compute_factor_scores(row: dict) -> dict:
    """個別銘柄のファクタースコアを算出（0-100）。"""
    scores = {}

    pbr = row.get("pbr")
    scores["score_pbr"] = min(max((1.5 - pbr) / 1.5, 0), 1) * 100 if pbr and pbr > 0 else 0

    ncr = row.get("net_cash_ratio")
    scores["score_nc"] = min(max((ncr + 0.5) / 1.0, 0), 1) * 100 if ncr is not None else 0

    mc = row.get("market_cap")
    if mc:
        mc_b = mc / 1e9
        scores["score_smallcap"] = min(max((100 - mc_b) / 90, 0), 1) * 100
    else:
        scores["score_smallcap"] = 0

    vr = row.get("volume_ratio")
    scores["score_volume"] = min(max((vr - 1.0) / 2.0, 0), 1) * 100 if vr else 0

    pdp = row.get("price_drop_pct")
    scores["score_pricedrop"] = min(max(pdp / 0.5, 0), 1) * 100 if pdp else 0

    # TOB Score (default weights, parent score excluded for backtest)
    w = {"pbr": 25, "nc": 25, "smallcap": 15, "volume": 20, "pricedrop": 15}
    w_total = sum(w.values())
    scores["tob_score"] = (
        scores["score_pbr"] * w["pbr"]
        + scores["score_nc"] * w["nc"]
        + scores["score_smallcap"] * w["smallcap"]
        + scores["score_volume"] * w["volume"]
        + scores["score_pricedrop"] * w["pricedrop"]
    ) / w_total

    return scores


# ---------------------------------------------------------------------------
# バックテスト実行
# ---------------------------------------------------------------------------
def run_backtest(csv_path: str = "tob_cases_2015_2025.csv"):
    """バックテスト実行。"""
    log.info("=== TOB スコアリング バックテスト開始 ===")

    cases = pd.read_csv(csv_path)
    cases["target_code"] = cases["target_code"].astype(str).str.zfill(4)
    cases = cases.drop_duplicates(subset=["target_code", "announcement_date"], keep="last")
    log.info(f"TOB事例: {len(cases)} 件")

    results = []
    for i, (_, case) in enumerate(cases.iterrows()):
        code = str(case["target_code"]).zfill(4)
        ann = case["announcement_date"]
        tob_type = case.get("tob_type", "")

        log.info(f"  [{i+1}/{len(cases)}] {code} {case['target_name']} ({ann}) [{tob_type}]")

        data = fetch_stock_data(code, ann)
        if data is None:
            log.warning(f"    → データ取得失敗、スキップ")
            continue

        scores = compute_factor_scores(data)

        results.append({
            "code": code,
            "name": case["target_name"],
            "announcement_date": ann,
            "tob_type": tob_type,
            "premium_pct": case.get("premium_pct"),
            "current_price": data["current_price"],
            "pbr": data["pbr"],
            "net_cash_ratio": data["net_cash_ratio"],
            "market_cap_b": data["market_cap"] / 1e9 if data["market_cap"] else None,
            "volume_ratio": data["volume_ratio"],
            "price_drop_pct": data["price_drop_pct"],
            "free_float_ratio": data["free_float_ratio"],
            **scores,
        })

        time.sleep(1)  # yfinance rate limit

    if not results:
        log.warning("分析結果なし")
        return None

    df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # 分析結果出力
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  バックテスト結果: {len(df)} 件分析完了")
    print(f"{'='*60}")

    # 1. TOBスコアの分布
    print(f"\n■ TOBターゲットのスコア分布")
    print(f"  平均:     {df['tob_score'].mean():.1f}")
    print(f"  中央値:   {df['tob_score'].median():.1f}")
    print(f"  最小:     {df['tob_score'].min():.1f}")
    print(f"  最大:     {df['tob_score'].max():.1f}")
    print(f"  標準偏差: {df['tob_score'].std():.1f}")

    # 2. スコア帯別の分布
    print(f"\n■ スコア帯別分布")
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
    df["score_bin"] = pd.cut(df["tob_score"], bins=bins, labels=labels, right=False)
    dist = df["score_bin"].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>7}: {count:3d} ({pct:5.1f}%) {bar}")

    # 3. ファクター別の統計
    print(f"\n■ ファクター別スコア (TOBターゲット)")
    factors = {
        "score_pbr": "PBR割安",
        "score_nc": "ネットキャッシュ",
        "score_smallcap": "低時価総額",
        "score_volume": "出来高急増",
        "score_pricedrop": "株価下落",
    }
    for col, name in factors.items():
        mean = df[col].mean()
        med = df[col].median()
        nonzero = (df[col] > 0).sum()
        print(f"  {name:>12}: 平均 {mean:5.1f}  中央値 {med:5.1f}  非ゼロ {nonzero}/{len(df)}")

    # 4. TOBタイプ別
    print(f"\n■ TOBタイプ別スコア")
    for tob_type, group in df.groupby("tob_type"):
        print(f"  {tob_type:>8}: 平均 {group['tob_score'].mean():.1f} "
              f"(中央値 {group['tob_score'].median():.1f}, n={len(group)})")

    # 5. PBR分布
    print(f"\n■ PBR 分布")
    pbr_valid = df["pbr"].dropna()
    if len(pbr_valid) > 0:
        print(f"  PBR < 1.0: {(pbr_valid < 1.0).sum()} / {len(pbr_valid)} ({(pbr_valid < 1.0).mean()*100:.0f}%)")
        print(f"  PBR < 0.7: {(pbr_valid < 0.7).sum()} / {len(pbr_valid)} ({(pbr_valid < 0.7).mean()*100:.0f}%)")
        print(f"  平均: {pbr_valid.mean():.2f}  中央値: {pbr_valid.median():.2f}")

    # 6. 時価総額分布
    print(f"\n■ 時価総額分布 (億円)")
    mc_valid = df["market_cap_b"].dropna()
    if len(mc_valid) > 0:
        mc_oku = mc_valid * 10  # 十億→億
        print(f"  100億以下: {(mc_oku <= 100).sum()} / {len(mc_oku)}")
        print(f"  500億以下: {(mc_oku <= 500).sum()} / {len(mc_oku)}")
        print(f"  1000億以下: {(mc_oku <= 1000).sum()} / {len(mc_oku)}")
        print(f"  中央値: {mc_oku.median():.0f}億円")

    # 7. プレミアム vs スコアの相関
    print(f"\n■ プレミアム vs TOBスコア")
    prem_valid = df[["tob_score", "premium_pct"]].dropna()
    if len(prem_valid) > 5:
        corr = prem_valid["tob_score"].corr(prem_valid["premium_pct"])
        print(f"  相関係数: {corr:.3f}")
        high_score = prem_valid[prem_valid["tob_score"] >= 40]
        low_score = prem_valid[prem_valid["tob_score"] < 40]
        if len(high_score) > 0 and len(low_score) > 0:
            print(f"  スコア40以上のプレミアム平均: {high_score['premium_pct'].mean():.1f}%")
            print(f"  スコア40未満のプレミアム平均: {low_score['premium_pct'].mean():.1f}%")

    # 8. 改善提案
    print(f"\n■ スコアリング改善の示唆")
    # Which factors discriminate best?
    for col, name in factors.items():
        high = df[df[col] >= 50]
        print(f"  {name} >= 50: {len(high)}/{len(df)} ({len(high)/len(df)*100:.0f}%)")

    # 結果をCSV出力
    output_path = "tob_backtest_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 詳細結果を {output_path} に出力しました")

    return df


if __name__ == "__main__":
    run_backtest()
