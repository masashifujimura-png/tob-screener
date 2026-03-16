"""TOB スコアリングモデル バックテスト v2

過去TOB事例のデータを使い、スコアリングモデルの予測力を検証する。
データ取得には yfinance を使用（全期間対応）。

改善点 (v2):
- スコアリングロジックをダッシュボード準拠に統一
- BPSを過去のバランスシートから算出（現在値フォールバック付き）
- キャッシュ機構で再実行を高速化
- コントロールグループ（Supabase現在データ）との比較
- 親子上場・アクティビストの簡易推定
- 部分データでも進行（取得できたファクターだけでスコア算出）
"""

import os
import json
import time
import logging
import random

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CACHE_FILE = "tob_backtest_cache.json"


def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# データ取得 (yfinance)
# ---------------------------------------------------------------------------
def fetch_stock_data(code: str, ann_date: str, max_retries: int = 3) -> dict | None:
    """TOB発表前のデータを取得。ann_date は 'YYYY-MM-DD' 形式。"""
    ticker = f"{code}.T"
    ann_dt = pd.Timestamp(ann_date)

    for attempt in range(max_retries):
        try:
            tk = yf.Ticker(ticker)

            # 発表日の1年前〜発表前日の株価
            start = ann_dt - pd.DateOffset(years=1, days=45)
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
            market_cap = info.get("marketCap")

            if market_cap is None and shares and current_price:
                market_cap = current_price * shares

            # バランスシートから BPS とネットキャッシュを取得
            bps = None
            bps_source = None
            net_cash_ratio = None
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                bs_dates = [c for c in bs.columns if c <= ann_dt]
                if bs_dates:
                    latest_bs = bs[max(bs_dates)]

                    # BPS: 過去のバランスシートから算出
                    equity = latest_bs.get("Stockholders Equity")
                    if equity and shares and shares > 0:
                        bps = float(equity) / shares
                        bps_source = "historical"

                    # ネットキャッシュ
                    if market_cap:
                        cash = latest_bs.get("Cash And Cash Equivalents", 0) or 0
                        short_inv = latest_bs.get("Other Short Term Investments", 0) or 0
                        long_inv = latest_bs.get("Long Term Equity Investment", 0) or 0
                        securities = short_inv + long_inv
                        total_liab = latest_bs.get("Total Liabilities Net Minority Interest", 0) or 0
                        net_cash = cash + securities * 0.7 - total_liab
                        net_cash_ratio = net_cash / market_cap

            # BPS フォールバック: 過去BSから取れなければ現在値を使用
            if bps is None:
                bps_current = info.get("bookValue")
                if bps_current:
                    bps = float(bps_current)
                    bps_source = "current"

            pbr = current_price / bps if bps and bps > 0 else None

            return {
                "current_price": current_price,
                "pbr": safe_float(pbr),
                "net_cash_ratio": safe_float(net_cash_ratio),
                "market_cap": safe_float(market_cap),
                "volume_ratio": safe_float(volume_ratio),
                "price_drop_pct": safe_float(price_drop_pct),
                "shares": shares,
                "bps": safe_float(bps),
                "bps_source": bps_source,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                log.debug(f"  {code} リトライ {attempt+1}: {e}")
                time.sleep(2 ** attempt)
            else:
                log.warning(f"  {code} データ取得エラー: {e}")
                return None
    return None


def fetch_with_cache(code: str, ann_date: str, cache: dict) -> dict | None:
    """キャッシュ付きデータ取得。"""
    key = f"{code}_{ann_date}"
    if key in cache:
        cached = cache[key]
        if cached is None:
            return None
        return cached

    data = fetch_stock_data(code, ann_date)
    cache[key] = data
    return data


# ---------------------------------------------------------------------------
# スコア算出（ダッシュボード準拠）
# ---------------------------------------------------------------------------
def compute_factor_scores(row: dict, tob_type: str = "") -> dict:
    """個別銘柄のファクタースコアを算出（0-100）。ダッシュボードと同一ロジック。"""
    scores = {}

    # PBR Score: PBR 2.0以下で高スコア
    pbr = row.get("pbr")
    scores["score_pbr"] = min(max((2.0 - pbr) / 2.0, 0), 1) * 100 if pbr and pbr > 0 else 0

    # Price Drop Score: 40%下落で満点（バックテスト: リフト1.8x）
    pdp = row.get("price_drop_pct")
    scores["score_pricedrop"] = min(max(pdp / 0.4, 0), 1) * 100 if pdp else 0

    # 親子上場スコア（簡易推定: tob_typeが「親子」なら100）（バックテスト: リフト9.1x）
    scores["score_top_sh"] = 100 if tob_type == "親子" else 0

    # アクティビストスコア（簡易推定: 敵対的なら100）
    scores["score_activist"] = 100 if tob_type == "敵対的" else 0

    # ダッシュボードのデフォルト重み
    w = {"pbr": 20, "pricedrop": 25, "top_sh": 35, "activist": 20}
    w_total = sum(w.values())
    scores["tob_score"] = (
        scores["score_pbr"] * w["pbr"]
        + scores["score_pricedrop"] * w["pricedrop"]
        + scores["score_top_sh"] * w["top_sh"]
        + scores["score_activist"] * w["activist"]
    ) / w_total

    return scores


# ---------------------------------------------------------------------------
# コントロールグループ（Supabase 現在データとの比較）
# ---------------------------------------------------------------------------
def load_control_scores() -> pd.DataFrame | None:
    """Supabase の tob_stocks から全銘柄のスコア分布を取得。"""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_SERVICE_KEY", ""))
    if not url or not key:
        return None

    try:
        from supabase import create_client
        sb = create_client(url, key)

        all_data = []
        offset = 0
        while True:
            resp = (sb.table("tob_stocks")
                    .select("code, pbr, net_cash_ratio, market_cap, volume_ratio, price_drop_pct")
                    .range(offset, offset + 999).execute())
            if not resp.data:
                break
            all_data.extend(resp.data)
            if len(resp.data) < 1000:
                break
            offset += 1000

        if not all_data:
            return None

        df = pd.DataFrame(all_data)
        for col in ["pbr", "net_cash_ratio", "market_cap", "volume_ratio", "price_drop_pct"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 各銘柄のスコアを算出
        scores_list = []
        for _, row in df.iterrows():
            data = row.to_dict()
            s = compute_factor_scores(data)
            s["code"] = data["code"]
            scores_list.append(s)

        return pd.DataFrame(scores_list)
    except Exception as e:
        log.warning(f"コントロールデータ取得失敗: {e}")
        return None


# ---------------------------------------------------------------------------
# バックテスト実行
# ---------------------------------------------------------------------------
def run_backtest(csv_path: str = "tob_cases_2015_2025.csv"):
    """バックテスト実行。"""
    log.info("=== TOB スコアリング バックテスト v2 ===")

    cases = pd.read_csv(csv_path)
    cases["target_code"] = cases["target_code"].astype(str).str.zfill(4)
    cases = cases.drop_duplicates(subset=["target_code", "announcement_date"], keep="last")
    log.info(f"TOB事例: {len(cases)} 件")

    cache = load_cache()
    cached_count = 0

    results = []
    for i, (_, case) in enumerate(cases.iterrows()):
        code = str(case["target_code"]).zfill(4)
        ann = case["announcement_date"]
        tob_type = case.get("tob_type", "")

        key = f"{code}_{ann}"
        is_cached = key in cache
        if is_cached:
            cached_count += 1

        log.info(f"  [{i+1}/{len(cases)}] {code} {case['target_name']} ({ann}) "
                 f"[{tob_type}]{' (cache)' if is_cached else ''}")

        data = fetch_with_cache(code, ann, cache)
        if data is None:
            continue

        scores = compute_factor_scores(data, tob_type=tob_type)

        # データ完全性
        factors_available = sum(1 for k in ["pbr", "net_cash_ratio", "market_cap",
                                             "volume_ratio", "price_drop_pct"]
                                if data.get(k) is not None)

        results.append({
            "code": code,
            "name": case["target_name"],
            "announcement_date": ann,
            "tob_type": tob_type,
            "premium_pct": case.get("premium_pct"),
            "current_price": data["current_price"],
            "pbr": data["pbr"],
            "bps_source": data.get("bps_source", ""),
            "net_cash_ratio": data["net_cash_ratio"],
            "market_cap_b": data["market_cap"] / 1e9 if data["market_cap"] else None,
            "volume_ratio": data["volume_ratio"],
            "price_drop_pct": data["price_drop_pct"],
            "data_completeness": factors_available / 5,
            **scores,
        })

        if not is_cached:
            time.sleep(1)

        # 定期的にキャッシュ保存
        if (i + 1) % 20 == 0:
            save_cache(cache)

    save_cache(cache)
    log.info(f"キャッシュ: {cached_count} 件ヒット, {len(cache)} 件保存済み")

    if not results:
        log.warning("分析結果なし")
        return None

    df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # コントロールグループ取得
    # ---------------------------------------------------------------------------
    log.info("コントロールグループ（全銘柄スコア分布）を取得中...")
    ctrl = load_control_scores()

    # ---------------------------------------------------------------------------
    # 分析結果出力
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  バックテスト結果: {len(df)}/{len(cases)} 件分析完了")
    print(f"  データ完全性: 平均 {df['data_completeness'].mean()*100:.0f}%")
    print(f"  BPS取得元: 過去BS {(df['bps_source']=='historical').sum()}件, "
          f"現在値 {(df['bps_source']=='current').sum()}件, "
          f"なし {(df['bps_source']=='').sum()}件")
    print(f"{'='*60}")

    # 1. TOBスコアの分布
    print(f"\n■ TOBターゲットのスコア分布")
    print(f"  平均:     {df['tob_score'].mean():.1f}")
    print(f"  中央値:   {df['tob_score'].median():.1f}")
    print(f"  最小:     {df['tob_score'].min():.1f}")
    print(f"  最大:     {df['tob_score'].max():.1f}")
    print(f"  標準偏差: {df['tob_score'].std():.1f}")

    # コントロールとの比較
    if ctrl is not None and not ctrl.empty:
        ctrl_mean = ctrl["tob_score"].mean()
        ctrl_med = ctrl["tob_score"].median()
        print(f"\n  【コントロール（全{len(ctrl)}銘柄）】")
        print(f"  平均:     {ctrl_mean:.1f}")
        print(f"  中央値:   {ctrl_med:.1f}")
        print(f"  差分:     平均 +{df['tob_score'].mean() - ctrl_mean:.1f}, "
              f"中央値 +{df['tob_score'].median() - ctrl_med:.1f}")

        # パーセンタイル
        percentiles = []
        for score in df["tob_score"]:
            pct = (ctrl["tob_score"] < score).mean() * 100
            percentiles.append(pct)
        df["percentile"] = percentiles
        print(f"\n  TOBターゲットのパーセンタイル（全銘柄中の順位）")
        print(f"  平均:   {df['percentile'].mean():.1f}%ile")
        print(f"  中央値: {df['percentile'].median():.1f}%ile")
        print(f"  上位25%に入った割合: "
              f"{(df['percentile'] >= 75).sum()}/{len(df)} "
              f"({(df['percentile'] >= 75).mean()*100:.0f}%)")
        print(f"  上位50%に入った割合: "
              f"{(df['percentile'] >= 50).sum()}/{len(df)} "
              f"({(df['percentile'] >= 50).mean()*100:.0f}%)")

    # 2. スコア帯別の分布
    print(f"\n■ スコア帯別分布")
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
    df["score_bin"] = pd.cut(df["tob_score"], bins=bins, labels=labels, right=False)
    dist = df["score_bin"].value_counts().sort_index()
    for label, count in dist.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:>7}: {count:3d} ({pct:5.1f}%) {bar}")

    # 3. ファクター別の統計
    print(f"\n■ ファクター別スコア (TOBターゲット)")
    factors = {
        "score_pbr": "PBR割安",
        "score_pricedrop": "株価下落",
        "score_top_sh": "親子上場",
        "score_activist": "アクティビスト",
    }
    for col, name in factors.items():
        mean = df[col].mean()
        med = df[col].median()
        nonzero = (df[col] > 0).sum()
        ctrl_str = ""
        if ctrl is not None and col in ctrl.columns:
            ctrl_mean = ctrl[col].mean()
            diff = mean - ctrl_mean
            ctrl_str = f"  (全銘柄平均 {ctrl_mean:5.1f}, 差 {diff:+.1f})"
        print(f"  {name:>12}: 平均 {mean:5.1f}  中央値 {med:5.1f}  "
              f"非ゼロ {nonzero}/{len(df)}{ctrl_str}")

    # 4. TOBタイプ別
    print(f"\n■ TOBタイプ別スコア")
    for tob_type, group in df.groupby("tob_type"):
        print(f"  {tob_type:>8}: 平均 {group['tob_score'].mean():.1f} "
              f"(中央値 {group['tob_score'].median():.1f}, n={len(group)})")

    # 5. PBR分布
    print(f"\n■ PBR 分布")
    pbr_valid = df["pbr"].dropna()
    if len(pbr_valid) > 0:
        print(f"  PBR < 1.0: {(pbr_valid < 1.0).sum()} / {len(pbr_valid)} "
              f"({(pbr_valid < 1.0).mean()*100:.0f}%)")
        print(f"  PBR < 0.7: {(pbr_valid < 0.7).sum()} / {len(pbr_valid)} "
              f"({(pbr_valid < 0.7).mean()*100:.0f}%)")
        print(f"  平均: {pbr_valid.mean():.2f}  中央値: {pbr_valid.median():.2f}")

    # 6. 時価総額分布
    print(f"\n■ 時価総額分布 (億円)")
    mc_valid = df["market_cap_b"].dropna()
    if len(mc_valid) > 0:
        mc_oku = mc_valid * 10  # 十億→億
        print(f"  100億以下:  {(mc_oku <= 100).sum()} / {len(mc_oku)}")
        print(f"  500億以下:  {(mc_oku <= 500).sum()} / {len(mc_oku)}")
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

    # 8. 識別力の評価
    print(f"\n■ ファクター識別力")
    for col, name in factors.items():
        high = df[df[col] >= 50]
        print(f"  {name} >= 50: {len(high)}/{len(df)} ({len(high)/len(df)*100:.0f}%)")

    # 結果をCSV出力
    output_cols = [c for c in df.columns if c != "score_bin"]
    output_path = "tob_backtest_results.csv"
    df[output_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 詳細結果を {output_path} に出力しました")

    return df


if __name__ == "__main__":
    run_backtest()
