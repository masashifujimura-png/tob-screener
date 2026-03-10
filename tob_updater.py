"""TOB スクリーニング データ更新スクリプト (v2 - 4フェーズアーキテクチャ)

Phase 1: マスター同期 — JPX一覧 → 全銘柄を tob_stocks に登録（NULLでもOK）
Phase 2: 一括株価更新 — yf.download() → 株価系指標 + market_cap/PBR算出
Phase 3: 静的データ補完 — NULLの銘柄だけ個別取得（日々対象が減る）
Phase 4: 親会社情報更新 — アクティビスト判定
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from supabase import create_client

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY"))
MAX_WORKERS = 3
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# 四半期リフレッシュ対象月（決算シーズン）
QUARTERLY_MONTHS = {1, 4, 7, 10}

ACTIVIST_KEYWORDS = [
    "effissimo", "エフィッシモ",
    "strategic capital", "ストラテジックキャピタル",
    "oasis", "オアシス",
    "dalton", "ダルトン",
    "valueact", "バリューアクト",
    "elliott", "エリオット",
    "third point", "サード・ポイント",
    "taiyo", "タイヨウ",
    "レノ", "南青山不動産", "シティインデックスイレブンス",
    "murakami", "村上",
    "いちごアセット", "ichigo",
    "silchester", "シルチェスター",
    "ブランデス", "brandes",
    "asset value investors", "アセットバリューインベスターズ",
    "sparx", "スパークス",
    "3d investment", "3Dインベストメント",
    "rmb capital",
    "nippon active value", "ニッポン・アクティブ・バリュー",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def safe_float(value) -> float | None:
    """NaN/None を None に、それ以外を float に変換。"""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upsert_batch(table: str, rows: list[dict], conflict_key: str,
                  batch_size: int = 500, default_to_null: bool = True):
    """Supabase にバッチ upsert する。
    default_to_null=False にすると、行に含まれないカラムは既存値を保持する。
    """
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table(table).upsert(
            batch, on_conflict=conflict_key, default_to_null=default_to_null,
        ).execute()


# ---------------------------------------------------------------------------
# Phase 1: マスター同期
# ---------------------------------------------------------------------------
def fetch_jpx_list() -> pd.DataFrame:
    """JPX 上場企業一覧を取得。"""
    log.info("JPX 上場企業一覧を取得中…")
    df = pd.read_excel(JPX_URL)
    df = df.rename(columns={
        "コード": "code",
        "銘柄名": "name",
        "市場・商品区分": "market",
    })
    df = df[["code", "name", "market"]].dropna(subset=["code"])
    df["code"] = df["code"].astype(str).str.strip()
    market_map = {
        "プライム（内国株式）": "プライム",
        "スタンダード（内国株式）": "スタンダード",
        "グロース（内国株式）": "グロース",
    }
    df["market"] = df["market"].map(market_map)
    df = df.dropna(subset=["market"])
    log.info(f"JPX 一覧: {len(df)} 銘柄")
    return df.reset_index(drop=True)


def master_sync(jpx_df: pd.DataFrame):
    """全銘柄をtob_stocksに登録（name/marketのみ更新、財務データは触らない）。"""
    log.info("Phase 1: マスター同期")
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in jpx_df.iterrows():
        rows.append({
            "code": r["code"],
            "name": r["name"],
            "market": r["market"],
            "updated_at": now,
        })
    upsert_batch("tob_stocks", rows, "code")
    log.info(f"  {len(rows)} 銘柄をマスター同期完了")


# ---------------------------------------------------------------------------
# Phase 2: 一括株価更新
# ---------------------------------------------------------------------------
def _parse_price_batch(data, batch_codes: list[str], batch_tickers: list[str]) -> dict:
    """yf.download() の結果から銘柄ごとの株価データを抽出。"""
    price_data = {}
    for code, ticker in zip(batch_codes, batch_tickers):
        try:
            tk_data = data if len(batch_tickers) == 1 else data[ticker]
            if tk_data.empty or tk_data["Close"].dropna().empty:
                continue

            close = tk_data["Close"].dropna()
            volume = tk_data["Volume"].dropna()

            volume_ratio = None
            if len(volume) >= 10:
                avg_5d = volume.tail(5).mean()
                avg_60d = volume.tail(60).mean() if len(volume) >= 60 else volume.mean()
                if avg_60d and avg_60d > 0:
                    volume_ratio = float(avg_5d / avg_60d)

            price_drop_pct = None
            if len(close) > 0:
                week52_high = float(close.max())
                current_price = float(close.iloc[-1])
                if week52_high > 0:
                    price_drop_pct = (week52_high - current_price) / week52_high

            price_data[code] = {
                "volume_ratio": volume_ratio,
                "price_drop_pct": price_drop_pct,
                "current_price": float(close.iloc[-1]) if len(close) > 0 else None,
            }
        except Exception:
            continue
    return price_data


def bulk_download_prices(codes: list[str]) -> dict:
    """yf.download() で全銘柄の株価を一括取得。レートリミット時はリトライ。"""
    price_data = {}
    batch_size = 200  # レートリミット回避のためバッチを小さく
    remaining_codes = list(codes)

    for attempt in range(3):  # 最大3ラウンド
        if not remaining_codes:
            break
        if attempt > 0:
            wait = 30 * attempt
            log.info(f"  リトライ {attempt}: {len(remaining_codes)} 銘柄を {wait}秒後に再取得")
            time.sleep(wait)

        failed_codes = []
        for i in range(0, len(remaining_codes), batch_size):
            batch_codes = remaining_codes[i:i + batch_size]
            batch_tickers = [f"{c}.T" for c in batch_codes]
            log.info(f"  株価取得中… {min(i + batch_size, len(remaining_codes))}/{len(remaining_codes)}"
                     + (f" (リトライ{attempt})" if attempt > 0 else ""))

            try:
                data = yf.download(
                    batch_tickers, period="1y",
                    group_by="ticker", threads=True, progress=False,
                )
                if data.empty:
                    failed_codes.extend(batch_codes)
                    continue

                batch_result = _parse_price_batch(data, batch_codes, batch_tickers)
                price_data.update(batch_result)

                # 取得できなかった銘柄を記録
                for c in batch_codes:
                    if c not in batch_result:
                        failed_codes.append(c)

            except Exception as e:
                log.warning(f"  バッチ株価取得エラー: {e}")
                failed_codes.extend(batch_codes)
                continue

            time.sleep(2)  # バッチ間のウェイト

        remaining_codes = failed_codes
        if remaining_codes:
            log.info(f"  ラウンド{attempt + 1}完了: 成功 {len(price_data)}, 残り {len(remaining_codes)}")

    if remaining_codes:
        log.warning(f"  株価取得断念: {len(remaining_codes)} 銘柄（次回リトライ）")

    return price_data


def update_prices(codes: list[str], price_data: dict):
    """株価データ + 静的データから market_cap/pbr を算出して upsert。"""
    log.info("Phase 2: 一括株価更新")
    log.info(f"  株価取得成功: {len(price_data)}/{len(codes)} 銘柄")

    # 既存の静的データを読み込み
    resp = supabase.table("tob_stocks").select("code, shares_outstanding, bps").execute()
    static_map = {r["code"]: r for r in resp.data} if resp.data else {}

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for code in codes:
        price = price_data.get(code)
        if not price:
            continue

        current_price = price.get("current_price")
        static = static_map.get(code, {})
        shares = static.get("shares_outstanding")
        bps = static.get("bps")

        # market_cap = 株価 × 発行済株式数
        market_cap = None
        if current_price and shares:
            market_cap = int(current_price * shares)

        # pbr = 株価 / BPS
        pbr = None
        if current_price and bps and bps > 0:
            pbr = current_price / float(bps)

        rows.append({
            "code": code,
            "current_price": safe_float(current_price),
            "volume_ratio": safe_float(price.get("volume_ratio")),
            "price_drop_pct": safe_float(price.get("price_drop_pct")),
            "market_cap": market_cap,
            "pbr": safe_float(pbr),
            "updated_at": now,
        })

    upsert_batch("tob_stocks", rows, "code", default_to_null=False)
    log.info(f"  {len(rows)} 銘柄の株価データを更新")


# ---------------------------------------------------------------------------
# Phase 3: 静的データ補完
# ---------------------------------------------------------------------------
def fetch_static_data(code: str, max_retries: int = 2) -> dict | None:
    """1銘柄の静的データ（発行済株式数, BPS, BS）を個別取得。"""
    ticker_str = f"{code}.T"
    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(1)
        try:
            tk = yf.Ticker(ticker_str)
            info = tk.info

            shares_outstanding = info.get("sharesOutstanding")
            if not shares_outstanding or shares_outstanding == 0:
                return None

            bps = info.get("bookValue")

            free_float_ratio = None
            float_shares = info.get("floatShares")
            if float_shares and shares_outstanding > 0:
                free_float_ratio = float_shares / shares_outstanding

            # バランスシートからネットキャッシュ比率を算出
            net_cash_ratio = None
            market_cap_info = info.get("marketCap")
            bs = tk.balance_sheet
            if bs is not None and not bs.empty and market_cap_info:
                latest = bs.iloc[:, 0]
                cash = latest.get("Cash And Cash Equivalents", 0) or 0
                short_inv = latest.get("Other Short Term Investments", 0) or 0
                long_inv = latest.get("Long Term Equity Investment", 0) or 0
                securities = short_inv + long_inv
                total_liabilities = latest.get("Total Liabilities Net Minority Interest", 0) or 0
                net_cash = cash + securities * 0.7 - total_liabilities
                net_cash_ratio = net_cash / market_cap_info

            return {
                "code": code,
                "shares_outstanding": int(shares_outstanding),
                "bps": safe_float(bps),
                "net_cash_ratio": safe_float(net_cash_ratio),
                "free_float_ratio": safe_float(free_float_ratio),
            }
        except Exception:
            if attempt == max_retries:
                return None
    return None


def gap_fill_static(is_quarterly_refresh: bool = False):
    """NULLの銘柄だけ静的データを補完。四半期リフレッシュ時は古いデータも対象。"""
    log.info("Phase 3: 静的データ補完")

    if is_quarterly_refresh:
        # 今月1日より前に取得したデータは再取得対象
        first_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        resp = supabase.table("tob_stocks").select("code") \
            .or_(f"static_updated_at.is.null,static_updated_at.lt.{first_of_month.isoformat()}") \
            .execute()
        log.info("  四半期リフレッシュモード: 古い静的データも再取得対象")
    else:
        resp = supabase.table("tob_stocks").select("code") \
            .or_("shares_outstanding.is.null,bps.is.null") \
            .execute()

    if not resp.data:
        log.info("  補完対象の銘柄なし")
        return

    target_codes = [r["code"] for r in resp.data]
    log.info(f"  補完対象: {len(target_codes)} 銘柄")

    now = datetime.now(timezone.utc).isoformat()
    results = []
    done = 0
    failed = 0
    batch_size = 50

    for i in range(0, len(target_codes), batch_size):
        batch = target_codes[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_static_data, c): c for c in batch}
            for future in as_completed(futures):
                done += 1
                code = futures[future]
                result = future.result()
                if result is not None:
                    result["static_updated_at"] = now
                    results.append(result)
                else:
                    failed += 1
                if done % 100 == 0:
                    log.info(f"  {done}/{len(target_codes)} 処理済 (成功: {len(results)}, 失敗: {failed})")

        if i + batch_size < len(target_codes):
            time.sleep(2)

    if results:
        upsert_batch("tob_stocks", results, "code", default_to_null=False)
    log.info(f"  静的データ補完完了: {len(results)} 成功 / {failed} 失敗 / {len(target_codes)} 対象")


# ---------------------------------------------------------------------------
# Phase 4: 親会社情報更新
# ---------------------------------------------------------------------------
def check_activist_holders(code: str) -> tuple[bool, list[str]]:
    """アクティビストファンドの保有判定。"""
    try:
        tk = yf.Ticker(f"{code}.T")
        holders = tk.institutional_holders
        if holders is None or holders.empty:
            return False, []
        found = []
        for _, row in holders.iterrows():
            holder_name = str(row.get("Holder", ""))
            holder_lower = holder_name.lower()
            for kw in ACTIVIST_KEYWORDS:
                if kw.lower() in holder_lower:
                    found.append(holder_name)
                    break
        return len(found) > 0, found
    except Exception:
        return False, []


def update_parent_extra():
    """親会社の追加情報（PBR・アクティビスト）を更新。"""
    log.info("Phase 4: 親会社情報更新")

    ps_resp = supabase.table("parent_subsidiary").select("parent_code").execute()
    if not ps_resp.data:
        log.info("  親子上場データなし")
        return

    parent_codes = list({r["parent_code"] for r in ps_resp.data})

    # 親会社PBRをtob_stocksから取得
    pbr_resp = supabase.table("tob_stocks").select("code, pbr").execute()
    code_to_pbr = {r["code"]: r.get("pbr") for r in pbr_resp.data} if pbr_resp.data else {}

    now = datetime.now(timezone.utc).isoformat()
    parent_extra_rows = []
    done = 0
    batch_size = 30

    for i in range(0, len(parent_codes), batch_size):
        batch = parent_codes[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(check_activist_holders, pc): pc for pc in batch}
            for future in as_completed(futures):
                done += 1
                pc = futures[future]
                has_activist, activist_names = future.result()
                parent_extra_rows.append({
                    "parent_code": pc,
                    "parent_pbr": safe_float(code_to_pbr.get(pc)),
                    "activist_in_parent": has_activist,
                    "activist_names": ", ".join(activist_names) if activist_names else "",
                    "updated_at": now,
                })
                if done % 20 == 0:
                    log.info(f"  親会社 {done}/{len(parent_codes)} 処理済")

        if i + batch_size < len(parent_codes):
            time.sleep(1)

    if parent_extra_rows:
        upsert_batch("parent_extra", parent_extra_rows, "parent_code")
    log.info(f"  親会社情報更新完了: {len(parent_extra_rows)} 件")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    log.info("=== TOB データ更新開始 (v2) ===")

    # Phase 1: マスター同期
    jpx_df = fetch_jpx_list()
    master_sync(jpx_df)
    codes = jpx_df["code"].tolist()

    # Phase 2: 一括株価更新
    price_data = bulk_download_prices(codes)
    update_prices(codes, price_data)

    # Phase 3: 静的データ補完
    current_month = datetime.now(timezone.utc).month
    is_quarterly = current_month in QUARTERLY_MONTHS
    gap_fill_static(is_quarterly_refresh=is_quarterly)

    # Phase 4: 親会社情報更新
    update_parent_extra()

    elapsed = time.time() - start
    log.info(f"=== TOB データ更新完了 ({elapsed:.0f}秒) ===")


if __name__ == "__main__":
    main()
