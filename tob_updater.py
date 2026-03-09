"""TOB スクリーニング データ更新スクリプト (Cron Job 用)

JPX 上場企業一覧を取得し、yfinance で株価・財務データを収集して
Supabase に書き込む。歯抜け銘柄は最大3巡リトライする。
"""

import os
import sys
import time
import logging
from datetime import datetime
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
MAX_RETRY_ROUNDS = 3
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

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

# ---------------------------------------------------------------------------
# Supabase クライアント
# ---------------------------------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# JPX 上場企業一覧
# ---------------------------------------------------------------------------
def fetch_jpx_list() -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# 株価一括取得
# ---------------------------------------------------------------------------
def bulk_download_prices(codes: list[str]) -> dict:
    tickers = [f"{c}.T" for c in codes]
    price_data = {}
    batch_size = 500

    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        batch_codes = codes[i:i + batch_size]
        log.info(f"株価取得中… {min(i + batch_size, len(tickers))}/{len(tickers)}")

        try:
            data = yf.download(
                batch_tickers, period="1y",
                group_by="ticker", threads=True, progress=False,
            )
            if data.empty:
                continue

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
        except Exception:
            continue

        if i + batch_size < len(tickers):
            time.sleep(1)

    return price_data


# ---------------------------------------------------------------------------
# 個別財務データ取得
# ---------------------------------------------------------------------------
def fetch_single_financials(code: str, max_retries: int = 2) -> dict | None:
    ticker_str = f"{code}.T"
    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(1)
        try:
            tk = yf.Ticker(ticker_str)
            info = tk.info
            market_cap = info.get("marketCap")
            if not market_cap or market_cap == 0:
                try:
                    market_cap = tk.fast_info.get("marketCap", None)
                except Exception:
                    pass
            if not market_cap or market_cap == 0:
                return None

            pbr = info.get("priceToBook")

            free_float_ratio = None
            float_shares = info.get("floatShares")
            shares_out = info.get("sharesOutstanding")
            if float_shares and shares_out and shares_out > 0:
                free_float_ratio = float_shares / shares_out

            bs = tk.balance_sheet
            net_cash_ratio = None
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                cash = latest.get("Cash And Cash Equivalents", 0) or 0
                short_inv = latest.get("Other Short Term Investments", 0) or 0
                long_inv = latest.get("Long Term Equity Investment", 0) or 0
                securities = short_inv + long_inv
                total_liabilities = latest.get("Total Liabilities Net Minority Interest", 0) or 0
                net_cash = cash + securities * 0.7 - total_liabilities
                net_cash_ratio = net_cash / market_cap

            return {
                "code": code,
                "market_cap": market_cap,
                "pbr": pbr,
                "net_cash_ratio": net_cash_ratio,
                "free_float_ratio": free_float_ratio,
            }
        except Exception:
            if attempt == max_retries:
                return None
    return None


# ---------------------------------------------------------------------------
# アクティビスト判定
# ---------------------------------------------------------------------------
def check_activist_holders(code: str) -> tuple[bool, list[str]]:
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


# ---------------------------------------------------------------------------
# tob_stocks テーブルへ upsert
# ---------------------------------------------------------------------------
def upsert_stocks(records: list[dict]):
    if not records:
        return
    now = datetime.utcnow().isoformat()
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        rows = []
        for r in batch:
            rows.append({
                "code": r["code"],
                "name": r["name"],
                "market": r["market"],
                "market_cap": r.get("market_cap"),
                "pbr": float(r["pbr"]) if r.get("pbr") is not None and not (isinstance(r.get("pbr"), float) and np.isnan(r["pbr"])) else None,
                "net_cash_ratio": float(r["net_cash_ratio"]) if r.get("net_cash_ratio") is not None and not (isinstance(r.get("net_cash_ratio"), float) and np.isnan(r["net_cash_ratio"])) else None,
                "free_float_ratio": float(r["free_float_ratio"]) if r.get("free_float_ratio") is not None and not (isinstance(r.get("free_float_ratio"), float) and np.isnan(r["free_float_ratio"])) else None,
                "volume_ratio": float(r["volume_ratio"]) if r.get("volume_ratio") is not None and not (isinstance(r.get("volume_ratio"), float) and np.isnan(r["volume_ratio"])) else None,
                "price_drop_pct": float(r["price_drop_pct"]) if r.get("price_drop_pct") is not None and not (isinstance(r.get("price_drop_pct"), float) and np.isnan(r["price_drop_pct"])) else None,
                "current_price": float(r["current_price"]) if r.get("current_price") is not None and not (isinstance(r.get("current_price"), float) and np.isnan(r["current_price"])) else None,
                "updated_at": now,
            })
        supabase.table("tob_stocks").upsert(rows, on_conflict="code").execute()
    log.info(f"tob_stocks に {len(records)} 件 upsert 完了")


# ---------------------------------------------------------------------------
# parent_extra テーブルへ upsert
# ---------------------------------------------------------------------------
def upsert_parent_extra(parent_extra: dict):
    if not parent_extra:
        return
    now = datetime.utcnow().isoformat()
    rows = []
    for pc, info in parent_extra.items():
        rows.append({
            "parent_code": pc,
            "parent_pbr": float(info["parent_pbr"]) if info.get("parent_pbr") is not None else None,
            "activist_in_parent": info.get("activist_in_parent", False),
            "activist_names": info.get("activist_names", ""),
            "updated_at": now,
        })
    batch_size = 500
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table("parent_extra").upsert(batch, on_conflict="parent_code").execute()
    log.info(f"parent_extra に {len(rows)} 件 upsert 完了")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    log.info("=== TOB データ更新開始 ===")

    # 1. JPX 一覧取得
    jpx_df = fetch_jpx_list()
    codes = jpx_df["code"].tolist()
    code_to_info = {r["code"]: r for _, r in jpx_df.iterrows()}

    # 2. 株価一括取得
    log.info("Phase 1: 株価一括取得")
    price_data = bulk_download_prices(codes)
    log.info(f"株価取得完了: {len(price_data)}/{len(codes)} 銘柄")

    # 3. 財務データ取得 (リトライ付き)
    financials = {}  # code -> dict

    for round_num in range(1, MAX_RETRY_ROUNDS + 1):
        pending = [c for c in codes if c not in financials]
        if not pending:
            break
        log.info(f"Phase 2 (Round {round_num}/{MAX_RETRY_ROUNDS}): 財務取得 残り {len(pending)} 銘柄")

        done = 0
        batch_size = 50
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_single_financials, c): c for c in batch}
                for future in as_completed(futures):
                    done += 1
                    code = futures[future]
                    result = future.result()
                    if result is not None:
                        financials[code] = result
                    if done % 100 == 0:
                        log.info(f"  {done}/{len(pending)} 処理済 (成功: {len(financials)})")
            if i + batch_size < len(pending):
                time.sleep(2)

        log.info(f"Round {round_num} 完了: {len(financials)}/{len(codes)} 銘柄取得成功")

    # 4. レコード統合 & upsert
    records = []
    for code in codes:
        info = code_to_info[code]
        fin = financials.get(code, {})
        price = price_data.get(code, {})
        if not fin:
            continue
        records.append({
            "code": code,
            "name": info["name"],
            "market": info["market"],
            "market_cap": fin.get("market_cap"),
            "pbr": fin.get("pbr"),
            "net_cash_ratio": fin.get("net_cash_ratio"),
            "free_float_ratio": fin.get("free_float_ratio"),
            "volume_ratio": price.get("volume_ratio"),
            "price_drop_pct": price.get("price_drop_pct"),
            "current_price": price.get("current_price"),
        })

    log.info(f"upsert 対象: {len(records)} 銘柄")
    upsert_stocks(records)

    # 5. 親会社追加情報
    log.info("Phase 3: 親会社アクティビスト情報取得")
    ps_resp = supabase.table("parent_subsidiary").select("parent_code").execute()
    parent_codes = list({r["parent_code"] for r in ps_resp.data})

    # 親会社 PBR を取得済みデータから引く
    code_to_pbr = {r["code"]: r.get("pbr") for r in records}

    parent_extra = {}
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
                parent_extra[pc] = {
                    "parent_pbr": code_to_pbr.get(pc),
                    "activist_in_parent": has_activist,
                    "activist_names": ", ".join(activist_names) if activist_names else "",
                }
                if done % 20 == 0:
                    log.info(f"  親会社 {done}/{len(parent_codes)} 処理済")
        if i + batch_size < len(parent_codes):
            time.sleep(1)

    upsert_parent_extra(parent_extra)

    elapsed = time.time() - start
    log.info(f"=== TOB データ更新完了 ({elapsed:.0f}秒) ===")
    log.info(f"  銘柄数: {len(records)}/{len(codes)}, 歯抜け: {len(codes) - len(records)}")


if __name__ == "__main__":
    main()
