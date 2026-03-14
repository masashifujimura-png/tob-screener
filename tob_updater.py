"""TOB スクリーニング データ更新スクリプト (v3 - J-Quants API + EDINET)

Phase 1: マスター同期 — J-Quants 銘柄一覧 → tob_stocks
Phase 2: 株価更新   — J-Quants 日次株価 → price_history → 指標算出
Phase 3: 財務更新   — J-Quants 決算サマリー → tob_stocks
Phase 4: 指標算出   — price_history + 財務データ → market_cap / pbr / net_cash_ratio
Phase 5: 親会社情報 — yfinance アクティビスト判定（据置）
Phase 6: EDINET    — 大量保有報告書 → edinet_holders (アクティビスト検出強化)
Phase 7: EDINET    — 有価証券報告書 → キャッシュフロー + 大株主構成
"""

import os
import io
import re
import time
import logging
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import jquantsapi
import yfinance as yf
from supabase import create_client

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", os.environ.get("SUPABASE_KEY"))
JQUANTS_API_KEY = os.environ["JQUANTS_API_KEY"]
EDINET_API_KEY = os.environ.get("EDINET_API_KEY", "")

MAX_WORKERS = 3
RATE_LIMIT_INTERVAL = 15.0  # 秒 (Free tier 5 req/min → 12s間隔、余裕込みで15s)
PRICE_HISTORY_DAYS = 260    # 約1年分の営業日数
FIN_BOOTSTRAP_DAYS = 150    # 初回: 約7ヶ月分（ほぼ全銘柄カバー）
FIN_DAILY_MAX_DAYS = 30     # 日次更新時の最大取得日数
EDINET_API_BASE = "https://api.edinet-fsa.go.jp/api/v2"
EDINET_BOOTSTRAP_DAYS = 90  # 初回: 過去90日分
EDINET_DAILY_DAYS = 3       # 日次: 過去3日分（土日祝を考慮）

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
jquants = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
class RateLimiter:
    """J-Quants API Free tier 用レートリミッター。"""
    def __init__(self, interval: float = RATE_LIMIT_INTERVAL):
        self.interval = interval
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


rate_limiter = RateLimiter()


def jquants_call(fn, *args, max_retries=3, **kwargs):
    """J-Quants API呼び出しのラッパー。429エラー時にリトライ。"""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                log.warning(f"  429 Rate limit, {wait}秒待機... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    return None


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        v = float(value)
        return None if np.isnan(v) else int(v)
    except (TypeError, ValueError):
        return None


def upsert_batch(table: str, rows: list[dict], conflict_key: str,
                  batch_size: int = 500, default_to_null: bool = True):
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table(table).upsert(
            batch, on_conflict=conflict_key, default_to_null=default_to_null,
        ).execute()


def fetch_all_rows(table: str, select: str, filter_fn=None) -> list[dict]:
    """Supabase の1000件制限を回避して全行取得。"""
    all_data = []
    page_size = 1000
    offset = 0
    while True:
        query = supabase.table(table).select(select).range(offset, offset + page_size - 1)
        if filter_fn:
            query = filter_fn(query)
        resp = query.execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < page_size:
            break
        offset += page_size
    return all_data


# ---------------------------------------------------------------------------
# Phase 1: マスター同期 (J-Quants)
# ---------------------------------------------------------------------------
def master_sync() -> dict:
    """J-Quants 銘柄マスターから全銘柄を同期。code_to_info を返す。"""
    log.info("Phase 1: マスター同期 (J-Quants)")
    rate_limiter.wait()
    master = jquants_call(jquants.get_eq_master)

    target_markets = {"プライム", "スタンダード", "グロース"}
    master = master[master["MktNm"].isin(target_markets)].copy()
    master["code"] = master["Code"].str[:4]
    # 同一4桁コードの重複を除去（最初のエントリを採用）
    master = master.drop_duplicates(subset=["code"], keep="first")

    log.info(f"  J-Quants 銘柄一覧: {len(master)} 銘柄")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    code_to_info = {}
    for _, r in master.iterrows():
        code = r["code"]
        name = r["CoName"]
        market = r["MktNm"]
        sector = r.get("S33Nm", "")
        rows.append({
            "code": code,
            "name": name,
            "market": market,
            "sector_33": sector,
            "updated_at": now,
        })
        code_to_info[code] = {"name": name, "market": market, "sector_33": sector}

    upsert_batch("tob_stocks", rows, "code", default_to_null=False)
    log.info(f"  {len(rows)} 銘柄をマスター同期完了")
    return code_to_info


# ---------------------------------------------------------------------------
# Phase 2: 株価更新 (J-Quants → price_history)
# ---------------------------------------------------------------------------
def get_trading_days() -> list[str]:
    """J-Quants カレンダーから営業日一覧を取得 (YYYYMMDD 文字列)。"""
    rate_limiter.wait()
    cal = jquants_call(jquants.get_mkt_calendar)
    biz = cal[cal["HolDiv"].astype(str) == "1"].copy()
    dates = biz["Date"].sort_values().tolist()
    return [d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d).replace("-", "")
            for d in dates]


def get_latest_price_date() -> str | None:
    """price_history テーブルの最新日付 (YYYYMMDD)。"""
    resp = (supabase.table("price_history")
            .select("date")
            .order("date", desc=True)
            .limit(1)
            .execute())
    if resp.data:
        return resp.data[0]["date"].replace("-", "")
    return None


def fetch_and_store_daily_prices(date_str: str) -> int:
    """1日分の全銘柄株価を取得・保存。保存件数を返す。"""
    rate_limiter.wait()
    try:
        prices = jquants_call(jquants.get_eq_bars_daily, date_yyyymmdd=date_str)
    except Exception as e:
        log.warning(f"  株価取得失敗 {date_str}: {e}")
        return 0

    if prices.empty:
        return 0

    rows = []
    for _, r in prices.iterrows():
        code = str(r["Code"])[:4]
        close = safe_float(r.get("AdjC") or r.get("C"))
        volume = safe_int(r.get("AdjVo") or r.get("Vo"))
        if close is None:
            continue
        rows.append({
            "code": code,
            "date": str(r["Date"])[:10],
            "close": close,
            "volume": volume or 0,
        })

    # 同一 code+date の重複を除去
    seen = set()
    deduped = []
    for r in rows:
        key = (r["code"], r["date"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    if deduped:
        upsert_batch("price_history", deduped, "code,date")
    return len(deduped)


def get_oldest_price_date() -> str | None:
    """price_history テーブルの最古日付 (YYYYMMDD)。"""
    resp = (supabase.table("price_history")
            .select("date")
            .order("date")
            .limit(1)
            .execute())
    if resp.data:
        return resp.data[0]["date"].replace("-", "")
    return None


def sync_price_history(trading_days: list[str]):
    """price_history を最新まで同期（過去方向の欠落分も補完）。"""
    log.info("Phase 2: 株価更新 (J-Quants)")

    latest = get_latest_price_date()
    oldest = get_oldest_price_date()

    # 必要な営業日の範囲: 直近 PRICE_HISTORY_DAYS 日分
    target_days = trading_days[-PRICE_HISTORY_DAYS:]

    if latest and oldest:
        # 未来方向の差分 + 過去方向の欠落分を収集
        missing = [d for d in target_days if d > latest or d < oldest]
        if not missing:
            log.info(f"  株価データは最新です ({oldest}〜{latest})")
            return
        log.info(f"  差分/補完取得: {len(missing)} 日分 "
                 f"(DB: {oldest}〜{latest})")
    else:
        missing = target_days
        log.info(f"  初回ブートストラップ: {len(missing)} 日分")

    total_stored = 0
    for i, day in enumerate(missing):
        n = fetch_and_store_daily_prices(day)
        total_stored += n
        if (i + 1) % 10 == 0:
            log.info(f"  株価取得: {i + 1}/{len(missing)} 日完了 (累計 {total_stored} 件)")

    log.info(f"  株価同期完了: {len(missing)} 日分, {total_stored} 件保存")


# ---------------------------------------------------------------------------
# Phase 3: 財務データ更新 (J-Quants)
# ---------------------------------------------------------------------------
def get_latest_fin_date() -> str | None:
    """tob_stocks.static_updated_at の最新日付 (YYYYMMDD)。"""
    resp = (supabase.table("tob_stocks")
            .select("static_updated_at")
            .not_.is_("static_updated_at", "null")
            .order("static_updated_at", desc=True)
            .limit(1)
            .execute())
    if resp.data and resp.data[0].get("static_updated_at"):
        return resp.data[0]["static_updated_at"][:10].replace("-", "")
    return None


def get_fin_coverage() -> int:
    """tob_stocks で財務データ（total_assets）が設定されている銘柄数。"""
    resp = (supabase.table("tob_stocks")
            .select("code", count="exact")
            .not_.is_("total_assets", "null")
            .limit(1)
            .execute())
    return resp.count or 0


def sync_financials(trading_days: list[str]):
    """J-Quants 決算サマリーから財務データを更新。"""
    log.info("Phase 3: 財務データ更新 (J-Quants)")

    latest = get_latest_fin_date()
    coverage = get_fin_coverage()
    total_stocks = len(fetch_all_rows("tob_stocks", "code"))
    coverage_pct = coverage / total_stocks * 100 if total_stocks else 0

    # カバレッジが50%未満ならブートストラップ
    needs_bootstrap = coverage_pct < 50
    if latest and not needs_bootstrap:
        target_days = [d for d in trading_days if d > latest]
        if not target_days:
            log.info(f"  財務データは最新です (カバレッジ: {coverage}/{total_stocks} = {coverage_pct:.0f}%)")
            return
        target_days = target_days[-FIN_DAILY_MAX_DAYS:]
        log.info(f"  差分取得: {len(target_days)} 日分 (最終更新: {latest}, "
                 f"カバレッジ: {coverage}/{total_stocks})")
    else:
        target_days = trading_days[-FIN_BOOTSTRAP_DAYS:]
        log.info(f"  ブートストラップ: {len(target_days)} 日分 "
                 f"(カバレッジ: {coverage}/{total_stocks} = {coverage_pct:.0f}%)")

    all_fins = []
    for i, day in enumerate(target_days):
        rate_limiter.wait()
        try:
            fin = jquants_call(jquants.get_fin_summary, date_yyyymmdd=day)
            if not fin.empty:
                all_fins.append(fin)
        except Exception as e:
            log.warning(f"  財務取得失敗 {day}: {e}")
        if (i + 1) % 10 == 0:
            log.info(f"  財務取得: {i + 1}/{len(target_days)} 日完了")

    if not all_fins:
        log.info("  新規財務データなし")
        return

    fins = pd.concat(all_fins, ignore_index=True)
    fins["code"] = fins["Code"].str[:4]

    # 各銘柄の最新決算のみ残す（FY決算 > 四半期決算を優先）
    # DocType に "FY" が含まれるものを優先的に残す
    fins["is_fy"] = fins["DocType"].str.contains("FY", na=False).astype(int)
    fins = fins.sort_values(["code", "is_fy", "DiscDate"]).drop_duplicates(
        subset=["code"], keep="last"
    )

    # tob_stocks に存在し、name があるコードのみ対象（NOT NULL制約回避）
    existing = fetch_all_rows("tob_stocks", "code, name, market")
    existing_map = {r["code"]: r for r in existing if r.get("name")}
    existing_codes = set(existing_map.keys())

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in fins.iterrows():
        code = r["code"]
        if code not in existing_codes:
            continue

        shares = safe_int(r.get("ShOutFY"))
        if not shares or shares == 0:
            continue

        bps = safe_float(r.get("BPS"))
        eps = safe_float(r.get("EPS"))
        eq_ratio = safe_float(r.get("EqAR"))
        ta = safe_int(r.get("TA"))
        eq = safe_int(r.get("Eq"))
        cash_eq = safe_int(r.get("CashEq"))

        info = existing_map[code]
        row = {
            "code": code,
            "name": info["name"],
            "market": info["market"],
            "shares_outstanding": shares,
            "bps": bps,
            "eps": eps,
            "equity_ratio": eq_ratio,
            "total_assets": ta,
            "equity": eq,
            "cash_equivalents": cash_eq,
            "static_updated_at": now,
        }
        rows.append(row)

    if rows:
        upsert_batch("tob_stocks", rows, "code", default_to_null=False)
    log.info(f"  財務データ更新完了: {len(rows)} 銘柄")


# ---------------------------------------------------------------------------
# Phase 4: 指標算出 (price_history + 財務データ → tob_stocks)
# ---------------------------------------------------------------------------
def fetch_all_rows_large(table: str, select: str, filter_fn=None) -> list[dict]:
    """大量データ用の全行取得（count付きで確実に全件取得）。"""
    # まず総件数を取得
    count_q = supabase.table(table).select(select, count="exact").limit(1)
    if filter_fn:
        count_q = filter_fn(count_q)
    total = count_q.execute().count or 0
    if total == 0:
        return []

    page_size = 1000  # Supabase のデフォルト上限
    all_data = []
    offset = 0
    while offset < total:
        query = supabase.table(table).select(select).range(offset, offset + page_size - 1)
        if filter_fn:
            query = filter_fn(query)
        resp = query.execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        offset += page_size
        if offset % 100000 == 0:
            log.info(f"  データ読み込み中: {len(all_data)}/{total} 件...")
    return all_data


def compute_and_update_metrics(code_to_info: dict):
    """price_history と tob_stocks の財務データから指標を算出して更新。"""
    log.info("Phase 4: 指標算出")

    # 1. price_history から全データを読み込み
    all_prices = fetch_all_rows_large("price_history", "code, date, close, volume")
    if not all_prices:
        log.warning("  price_history にデータがありません")
        return
    log.info(f"  price_history: {len(all_prices)} 件読み込み完了")

    pdf = pd.DataFrame(all_prices)
    pdf["close"] = pd.to_numeric(pdf["close"], errors="coerce")
    pdf["volume"] = pd.to_numeric(pdf["volume"], errors="coerce")
    pdf = pdf.sort_values(["code", "date"])

    # 銘柄ごとに指標を算出
    price_metrics = {}
    for code, grp in pdf.groupby("code"):
        close = grp["close"].dropna()
        volume = grp["volume"].dropna()
        if close.empty:
            continue

        current_price = float(close.iloc[-1])

        volume_ratio = None
        if len(volume) >= 10:
            avg_5d = volume.tail(5).mean()
            avg_60d = volume.tail(60).mean() if len(volume) >= 60 else volume.mean()
            if avg_60d > 0:
                volume_ratio = float(avg_5d / avg_60d)

        price_drop_pct = None
        high_52w = float(close.max())
        if high_52w > 0:
            price_drop_pct = (high_52w - current_price) / high_52w

        price_metrics[code] = {
            "current_price": current_price,
            "volume_ratio": volume_ratio,
            "price_drop_pct": price_drop_pct,
        }

    log.info(f"  価格指標算出: {len(price_metrics)} 銘柄")

    # 2. tob_stocks から財務データ + name/market を読み込み
    fin_data = fetch_all_rows(
        "tob_stocks",
        "code, name, market, shares_outstanding, bps, total_assets, equity, cash_equivalents"
    )
    fin_map = {r["code"]: r for r in fin_data}

    # 3. tob_stocks に存在する銘柄コード一覧を取得
    existing_codes = set(r["code"] for r in fin_data)

    # 4. 統合して更新
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for code, pm in price_metrics.items():
        # tob_stocks に登録済みかつ name がある銘柄のみ対象
        if code not in existing_codes:
            continue
        fin = fin_map.get(code, {})
        if not fin.get("name"):
            continue
        current_price = pm["current_price"]
        shares = fin.get("shares_outstanding")
        bps_val = safe_float(fin.get("bps"))

        # BPS が未取得の場合、equity / shares で算出
        if bps_val is None and shares and shares > 0:
            eq_val = safe_float(fin.get("equity"))
            if eq_val:
                bps_val = eq_val / shares

        market_cap = int(current_price * shares) if shares else None
        # PostgreSQL bigint 上限チェック (bad data 回避)
        if market_cap and market_cap > 9_000_000_000_000_000_000:
            market_cap = None
        pbr = current_price / bps_val if bps_val and bps_val > 0 else None

        # net_cash_ratio = (現金同等物 - 総負債) / 時価総額
        # CashEq が FY データにのみ存在するため、ない場合は自己資本比率で近似
        net_cash_ratio = None
        ta = safe_float(fin.get("total_assets"))
        eq = safe_float(fin.get("equity"))
        cash = safe_float(fin.get("cash_equivalents"))
        if market_cap and ta and eq:
            total_liabilities = ta - eq
            if cash is not None and cash > 0:
                net_cash = cash - total_liabilities
            else:
                # CashEq 不明時: 自己資本 - 負債 の半分を近似値として使用
                # (保守的: 資産の流動性を考慮して割り引き)
                net_cash = eq * 0.5 - total_liabilities * 0.5
            net_cash_ratio = net_cash / market_cap

        row = {
            "code": code,
            "name": fin["name"],
            "market": fin["market"],
            "current_price": safe_float(current_price),
            "volume_ratio": safe_float(pm["volume_ratio"]),
            "price_drop_pct": safe_float(pm["price_drop_pct"]),
            "market_cap": market_cap,
            "pbr": safe_float(pbr),
            "net_cash_ratio": safe_float(net_cash_ratio),
            "updated_at": now,
        }
        rows.append(row)

    if rows:
        upsert_batch("tob_stocks", rows, "code", default_to_null=False)
    log.info(f"  指標更新完了: {len(rows)} 銘柄 "
             f"(market_cap算出: {sum(1 for r in rows if r['market_cap'])})")


# ---------------------------------------------------------------------------
# Phase 5: 親会社情報更新 (yfinance - 据置)
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


def update_parent_extra():
    """親会社の追加情報（PBR・アクティビスト）を更新。"""
    log.info("Phase 5: 親会社情報更新")

    ps_resp = supabase.table("parent_subsidiary").select("parent_code").execute()
    if not ps_resp.data:
        log.info("  親子上場データなし")
        return

    parent_codes = list({r["parent_code"] for r in ps_resp.data})

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
# Phase 6: EDINET 大量保有報告書 (アクティビスト検出強化)
# ---------------------------------------------------------------------------
def edinet_get_documents(date_str: str) -> list[dict]:
    """EDINET API: 指定日の書類一覧取得。date_str は YYYY-MM-DD 形式。"""
    resp = requests.get(
        f"{EDINET_API_BASE}/documents.json",
        params={"date": date_str, "type": 2, "Subscription-Key": EDINET_API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])


def edinet_download_xbrl(doc_id: str) -> bytes | None:
    """EDINET API: XBRL ZIP ダウンロード。"""
    try:
        resp = requests.get(
            f"{EDINET_API_BASE}/documents/{doc_id}",
            params={"type": 1, "Subscription-Key": EDINET_API_KEY},
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        return resp.content
    except Exception as e:
        log.warning(f"  XBRL ダウンロード失敗 {doc_id}: {e}")
        return None


def parse_holder_xbrl(zip_bytes: bytes) -> dict | None:
    """大量保有報告書 XBRL ZIP から保有割合等を抽出。"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            xbrl_files = [f for f in zf.namelist()
                          if f.endswith('.xbrl') and 'PublicDoc' in f]
            if not xbrl_files:
                return None

            with zf.open(xbrl_files[0]) as f:
                tree = ET.parse(f)

            root = tree.getroot()
            result = {}

            # 名前空間に依存しない検索で保有割合を抽出
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

                # 保有割合（変更後 or 初回）
                # PerLastReport は前回の値なので除外
                if ('HoldingRatio' in tag
                        and 'PerLastReport' not in tag
                        and elem.text):
                    try:
                        ratio = float(elem.text.strip())
                        if ratio > 0:
                            # AfterChange (変更後) を優先、なければ最初に見つかった値
                            if 'AfterChange' in tag:
                                result['holding_ratio'] = ratio
                            elif 'holding_ratio' not in result:
                                result['holding_ratio'] = ratio
                    except ValueError:
                        pass

                # 保有目的
                if 'PurposeOfHolding' in tag and elem.text and elem.text.strip():
                    purpose = elem.text.strip()[:500]
                    if len(purpose) > 2:  # 空でない実質的な内容のみ
                        result['purpose'] = purpose

            return result if result else None
    except (zipfile.BadZipFile, ET.ParseError):
        return None
    except Exception as e:
        log.debug(f"  XBRL解析エラー: {e}")
        return None


def check_activist_name(name: str) -> bool:
    """ファイラー名がアクティビストキーワードに該当するか判定。"""
    name_lower = name.lower()
    for kw in ACTIVIST_KEYWORDS:
        if kw.lower() in name_lower:
            return True
    return False


def get_latest_edinet_date() -> str | None:
    """edinet_holders テーブルの最新 report_date (YYYY-MM-DD)。"""
    resp = (supabase.table("edinet_holders")
            .select("report_date")
            .order("report_date", desc=True)
            .limit(1)
            .execute())
    if resp.data and resp.data[0].get("report_date"):
        return resp.data[0]["report_date"]
    return None


def sync_edinet_holders():
    """Phase 6: EDINET 大量保有報告書の同期。"""
    if not EDINET_API_KEY:
        log.info("Phase 6: EDINET_API_KEY 未設定、スキップ")
        return

    log.info("Phase 6: EDINET 大量保有報告書同期")

    # 対象銘柄コード一覧
    stocks = fetch_all_rows("tob_stocks", "code")
    valid_codes = {r["code"] for r in stocks}

    # 既存 doc_id 一覧（重複ダウンロード回避）
    existing_docs = fetch_all_rows("edinet_holders", "doc_id")
    existing_doc_ids = {r["doc_id"] for r in existing_docs}

    # 取得日範囲の決定
    latest = get_latest_edinet_date()
    today = datetime.now(timezone.utc).date()

    if latest:
        # 差分更新: 最終更新日の翌日から
        start_date = datetime.strptime(latest, "%Y-%m-%d").date() + timedelta(days=1)
        if start_date > today:
            log.info("  EDINET データは最新です")
            return
        days_to_fetch = (today - start_date).days + 1
        log.info(f"  差分取得: {start_date} 〜 {today} ({days_to_fetch} 日)")
    else:
        # 初回ブートストラップ
        days_to_fetch = EDINET_BOOTSTRAP_DAYS
        start_date = today - timedelta(days=days_to_fetch - 1)
        log.info(f"  ブートストラップ: 過去 {days_to_fetch} 日分")

    # 日ごとに書類一覧を取得
    total_found = 0
    total_parsed = 0
    rows_to_upsert = []

    for day_offset in range(days_to_fetch):
        target_date = start_date + timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")

        try:
            docs = edinet_get_documents(date_str)
        except Exception as e:
            log.warning(f"  EDINET 書類一覧取得失敗 {date_str}: {e}")
            time.sleep(1)
            continue

        # 大量保有報告書 (350) / 変更報告書 (360) をフィルタ
        holder_docs = [
            d for d in docs
            if d.get("docTypeCode") in ("350", "360")
            and d.get("secCode")
            and d.get("xbrlFlag") == "1"
        ]

        for doc in holder_docs:
            doc_id = doc.get("docID")
            if not doc_id or doc_id in existing_doc_ids:
                continue

            sec_code = doc["secCode"][:4]  # 5桁 → 4桁
            if sec_code not in valid_codes:
                continue

            filer_name = doc.get("filerName", "")
            if not filer_name:
                continue

            total_found += 1

            # XBRL ダウンロード＆解析
            zip_bytes = edinet_download_xbrl(doc_id)
            if not zip_bytes:
                continue

            parsed = parse_holder_xbrl(zip_bytes)
            holding_ratio = parsed.get("holding_ratio") if parsed else None
            purpose = parsed.get("purpose") if parsed else None

            is_activist = check_activist_name(filer_name)
            if purpose:
                # 保有目的にもアクティビスト関連キーワードがないか確認
                purpose_keywords = ["経営改善", "株主提案", "株主価値", "企業価値向上",
                                    "資本効率", "ガバナンス", "支配", "買収"]
                for pk in purpose_keywords:
                    if pk in purpose:
                        is_activist = True
                        break

            row = {
                "doc_id": doc_id,
                "code": sec_code,
                "filer_name": filer_name,
                "holding_ratio": holding_ratio,
                "purpose": purpose,
                "report_date": date_str,
                "doc_type_code": doc["docTypeCode"],
                "is_activist": is_activist,
            }
            rows_to_upsert.append(row)
            existing_doc_ids.add(doc_id)
            total_parsed += 1

            # EDINET に過度な負荷をかけない
            time.sleep(0.5)

        # 日ごとのバッチ保存
        if rows_to_upsert and len(rows_to_upsert) >= 50:
            upsert_batch("edinet_holders", rows_to_upsert, "doc_id")
            log.info(f"  中間保存: {len(rows_to_upsert)} 件")
            rows_to_upsert = []

        if (day_offset + 1) % 10 == 0:
            log.info(f"  EDINET 取得: {day_offset + 1}/{days_to_fetch} 日完了 "
                     f"(発見: {total_found}, 解析: {total_parsed})")

        # API レート配慮
        time.sleep(0.3)

    # 残りを保存
    if rows_to_upsert:
        upsert_batch("edinet_holders", rows_to_upsert, "doc_id")

    log.info(f"  EDINET 同期完了: {total_found} 件発見, {total_parsed} 件解析・保存")


# ---------------------------------------------------------------------------
# Phase 7: EDINET 有価証券報告書 (キャッシュフロー + 株主構成)
# ---------------------------------------------------------------------------
YUHO_BOOTSTRAP_DAYS = 365   # 初回: 過去1年（年次報告をカバー）
YUHO_DAILY_DAYS = 7         # 日次: 過去7日分


def parse_yuho_csv(zip_bytes: bytes) -> dict | None:
    """有価証券報告書 CSV ZIP からキャッシュフロー・株主データを抽出。"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_files = [f for f in zf.namelist()
                         if 'jpcrp' in f and f.endswith('.csv')]
            if not csv_files:
                return None

            with zf.open(csv_files[0]) as f:
                content = f.read().decode('utf-16', errors='ignore')

        lines = content.split('\n')
        result = {
            'cash_equivalents': None,
            'operating_cf': None,
            'foreign_ownership_ratio': None,
            'shareholders': [],   # [{rank, name, ratio, shares}]
        }

        # 株主名とランクの一時格納
        names = {}   # rank -> name
        ratios = {}  # rank -> ratio
        shares = {}  # rank -> shares_held

        for line in lines:
            parts = line.split('\t')
            if len(parts) < 9:
                continue

            elem = parts[0].strip().strip('"')
            context = parts[2].strip().strip('"') if len(parts) > 2 else ''
            value = parts[8].strip().strip('"') if len(parts) > 8 else ''

            if not value or value == '－':
                # 名前は値が空でないことが多いが、数値データが「－」の場合はスキップ
                if 'NameMajorShareholders' not in elem:
                    continue

            # --- キャッシュフロー（Summary でない詳細版を優先）---
            if 'CurrentYear' not in context:
                continue

            # 現金同等物
            if ('jppfs_cor:CashAndCashEquivalents' == elem
                    and 'CurrentYearInstant' in context
                    and 'NonConsolidated' not in context):
                try:
                    result['cash_equivalents'] = int(value)
                except (ValueError, TypeError):
                    pass

            # 営業CF
            if ('jppfs_cor:NetCashProvidedByUsedInOperatingActivities' == elem
                    and 'CurrentYear' in context
                    and 'NonConsolidated' not in context):
                try:
                    result['operating_cf'] = int(value)
                except (ValueError, TypeError):
                    pass

            # 外国人持株比率
            if 'PercentageOfShareholdingsForeigners' in elem and value:
                try:
                    v = float(value)
                    # 複数あれば合算（個人以外 + 個人）
                    if result['foreign_ownership_ratio'] is None:
                        result['foreign_ownership_ratio'] = v
                    else:
                        result['foreign_ownership_ratio'] += v
                except (ValueError, TypeError):
                    pass

            # 大株主名
            if 'NameMajorShareholders' in elem:
                rank_m = re.search(r'No(\d+)', context)
                if rank_m and value and value != '－':
                    names[int(rank_m.group(1))] = value.strip()

            # 大株主保有比率
            if 'jpcrp_cor:ShareholdingRatio' == elem and 'MajorShareholder' in context:
                rank_m = re.search(r'No(\d+)', context)
                if rank_m and value:
                    try:
                        ratios[int(rank_m.group(1))] = float(value)
                    except (ValueError, TypeError):
                        pass

            # 大株主株数
            if 'jpcrp_cor:NumberOfSharesHeld' == elem and 'MajorShareholder' in context:
                rank_m = re.search(r'No(\d+)', context)
                if rank_m and value:
                    try:
                        shares[int(rank_m.group(1))] = int(value)
                    except (ValueError, TypeError):
                        pass

        # 株主データを統合
        for rank in sorted(set(names.keys()) | set(ratios.keys())):
            if rank not in names:
                continue
            sh = {
                'rank': rank,
                'name': names[rank],
                'ratio': ratios.get(rank),
                'shares': shares.get(rank),
            }
            result['shareholders'].append(sh)

        return result
    except (zipfile.BadZipFile, UnicodeDecodeError):
        return None
    except Exception as e:
        log.debug(f"  有報CSV解析エラー: {e}")
        return None


def get_latest_yuho_date() -> str | None:
    """tob_stocks.yuho_date の最新日付。"""
    resp = (supabase.table("tob_stocks")
            .select("yuho_date")
            .not_.is_("yuho_date", "null")
            .order("yuho_date", desc=True)
            .limit(1)
            .execute())
    if resp.data and resp.data[0].get("yuho_date"):
        return resp.data[0]["yuho_date"]
    return None


def sync_edinet_yuho():
    """Phase 7: EDINET 有価証券報告書の同期（キャッシュフロー + 株主構成）。"""
    if not EDINET_API_KEY:
        log.info("Phase 7: EDINET_API_KEY 未設定、スキップ")
        return

    log.info("Phase 7: EDINET 有価証券報告書同期（CF + 株主構成）")

    # 対象銘柄コード一覧
    stocks = fetch_all_rows("tob_stocks", "code, name, market")
    valid_codes = {r["code"]: r for r in stocks if r.get("name")}

    # 既にyuho_dateがある銘柄は処理済み（日次更新時にスキップ判定用）
    latest_yuho = get_latest_yuho_date()
    today = datetime.now(timezone.utc).date()

    if latest_yuho:
        # 差分更新: 最終更新日の翌日から
        start_date = datetime.strptime(latest_yuho, "%Y-%m-%d").date() + timedelta(days=1)
        if start_date > today:
            log.info("  有報データは最新です")
            return
        days_to_fetch = min((today - start_date).days + 1, YUHO_DAILY_DAYS)
        log.info(f"  差分取得: {start_date} 〜 (最大 {days_to_fetch} 日)")
    else:
        days_to_fetch = YUHO_BOOTSTRAP_DAYS
        start_date = today - timedelta(days=days_to_fetch - 1)
        log.info(f"  ブートストラップ: 過去 {days_to_fetch} 日分")

    # 処理済み doc_id 追跡（銘柄ごとに最新1件のみ処理）
    processed_codes = set()
    total_found = 0
    total_parsed = 0

    # 新しい日付から逆順に処理（最新の有報を優先）
    for day_offset in range(days_to_fetch - 1, -1, -1):
        target_date = start_date + timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")

        try:
            docs = edinet_get_documents(date_str)
        except Exception as e:
            log.warning(f"  EDINET 書類一覧取得失敗 {date_str}: {e}")
            time.sleep(1)
            continue

        # 有価証券報告書 (120) でセキュリティコードありのみ
        yuho_docs = [
            d for d in docs
            if d.get("docTypeCode") == "120"
            and d.get("secCode")
            and d.get("csvFlag") == "1"
        ]

        for doc in yuho_docs:
            sec_code = doc["secCode"][:4]
            if sec_code not in valid_codes or sec_code in processed_codes:
                continue

            total_found += 1

            # CSV ダウンロード
            try:
                csv_resp = requests.get(
                    f"{EDINET_API_BASE}/documents/{doc['docID']}",
                    params={"type": 5, "Subscription-Key": EDINET_API_KEY},
                    timeout=120,
                )
                if csv_resp.status_code != 200:
                    continue
            except Exception as e:
                log.warning(f"  CSV ダウンロード失敗 {doc['docID']}: {e}")
                continue

            parsed = parse_yuho_csv(csv_resp.content)
            if not parsed:
                time.sleep(0.5)
                continue

            info = valid_codes[sec_code]
            now = datetime.now(timezone.utc).isoformat()

            # tob_stocks 更新（キャッシュフロー + 外国人持株比率）
            update_row = {
                "code": sec_code,
                "name": info["name"],
                "market": info["market"],
                "yuho_date": date_str,
            }
            if parsed['cash_equivalents'] is not None:
                update_row['cash_equivalents'] = parsed['cash_equivalents']
            if parsed['operating_cf'] is not None:
                update_row['operating_cf'] = parsed['operating_cf']
            if parsed['foreign_ownership_ratio'] is not None:
                update_row['foreign_ownership_ratio'] = parsed['foreign_ownership_ratio']

            upsert_batch("tob_stocks", [update_row], "code", default_to_null=False)

            # 大株主データ保存
            if parsed['shareholders']:
                sh_rows = []
                for sh in parsed['shareholders']:
                    is_activist = check_activist_name(sh['name'])
                    sh_rows.append({
                        "code": sec_code,
                        "rank": sh['rank'],
                        "shareholder_name": sh['name'],
                        "holding_ratio": sh.get('ratio'),
                        "shares_held": sh.get('shares'),
                        "report_date": date_str,
                        "is_activist": is_activist,
                    })
                upsert_batch("edinet_shareholders", sh_rows, "code,rank")

            processed_codes.add(sec_code)
            total_parsed += 1

            time.sleep(0.5)

        if (days_to_fetch - day_offset) % 30 == 0:
            log.info(f"  有報取得: {days_to_fetch - day_offset}/{days_to_fetch} 日完了 "
                     f"(発見: {total_found}, 解析: {total_parsed})")

        time.sleep(0.3)

    log.info(f"  有報同期完了: {total_found} 件発見, {total_parsed} 件解析・保存 "
             f"(対象銘柄: {len(processed_codes)})")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    log.info("=== TOB データ更新開始 (v3 - J-Quants) ===")

    # Phase 1: マスター同期
    code_to_info = master_sync()

    # 営業日カレンダー取得
    trading_days = get_trading_days()
    log.info(f"  営業日カレンダー: {len(trading_days)} 日 "
             f"({trading_days[0]}〜{trading_days[-1]})")

    # Phase 2: 株価更新
    sync_price_history(trading_days)

    # Phase 3: 財務データ更新
    sync_financials(trading_days)

    # Phase 4: 指標算出
    compute_and_update_metrics(code_to_info)

    # Phase 5: 親会社情報更新
    update_parent_extra()

    # Phase 6: EDINET 大量保有報告書
    sync_edinet_holders()

    # Phase 7: EDINET 有価証券報告書
    sync_edinet_yuho()

    elapsed = time.time() - start
    log.info(f"=== TOB データ更新完了 ({elapsed:.0f}秒) ===")


if __name__ == "__main__":
    main()
