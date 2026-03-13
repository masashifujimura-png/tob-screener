-- migration_v3.sql: J-Quants API 対応 + price_history テーブル追加
-- Supabase SQL Editor で実行

-- 1. price_history テーブル（日次株価の蓄積）
CREATE TABLE IF NOT EXISTS price_history (
    code TEXT NOT NULL,
    date DATE NOT NULL,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (code, date)
);

CREATE INDEX IF NOT EXISTS idx_ph_date ON price_history(date);

ALTER TABLE price_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read price_history" ON price_history FOR SELECT USING (true);

-- 2. tob_stocks に新カラム追加 (J-Quants 由来データ)
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS sector_33 TEXT DEFAULT '';
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS eps NUMERIC;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS equity_ratio NUMERIC;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS total_assets BIGINT;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS equity BIGINT;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS cash_equivalents BIGINT;
