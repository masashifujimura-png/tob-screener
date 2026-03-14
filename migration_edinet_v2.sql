-- migration_edinet_v2.sql: 有価証券報告書データ用テーブル・カラム追加
-- Supabase SQL Editor で実行

-- 1. tob_stocks に新カラム追加
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS operating_cf BIGINT;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS foreign_ownership_ratio REAL;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS yuho_date DATE;

-- 2. 大株主テーブル（有価証券報告書の大株主上位10）
CREATE TABLE IF NOT EXISTS edinet_shareholders (
    code TEXT NOT NULL,
    rank INT NOT NULL,
    shareholder_name TEXT NOT NULL,
    holding_ratio REAL,
    shares_held BIGINT,
    report_date DATE NOT NULL,
    is_activist BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (code, rank)
);

CREATE INDEX IF NOT EXISTS idx_edinet_sh_code ON edinet_shareholders(code);

-- RLS
ALTER TABLE edinet_shareholders ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read edinet_shareholders" ON edinet_shareholders FOR SELECT USING (true);
CREATE POLICY "Service can write edinet_shareholders" ON edinet_shareholders
    FOR ALL USING (true) WITH CHECK (true);
