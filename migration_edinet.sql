-- migration_edinet.sql: EDINET 大量保有報告書テーブル追加
-- Supabase SQL Editor で実行

-- 1. 大量保有報告書データ
CREATE TABLE IF NOT EXISTS edinet_holders (
    doc_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    filer_name TEXT NOT NULL,
    holding_ratio REAL,
    purpose TEXT,
    report_date DATE NOT NULL,
    doc_type_code TEXT NOT NULL,
    is_activist BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_edinet_holders_code ON edinet_holders(code);
CREATE INDEX IF NOT EXISTS idx_edinet_holders_date ON edinet_holders(report_date DESC);

-- RLS
ALTER TABLE edinet_holders ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read edinet_holders" ON edinet_holders FOR SELECT USING (true);

-- Service role can insert/update
CREATE POLICY "Service can write edinet_holders" ON edinet_holders
    FOR ALL USING (true) WITH CHECK (true);
