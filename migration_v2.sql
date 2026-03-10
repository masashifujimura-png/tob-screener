-- migration_v2.sql: 静的データ分離のためのカラム追加
-- Supabase SQL Editor で実行

ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS shares_outstanding bigint;
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS bps numeric;              -- Book Value Per Share
ALTER TABLE tob_stocks ADD COLUMN IF NOT EXISTS static_updated_at timestamptz;  -- 静的データ最終取得日
