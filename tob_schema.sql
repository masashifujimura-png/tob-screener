-- Supabase SQL Editor で実行するスキーマ（TOBスクリーニング用）

-- tob_stocks: メインの銘柄データ（cronジョブが書き込み、Webアプリが読み取り）
create table tob_stocks (
  code text primary key,
  name text not null,
  market text not null,
  market_cap bigint,
  pbr numeric,
  net_cash_ratio numeric,
  free_float_ratio numeric,
  volume_ratio numeric,
  price_drop_pct numeric,
  current_price numeric,
  shares_outstanding bigint,
  bps numeric,                          -- Book Value Per Share
  updated_at timestamptz default now(),
  static_updated_at timestamptz         -- 静的データ最終取得日
);

-- parent_subsidiary: 親子上場データ（CSVの代わり）
create table parent_subsidiary (
  id serial primary key,
  parent_code text not null,
  parent_name text not null,
  child_code text not null,
  child_name text not null,
  holding_pct numeric,
  unique(parent_code, child_code)
);

-- parent_extra: 親会社の追加情報（PBR・アクティビスト）
create table parent_extra (
  parent_code text primary key,
  parent_pbr numeric,
  activist_in_parent boolean default false,
  activist_names text default '',
  updated_at timestamptz default now()
);

-- RLS: cronジョブはservice_roleキーで書き込み、Webアプリはanonキーで読み取り
alter table tob_stocks enable row level security;
alter table parent_subsidiary enable row level security;
alter table parent_extra enable row level security;

create policy "Anyone can read tob_stocks" on tob_stocks for select using (true);
create policy "Anyone can read parent_subsidiary" on parent_subsidiary for select using (true);
create policy "Anyone can read parent_extra" on parent_extra for select using (true);

-- インデックス
create index idx_tob_stocks_market on tob_stocks(market);
create index idx_ps_child on parent_subsidiary(child_code);
create index idx_ps_parent on parent_subsidiary(parent_code);
