import type {
  BacktestRequest, BacktestResponse, DefaultsResponse,
  CointResponse, GridResponse,
} from "./types";

const BASE = "/api";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* empty */ }
    throw new Error(msg);
  }
  return r.json();
}

export async function fetchDefaults(): Promise<DefaultsResponse> {
  const r = await fetch(`${BASE}/config/defaults`);
  if (!r.ok) throw new Error(`defaults: HTTP ${r.status}`);
  return r.json();
}

export const runBacktest = (req: BacktestRequest) =>
  postJson<BacktestResponse>("/backtest", req);

export const runCointegration = (body: {
  data_source: string; tickers: string[]; start_date: string; end_date: string;
  pvalue_threshold: number; lookback: number;
  pair_t1?: string; pair_t2?: string;
}) => postJson<CointResponse>("/cointegration", body);

export const runGridSearch = (req: BacktestRequest & {
  s_bo_values: number[]; s_so_values: number[];
}) => postJson<GridResponse>("/grid-search", req);
