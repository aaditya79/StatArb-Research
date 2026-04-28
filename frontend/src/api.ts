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

export interface ProgressEvent {
  stage: string;
  message: string;
  progress: number;
}

export async function runBacktestStream(
  req: BacktestRequest,
  onProgress: (e: ProgressEvent) => void,
): Promise<BacktestResponse> {
  const r = await fetch(`${BASE}/backtest/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* empty */ }
    throw new Error(msg);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let result: BacktestResponse | null = null;
  let errMsg: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE frames are separated by blank lines
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          const obj = JSON.parse(payload);
          if (obj.event === "progress") {
            onProgress({
              stage: obj.stage,
              message: obj.message,
              progress: obj.progress,
            });
          } else if (obj.event === "result") {
            result = obj.data as BacktestResponse;
          } else if (obj.event === "error") {
            errMsg = obj.message ?? "Backtest failed";
          }
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  }

  if (errMsg) throw new Error(errMsg);
  if (!result) throw new Error("Stream ended without a result");
  return result;
}

export const runCointegration = (body: {
  data_source: string; tickers: string[]; start_date: string; end_date: string;
  pvalue_threshold: number; lookback: number;
  pair_t1?: string; pair_t2?: string;
}) => postJson<CointResponse>("/cointegration", body);

export const runGridSearch = (req: BacktestRequest & {
  s_bo_values: number[]; s_so_values: number[];
}) => postJson<GridResponse>("/grid-search", req);
