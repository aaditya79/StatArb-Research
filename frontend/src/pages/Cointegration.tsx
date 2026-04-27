import { useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from "recharts";
import { runCointegration } from "../api";
import type { BacktestRequest, CointResponse } from "../types";

export default function CointegrationPage({ baseReq }: { baseReq: BacktestRequest | null }) {
  const [pvalue, setPvalue] = useState(0.05);
  const [lookback, setLookback] = useState(252);
  const [data, setData] = useState<CointResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pair, setPair] = useState<{ t1: string; t2: string } | null>(null);

  if (!baseReq) {
    return (
      <div className="card py-16 text-center">
        <p className="text-sm text-slate-500">
          Set a universe in the Backtest tab first — cointegration uses the same tickers and date range.
        </p>
      </div>
    );
  }

  async function run(t1?: string, t2?: string) {
    setBusy(true); setErr(null);
    try {
      const res = await runCointegration({
        data_source: baseReq!.data_source,
        tickers: baseReq!.tickers,
        start_date: baseReq!.start_date,
        end_date: baseReq!.end_date,
        pvalue_threshold: pvalue,
        lookback,
        pair_t1: t1, pair_t2: t2,
      });
      setData(res);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="space-y-6">
      <section className="card card-pad">
        <div className="flex items-end gap-4 flex-wrap">
          <div className="flex-1 min-w-[160px]">
            <label className="field">p-value threshold</label>
            <input type="number" step={0.01} min={0.01} max={0.20}
              value={pvalue} onChange={(e) => setPvalue(parseFloat(e.target.value))}
              className="w-full font-mono" />
          </div>
          <div className="flex-1 min-w-[160px]">
            <label className="field">Lookback (days)</label>
            <input type="number" step={1} min={60} max={504}
              value={lookback} onChange={(e) => setLookback(parseInt(e.target.value))}
              className="w-full font-mono" />
          </div>
          <div className="flex-1 min-w-[200px] text-sm text-slate-500">
            <div><span className="font-semibold text-navy-900">{baseReq.tickers.length}</span> tickers</div>
            <div className="font-mono text-xs">{baseReq.start_date} → {baseReq.end_date}</div>
          </div>
          <button className="btn-primary"
            disabled={busy} onClick={() => { setPair(null); run(); }}>
            {busy ? "Testing…" : "Run cointegration test"}
          </button>
        </div>
        {err && <p className="text-sm text-red-700 mt-3">Error: {err}</p>}
      </section>

      {data && (
        <>
          <section className="card">
            <div className="px-6 pt-5 pb-3 flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-navy-950">Cointegrated pairs</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  {data.pairs.length} pairs · sorted by p-value · click a row to view spread
                </p>
              </div>
            </div>
            <div className="px-6 pb-5">
              <div className="max-h-[28rem] overflow-y-auto rounded-xl border border-slate-200">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-canvas/95 backdrop-blur">
                    <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                      <Th>Pair</Th>
                      <Th right>p-value</Th>
                      <Th right>Score</Th>
                      <Th right>Hedge ratio</Th>
                      <Th right>Spread σ</Th>
                      <Th right>Half-life (days)</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.pairs.map((p) => {
                      const sel = pair && pair.t1 === p.ticker1 && pair.t2 === p.ticker2;
                      return (
                        <tr key={`${p.ticker1}_${p.ticker2}`}
                          className={"border-t border-slate-100 cursor-pointer hover:bg-navy-50/50 "
                            + (sel ? "bg-navy-50/70" : "")}
                          onClick={() => {
                            setPair({ t1: p.ticker1, t2: p.ticker2 });
                            run(p.ticker1, p.ticker2);
                          }}>
                          <Td><span className="font-mono font-semibold text-navy-900">{p.ticker1} / {p.ticker2}</span></Td>
                          <Td right>{p.pvalue.toExponential(2)}</Td>
                          <Td right>{p.score.toFixed(2)}</Td>
                          <Td right>{p.hedge_ratio.toFixed(3)}</Td>
                          <Td right>{p.spread_std.toFixed(4)}</Td>
                          <Td right>{p.half_life === null ? "—" : p.half_life.toFixed(1)}</Td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          {pair && data.spread.length > 0 && (
            <section className="card">
              <div className="px-6 pt-5 pb-3">
                <h3 className="font-semibold text-navy-950">{pair.t1} / {pair.t2} — spread (z-score)</h3>
                <p className="text-xs text-slate-500 mt-0.5">Reference lines at ±2σ</p>
              </div>
              <div className="px-6 pb-5">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={data.spread} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#e2e8f0" vertical={false} />
                    <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} minTickGap={40} />
                    <YAxis stroke="#94a3b8" fontSize={11} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#cbd5e1" />
                    <ReferenceLine y={2} stroke="#dc2626" strokeDasharray="3 3" />
                    <ReferenceLine y={-2} stroke="#dc2626" strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="z" stroke="#1e3a73" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={"px-4 py-2.5 font-semibold " + (right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <td className={"px-4 py-2 " + (right ? "text-right font-mono tabular-nums" : "")}>{children}</td>;
}
