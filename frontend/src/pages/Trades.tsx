import { useMemo, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Cell,
} from "recharts";
import type { BacktestResponse, Trade } from "../types";

export default function TradesPage({ r }: { r: BacktestResponse | null }) {
  if (!r) return <NeedBacktest />;
  const trades = r.trades || [];

  const tickerStats = useMemo(() => byTicker(trades), [trades]);
  const pnlBuckets = useMemo(() => bucketPnL(trades), [trades]);

  const winners = trades.filter((t) => t.pnl > 0).length;
  const losers = trades.filter((t) => t.pnl <= 0).length;
  const grossWins = trades.filter((t) => t.pnl > 0).reduce((a, t) => a + t.pnl, 0);
  const grossLosses = Math.abs(trades.filter((t) => t.pnl <= 0).reduce((a, t) => a + t.pnl, 0));
  const avgPnL = trades.length ? trades.reduce((a, t) => a + t.pnl, 0) / trades.length : 0;

  const [filter, setFilter] = useState<"all" | "winners" | "losers">("all");
  const [tk, setTk] = useState<string>("");
  const filtered = trades.filter((t) =>
    (filter === "all" || (filter === "winners" ? t.pnl > 0 : t.pnl <= 0))
    && (!tk || t.ticker === tk)
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat label="# Trades" v={trades.length.toLocaleString()} />
        <Stat label="Winners" v={winners.toLocaleString()} tone="pos" />
        <Stat label="Losers" v={losers.toLocaleString()} tone="neg" />
        <Stat label="Avg PnL" v={fmtCash(avgPnL)} tone={avgPnL >= 0 ? "pos" : "neg"} />
        <Stat label="Profit factor" v={(grossLosses > 0 ? grossWins / grossLosses : 0).toFixed(2)} />
      </div>

      <Card title="PnL distribution" subtitle="Histogram of realized trade PnL">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={pnlBuckets} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
            <YAxis stroke="#94a3b8" fontSize={11} />
            <Tooltip cursor={{ fill: "rgba(30,58,115,0.06)" }} />
            <ReferenceLine y={0} stroke="#cbd5e1" />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {pnlBuckets.map((b, i) => (
                <Cell key={i} fill={b.mid < 0 ? "#dc2626" : "#1e3a73"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-ticker breakdown" subtitle="Trade count, total PnL, win rate by symbol">
        <div className="max-h-96 overflow-y-auto rounded-xl border border-slate-200">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-canvas/95 backdrop-blur">
              <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                <Th>Ticker</Th>
                <Th right># Trades</Th>
                <Th right>Total PnL</Th>
                <Th right>Avg PnL</Th>
                <Th right>Win rate</Th>
              </tr>
            </thead>
            <tbody>
              {tickerStats.map((s) => (
                <tr key={s.ticker}
                  className={"border-t border-slate-100 hover:bg-navy-50/50 cursor-pointer "
                    + (tk === s.ticker ? "bg-navy-50/70" : "")}
                  onClick={() => setTk(tk === s.ticker ? "" : s.ticker)}
                >
                  <Td><span className="font-mono font-semibold text-navy-900">{s.ticker}</span></Td>
                  <Td right>{s.n}</Td>
                  <Td right tone={s.total_pnl >= 0 ? "pos" : "neg"}>{fmtCash(s.total_pnl)}</Td>
                  <Td right tone={s.avg_pnl >= 0 ? "pos" : "neg"}>{fmtCash(s.avg_pnl)}</Td>
                  <Td right>{(s.win_rate * 100).toFixed(0)}%</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Trade log"
        subtitle={`${filtered.length} trades${tk ? ` · ${tk}` : ""}`}
        right={
          <div className="flex items-center gap-2">
            {(["all", "winners", "losers"] as const).map((f) => (
              <button key={f}
                onClick={() => setFilter(f)}
                className={
                  "px-3 py-1.5 rounded-lg text-xs font-semibold transition " +
                  (filter === f
                    ? "bg-navy-800 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200")
                }
              >
                {f}
              </button>
            ))}
            {tk && (
              <button className="btn-ghost" onClick={() => setTk("")}>clear {tk}</button>
            )}
          </div>
        }
      >
        <div className="max-h-[28rem] overflow-y-auto rounded-xl border border-slate-200">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-canvas/95 backdrop-blur">
              <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                <Th>Ticker</Th>
                <Th>Direction</Th>
                <Th>Entry</Th>
                <Th>Exit</Th>
                <Th right>Entry $</Th>
                <Th right>Exit $</Th>
                <Th right>PnL</Th>
                <Th right>Notional</Th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(-2000).reverse().map((t, i) => (
                <tr key={i} className="border-t border-slate-100">
                  <Td><span className="font-mono font-semibold text-navy-900">{t.ticker}</span></Td>
                  <Td>
                    <span className={
                      "inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold border " +
                      (t.direction === 1
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                        : "bg-red-50 text-red-700 border-red-200")
                    }>
                      {t.direction === 1 ? "LONG" : "SHORT"}
                    </span>
                  </Td>
                  <Td><span className="font-mono text-slate-600">{t.entry_date}</span></Td>
                  <Td><span className="font-mono text-slate-600">{t.exit_date}</span></Td>
                  <Td right><span className="font-mono">${t.entry_price.toFixed(2)}</span></Td>
                  <Td right><span className="font-mono">${t.exit_price.toFixed(2)}</span></Td>
                  <Td right tone={t.pnl >= 0 ? "pos" : "neg"}>{fmtCash(t.pnl)}</Td>
                  <Td right><span className="font-mono">{fmtCash(t.notional)}</span></Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function byTicker(trades: Trade[]) {
  const byT: Record<string, Trade[]> = {};
  trades.forEach((t) => { (byT[t.ticker] ??= []).push(t); });
  return Object.entries(byT).map(([ticker, ts]) => {
    const total_pnl = ts.reduce((a, t) => a + t.pnl, 0);
    const win = ts.filter((t) => t.pnl > 0).length;
    return {
      ticker, n: ts.length, total_pnl,
      avg_pnl: total_pnl / ts.length,
      win_rate: ts.length ? win / ts.length : 0,
    };
  }).sort((a, b) => b.total_pnl - a.total_pnl);
}

function bucketPnL(trades: Trade[]) {
  if (!trades.length) return [];
  const pnls = trades.map((t) => t.pnl);
  const lo = Math.min(...pnls), hi = Math.max(...pnls);
  const NB = 24;
  const w = (hi - lo) / NB || 1;
  const buckets: { label: string; mid: number; count: number }[] = [];
  for (let i = 0; i < NB; i++) {
    const a = lo + i * w, b = a + w;
    const c = pnls.filter((x) => x >= a && (i === NB - 1 ? x <= b : x < b)).length;
    const mid = (a + b) / 2;
    buckets.push({ label: fmtCashShort(mid), mid, count: c });
  }
  return buckets;
}

function fmtCash(v: number) {
  const s = v < 0 ? "-" : "";
  return s + "$" + Math.abs(Math.round(v)).toLocaleString();
}
function fmtCashShort(v: number) {
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(1)}k`;
  return `$${v.toFixed(0)}`;
}

function Stat({ label, v, tone }: { label: string; v: string; tone?: "pos" | "neg" }) {
  return (
    <div className="card card-pad !p-4">
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
      <div className={
        "text-2xl font-semibold mt-1 font-mono tabular-nums " +
        (tone === "pos" ? "text-emerald-700" : tone === "neg" ? "text-red-700" : "text-navy-900")
      }>{v}</div>
    </div>
  );
}

function Card({
  title, subtitle, right, children,
}: { title: string; subtitle?: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card">
      <div className="px-6 pt-5 pb-3 flex items-center justify-between gap-4">
        <div>
          <h3 className="font-semibold text-navy-950">{title}</h3>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
        {right}
      </div>
      <div className="px-6 pb-5">{children}</div>
    </section>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th className={"px-4 py-2.5 font-semibold " + (right ? "text-right" : "text-left")}>{children}</th>
  );
}

function Td({ children, right, tone }: {
  children: React.ReactNode; right?: boolean; tone?: "pos" | "neg"
}) {
  const cls = "px-4 py-2 " + (right ? "text-right" : "")
    + (tone === "pos" ? " text-emerald-700 font-mono tabular-nums"
      : tone === "neg" ? " text-red-700 font-mono tabular-nums" : "");
  return <td className={cls}>{children}</td>;
}

function NeedBacktest() {
  return (
    <div className="card py-20 text-center">
      <div className="max-w-md mx-auto">
        <h3 className="text-lg font-semibold text-navy-950">No backtest in memory</h3>
        <p className="text-sm text-slate-500 mt-2">
          Run a backtest from the <span className="font-semibold text-navy-800">Backtest</span> tab,
          then return here for trade-level analytics.
        </p>
      </div>
    </div>
  );
}
