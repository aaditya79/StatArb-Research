import type { Metrics } from "../types";

const fmtPct = (v: number) => `${(v * 100).toFixed(2)}%`;
const fmtNum = (v: number, d = 2) => v.toFixed(d);
const fmtInt = (v: number) => Number(v).toLocaleString();
const fmtCash = (v: number) => `$${Math.round(v).toLocaleString()}`;

export default function KPICards({ m }: { m: Metrics }) {
  const cards = [
    { label: "Total return", value: fmtPct(m.total_return), tone: m.total_return >= 0 ? "pos" : "neg" },
    { label: "Annualized return", value: fmtPct(m.annualized_return), tone: m.annualized_return >= 0 ? "pos" : "neg" },
    { label: "Sharpe", value: fmtNum(m.sharpe_ratio), tone: m.sharpe_ratio >= 1 ? "pos" : m.sharpe_ratio >= 0 ? "neutral" : "neg" },
    { label: "Sortino", value: fmtNum(m.sortino_ratio), tone: m.sortino_ratio >= 1 ? "pos" : m.sortino_ratio >= 0 ? "neutral" : "neg" },
    { label: "Max drawdown", value: fmtPct(m.max_drawdown), tone: "neg" },
    { label: "Annualized vol", value: fmtPct(m.annualized_vol), tone: "neutral" },
    { label: "# Trades", value: fmtInt(m.num_trades), tone: "neutral" },
    { label: "Win rate", value: fmtPct(m.trade_win_rate || m.win_rate), tone: "neutral" },
    { label: "Profit factor", value: fmtNum(m.profit_factor), tone: m.profit_factor >= 1 ? "pos" : "neg" },
    { label: "Total costs", value: fmtCash(m.total_costs), tone: "neutral" },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="card card-pad !p-4 group hover:shadow-cardLg transition">
          <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">
            {c.label}
          </div>
          <div className={
            "text-2xl font-semibold mt-1 font-mono tabular-nums " +
            (c.tone === "pos" ? "text-emerald-700"
              : c.tone === "neg" ? "text-red-700"
              : "text-navy-900")
          }>
            {c.value}
          </div>
        </div>
      ))}
    </div>
  );
}
