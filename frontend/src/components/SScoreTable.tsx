import { useState } from "react";

type Row = { ticker: string; sscore: number; signal: string };

export default function SScoreTable({ rows }: { rows: Row[] }) {
  const [filter, setFilter] = useState("all");
  const filtered = rows.filter((r) => filter === "all" || r.signal === filter);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        {["all", "LONG", "SHORT", "NEUTRAL"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={
              "px-3 py-1.5 rounded-lg text-xs font-semibold transition " +
              (filter === f
                ? "bg-navy-800 text-white"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200")
            }
          >
            {f}
            <span className={"ml-1.5 " + (filter === f ? "text-white/70" : "text-slate-400")}>
              {f === "all" ? rows.length : rows.filter((r) => r.signal === f).length}
            </span>
          </button>
        ))}
      </div>
      <div className="max-h-96 overflow-y-auto rounded-xl border border-slate-200">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-canvas/95 backdrop-blur">
            <tr className="text-[11px] uppercase tracking-wider text-slate-500">
              <th className="text-left px-4 py-2.5 font-semibold">Ticker</th>
              <th className="text-right px-4 py-2.5 font-semibold">S-Score</th>
              <th className="text-right px-4 py-2.5 font-semibold">Signal</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => (
              <tr key={r.ticker} className="border-t border-slate-100 hover:bg-navy-50/50">
                <td className="px-4 py-2 font-mono font-semibold text-navy-900">{r.ticker}</td>
                <td className={
                  "px-4 py-2 text-right font-mono tabular-nums " +
                  (r.sscore < 0 ? "text-emerald-700" : r.sscore > 0 ? "text-red-700" : "text-slate-700")
                }>
                  {r.sscore.toFixed(3)}
                </td>
                <td className="px-4 py-2 text-right">
                  <SignalBadge s={r.signal} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SignalBadge({ s }: { s: string }) {
  const cls =
    s === "LONG" ? "bg-emerald-50 text-emerald-700 border-emerald-200"
    : s === "SHORT" ? "bg-red-50 text-red-700 border-red-200"
    : "bg-slate-50 text-slate-600 border-slate-200";
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${cls}`}>
      {s}
    </span>
  );
}
