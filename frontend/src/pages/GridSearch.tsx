import { useMemo, useState } from "react";
import { runGridSearch } from "../api";
import type { BacktestRequest, GridResponse } from "../types";

export default function GridSearchPage({ baseReq }: { baseReq: BacktestRequest | null }) {
  const [bo, setBo] = useState("1.0, 1.25, 1.5, 1.75, 2.0");
  const [so, setSo] = useState("1.0, 1.25, 1.5, 1.75, 2.0");
  const [data, setData] = useState<GridResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [metric, setMetric] = useState<"sharpe" | "total_return" | "max_drawdown">("sharpe");

  if (!baseReq) {
    return (
      <div className="card py-16 text-center">
        <p className="text-sm text-slate-500">
          Set a configuration in the Backtest tab first — grid search reuses the same config and only
          sweeps the entry-threshold pair.
        </p>
      </div>
    );
  }

  function parse(s: string) {
    return s.split(",").map((x) => parseFloat(x.trim())).filter((x) => isFinite(x));
  }

  async function run() {
    setBusy(true); setErr(null);
    try {
      const res = await runGridSearch({
        ...baseReq!,
        s_bo_values: parse(bo),
        s_so_values: parse(so),
      });
      setData(res);
    } catch (e: any) { setErr(e.message ?? String(e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="space-y-6">
      <section className="card card-pad">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="field">s_bo values (comma-separated)</label>
            <input className="w-full font-mono" value={bo}
              onChange={(e) => setBo(e.target.value)} />
          </div>
          <div>
            <label className="field">s_so values (comma-separated)</label>
            <input className="w-full font-mono" value={so}
              onChange={(e) => setSo(e.target.value)} />
          </div>
          <div className="flex items-end gap-3">
            <button className="btn-primary flex-1" disabled={busy} onClick={run}>
              {busy ? "Running…" : "Run grid search"}
            </button>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Each cell is a full backtest with the chosen (s_bo, s_so) pair. The factor model is fit once and
          reused, so a 5×5 sweep is roughly 5× the cost of a single run.
        </p>
        {err && <p className="text-sm text-red-700 mt-3">Error: {err}</p>}
      </section>

      {data && <Heatmap data={data} metric={metric} setMetric={setMetric} />}
      {data && data.best && <BestCard cell={data.best} />}
    </div>
  );
}

function Heatmap({
  data, metric, setMetric,
}: {
  data: GridResponse; metric: "sharpe" | "total_return" | "max_drawdown";
  setMetric: (m: any) => void;
}) {
  const lookup = useMemo(() => {
    const m = new Map<string, GridResponse["cells"][number]>();
    data.cells.forEach((c) => m.set(`${c.s_bo}_${c.s_so}`, c));
    return m;
  }, [data]);

  const values = data.cells.map((c) => c[metric]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);

  function color(v: number) {
    const t = (v - lo) / (hi - lo || 1);
    // dark navy → light navy → white
    const c1 = [12, 29, 68], c2 = [137, 171, 209];
    const r = Math.round(c1[0] + t * (c2[0] - c1[0]));
    const g = Math.round(c1[1] + t * (c2[1] - c1[1]));
    const b = Math.round(c1[2] + t * (c2[2] - c1[2]));
    return `rgb(${r},${g},${b})`;
  }

  function fmt(v: number) {
    if (metric === "sharpe") return v.toFixed(2);
    return `${(v * 100).toFixed(1)}%`;
  }

  return (
    <section className="card">
      <div className="px-6 pt-5 pb-3 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-navy-950">Sweep heatmap</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Rows: s_so (short entry) · Cols: s_bo (long entry)
          </p>
        </div>
        <div className="tabs">
          {(["sharpe", "total_return", "max_drawdown"] as const).map((m) => (
            <button key={m} onClick={() => setMetric(m)}
              className={"tab " + (metric === m ? "tab-active" : "")}>
              {m === "sharpe" ? "Sharpe" : m === "total_return" ? "Total return" : "Max DD"}
            </button>
          ))}
        </div>
      </div>
      <div className="px-6 pb-5 overflow-x-auto">
        <table className="border-separate border-spacing-1">
          <thead>
            <tr>
              <th></th>
              {data.s_bo_values.map((b) => (
                <th key={b} className="text-xs font-mono text-slate-500 px-2 pb-1">{b}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.s_so_values.map((s) => (
              <tr key={s}>
                <th className="text-xs font-mono text-slate-500 pr-2">{s}</th>
                {data.s_bo_values.map((b) => {
                  const c = lookup.get(`${b}_${s}`);
                  if (!c) return <td key={`${b}-${s}`} />;
                  const v = c[metric];
                  return (
                    <td key={`${b}-${s}`}
                      title={`s_bo=${b} s_so=${s} · sharpe=${c.sharpe.toFixed(2)} ret=${(c.total_return * 100).toFixed(1)}% dd=${(c.max_drawdown * 100).toFixed(1)}% trades=${c.num_trades}`}
                      style={{ background: color(v) }}
                      className="rounded-lg px-3 py-3 text-center font-mono text-sm shadow-card text-white min-w-[64px]"
                    >
                      {fmt(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function BestCard({ cell }: { cell: GridResponse["cells"][number] }) {
  return (
    <section className="card card-pad bg-navy-50/40 border-navy-200">
      <p className="text-[11px] uppercase tracking-wider text-navy-700 font-semibold">Best by Sharpe</p>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-3">
        <Stat label="s_bo" v={cell.s_bo.toFixed(2)} />
        <Stat label="s_so" v={cell.s_so.toFixed(2)} />
        <Stat label="Sharpe" v={cell.sharpe.toFixed(2)} tone="pos" />
        <Stat label="Total return" v={`${(cell.total_return * 100).toFixed(2)}%`}
          tone={cell.total_return >= 0 ? "pos" : "neg"} />
        <Stat label="Max DD" v={`${(cell.max_drawdown * 100).toFixed(2)}%`} tone="neg" />
      </div>
    </section>
  );
}

function Stat({ label, v, tone }: { label: string; v: string; tone?: "pos" | "neg" }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
      <div className={
        "text-xl font-semibold mt-1 font-mono tabular-nums " +
        (tone === "pos" ? "text-emerald-700" : tone === "neg" ? "text-red-700" : "text-navy-900")
      }>{v}</div>
    </div>
  );
}
