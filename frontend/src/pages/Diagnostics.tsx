import { useMemo } from "react";
import {
  ResponsiveContainer, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Cell,
} from "recharts";
import type { BacktestResponse } from "../types";

export default function DiagnosticsPage({ r }: { r: BacktestResponse | null }) {
  if (!r) return <NeedBacktest />;
  const d = r.diagnostics;

  const eigSpectrum = (d.all_eigenvalues_top || []).map((v, i) => ({ idx: i + 1, eigval: v }));
  const eigCumvar = useMemo(() => {
    const total = eigSpectrum.reduce((a, x) => a + x.eigval, 0) || 1;
    let cum = 0;
    return eigSpectrum.map((x) => {
      cum += x.eigval;
      return { idx: x.idx, cum: cum / total };
    });
  }, [eigSpectrum]);

  const r2Hist = useMemo(() => bucket(d.r_squared.map((x) => x.r2), 0, 1, 20, (v) => v.toFixed(2)), [d.r_squared]);

  const ouRows = [...r.ou_last].sort((a, b) => b.kappa - a.kappa);
  const halfLifes = ouRows.map((x) => x.half_life).filter((x) => isFinite(x) && x > 0);
  const hlHist = useMemo(
    () => bucket(halfLifes, 0, Math.max(60, ...halfLifes), 20, (v) => v.toFixed(0)),
    [halfLifes]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Model" v={d.model_type.toUpperCase()} />
        <Stat label="Components" v={d.n_components.toString()} />
        <Stat label="Var explained" v={`${(d.explained_variance_ratio * 100).toFixed(1)}%`} />
        <Stat label="Stocks at last day" v={ouRows.length.toString()} />
      </div>

      {eigSpectrum.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Eigenvalue spectrum"
            subtitle="Top eigenvalues of the rolling correlation matrix">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={eigSpectrum.slice(0, 30)}>
                <CartesianGrid stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="idx" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip cursor={{ fill: "rgba(30,58,115,0.06)" }} />
                <Bar dataKey="eigval" radius={[4, 4, 0, 0]}>
                  {eigSpectrum.slice(0, 30).map((_, i) => (
                    <Cell key={i} fill={i < d.n_components ? "#1e3a73" : "#94a3b8"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Cumulative variance explained"
            subtitle={`First ${d.n_components} components capture ${(d.explained_variance_ratio * 100).toFixed(1)}%`}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={eigCumvar.slice(0, 30)}>
                <CartesianGrid stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="idx" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} />
                <Tooltip formatter={(v: any) => `${(v * 100).toFixed(1)}%`} />
                <ReferenceLine y={d.explained_variance_ratio} stroke="#1e3a73" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="cum" stroke="#1e3a73" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {d.r_squared.length > 0 && (
        <Card title="R² distribution"
          subtitle="How much of each stock's variance the factors explain">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={r2Hist}>
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip cursor={{ fill: "rgba(30,58,115,0.06)" }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="#1e3a73" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {hlHist.length > 0 && (
        <Card title="OU half-life distribution"
          subtitle="Mean-reversion speed across stocks (last day)">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={hlHist}>
              <CartesianGrid stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip cursor={{ fill: "rgba(30,58,115,0.06)" }} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} fill="#3b6aa5" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {ouRows.length > 0 && (
        <Card title="Per-stock OU parameters"
          subtitle={`Snapshot at last close · ${ouRows.length} stocks`}>
          <div className="max-h-96 overflow-y-auto rounded-xl border border-slate-200">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-canvas/95 backdrop-blur">
                <tr className="text-[11px] uppercase tracking-wider text-slate-500">
                  <Th>Ticker</Th>
                  <Th right>κ (speed)</Th>
                  <Th right>m</Th>
                  <Th right>σ_eq</Th>
                  <Th right>Half-life (days)</Th>
                  <Th right>Factor β</Th>
                </tr>
              </thead>
              <tbody>
                {ouRows.map((o) => (
                  <tr key={o.ticker} className="border-t border-slate-100">
                    <Td><span className="font-mono font-semibold text-navy-900">{o.ticker}</span></Td>
                    <Td right>{o.kappa.toFixed(2)}</Td>
                    <Td right>{o.m.toFixed(4)}</Td>
                    <Td right>{o.sigma_eq.toFixed(4)}</Td>
                    <Td right>{isFinite(o.half_life) ? o.half_life.toFixed(1) : "—"}</Td>
                    <Td right>{o.factor_beta.toFixed(3)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function bucket(values: number[], lo: number, hi: number, nb: number, fmt: (v: number) => string) {
  if (!values.length) return [];
  const w = (hi - lo) / nb || 1;
  const out: { label: string; count: number }[] = [];
  for (let i = 0; i < nb; i++) {
    const a = lo + i * w, b = a + w;
    const c = values.filter((x) => x >= a && (i === nb - 1 ? x <= b : x < b)).length;
    out.push({ label: fmt(a + w / 2), count: c });
  }
  return out;
}

function Stat({ label, v }: { label: string; v: string }) {
  return (
    <div className="card card-pad !p-4">
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
      <div className="text-2xl font-semibold mt-1 font-mono tabular-nums text-navy-900">{v}</div>
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="card">
      <div className="px-6 pt-5 pb-3">
        <h3 className="font-semibold text-navy-950">{title}</h3>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
      <div className="px-6 pb-5">{children}</div>
    </section>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th className={"px-4 py-2.5 font-semibold " + (right ? "text-right" : "text-left")}>{children}</th>;
}
function Td({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <td className={"px-4 py-2 font-mono tabular-nums " + (right ? "text-right" : "")}>{children}</td>;
}

function NeedBacktest() {
  return (
    <div className="card py-20 text-center">
      <div className="max-w-md mx-auto">
        <h3 className="text-lg font-semibold text-navy-950">No backtest in memory</h3>
        <p className="text-sm text-slate-500 mt-2">
          Run a backtest first — diagnostics are extracted from the factor-model fit.
        </p>
      </div>
    </div>
  );
}
