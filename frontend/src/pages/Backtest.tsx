import ConfigPanel from "../components/ConfigPanel";
import KPICards from "../components/KPICard";
import {
  EquityChart, DrawdownChart, ExposureChart,
  AnnualReturnsChart, SScoreChart, RegimeChart,
} from "../components/Charts";
import SScoreTable from "../components/SScoreTable";
import type { BacktestRequest, BacktestResponse, DefaultsResponse } from "../types";
import type { ProgressEvent } from "../api";

interface Props {
  defaults: DefaultsResponse;
  result: BacktestResponse | null;
  busy: boolean;
  progress: ProgressEvent | null;
  error: string | null;
  onRun: (req: BacktestRequest) => void;
}

export default function BacktestPage({
  defaults, result, busy, progress, error, onRun,
}: Props) {
  return (
    <div className="space-y-6">
      <ConfigPanel defaults={defaults} onRun={onRun} busy={busy} />
      {error && <ErrorBanner msg={error} />}
      {!result && !busy && !error && <Placeholder />}
      {busy && <RunningCard progress={progress} />}
      {result && <Results r={result} />}
    </div>
  );
}

function Results({ r }: { r: BacktestResponse }) {
  return (
    <div className="space-y-6">
      <DataSummaryStrip
        n={r.data_summary.n_returned}
        req={r.data_summary.n_requested}
        dropped={r.data_summary.n_dropped}
      />
      <KPICards m={r.metrics} />
      <Card title="Equity curve" subtitle="Mark-to-market portfolio value">
        <EquityChart data={r.equity_curve} />
      </Card>
      {r.regime_curve && r.regime_curve.length > 0 && (
        <Card title="HMM regime probability"
          subtitle="P(favorable | data up to t) — entries gated when below threshold">
          <RegimeChart data={r.regime_curve} />
        </Card>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Drawdown" subtitle="Underwater curve from running peak">
          <DrawdownChart data={r.drawdown_curve} />
        </Card>
        <Card title="Gross exposure" subtitle="Long (top) vs short (bottom)">
          <ExposureChart data={r.exposure_curve} />
        </Card>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card title="Annual returns" subtitle="Calendar-year P&L">
          <AnnualReturnsChart data={r.annual_returns} />
        </Card>
        <Card title="Latest S-scores" subtitle="Distribution at last close" className="lg:col-span-2">
          {r.last_sscores.length === 0
            ? <Empty msg="No live signals" />
            : <SScoreChart data={r.last_sscores.slice(0, 30)} />}
        </Card>
      </div>
      <Card title="Signals book" subtitle="Filterable view of last s-scores">
        <SScoreTable rows={r.last_sscores} />
      </Card>
    </div>
  );
}

function Card({
  title, subtitle, children, className = "",
}: {
  title: string; subtitle?: string; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      <div className="px-6 pt-5 pb-3">
        <h3 className="font-semibold text-navy-950">{title}</h3>
        {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
      </div>
      <div className="px-6 pb-5">{children}</div>
    </section>
  );
}

function DataSummaryStrip({ n, req, dropped }: { n: number; req: number; dropped: number }) {
  const pct = (n / Math.max(1, req)) * 100;
  return (
    <div className="card px-6 py-4 flex items-center gap-8 flex-wrap">
      <Stat label="Requested" v={req} />
      <Stat label="Returned" v={n} />
      <Stat label="Dropped" v={dropped} tone={dropped > 0 ? "warn" : undefined} />
      <div className="flex-1 min-w-[160px]">
        <div className="flex items-center justify-between text-xs text-slate-500 mb-1">
          <span>Coverage</span>
          <span className="font-mono">{pct.toFixed(0)}%</span>
        </div>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-navy-500 to-navy-700"
            style={{ width: `${Math.min(100, pct)}%` }} />
        </div>
      </div>
    </div>
  );
}

function Stat({ label, v, tone }: { label: string; v: number; tone?: "warn" }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
      <div className={"font-mono tabular-nums text-xl mt-0.5 " +
        (tone === "warn" ? "text-amber-600" : "text-navy-950")}>
        {v}
      </div>
    </div>
  );
}

function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="card border-red-200 bg-red-50 px-5 py-4 text-sm">
      <span className="text-red-700 font-semibold">Error</span>
      <span className="ml-2 text-red-900">{msg}</span>
    </div>
  );
}

function Placeholder() {
  return (
    <div className="card py-20 grid place-items-center text-center">
      <div className="max-w-md">
        <div className="mx-auto mb-4 w-12 h-12 rounded-full bg-navy-50 grid place-items-center">
          <svg viewBox="0 0 24 24" className="w-6 h-6 text-navy-700" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 19V5m0 14h16M8 16l4-6 3 4 5-7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-navy-950">No backtest run yet</h3>
        <p className="text-sm text-slate-500 mt-2">
          Pick a model above and click <span className="kbd">Run Backtest</span>.
        </p>
      </div>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  starting: "Sending request",
  config: "Building configuration",
  fetch_data: "Fetching market data",
  universe: "Filtering universe",
  sector_map: "Resolving sector mapping",
  build_factor_model: "Building factor model",
  fetch_factors: "Fetching factor references",
  fit_factor_model: "Fitting factor model",
  run_backtest: "Running backtest engine",
  metrics: "Computing metrics",
  done: "Done",
};

function RunningCard({ progress }: { progress: ProgressEvent | null }) {
  const pct = progress ? Math.max(0, Math.min(1, progress.progress)) : 0;
  const stageLabel = progress
    ? STAGE_LABELS[progress.stage] ?? progress.stage
    : "Starting";
  const message = progress?.message ?? "Preparing run…";
  return (
    <div className="card py-12 px-8 grid place-items-center text-center">
      <div className="w-full max-w-md">
        <div className="mx-auto mb-4 w-10 h-10 rounded-full border-2 border-navy-700/20 border-t-navy-700 animate-spin" />
        <p className="text-sm font-semibold text-navy-950">{stageLabel}</p>
        <p className="text-xs text-slate-500 mt-1">{message}</p>
        <div className="mt-4">
          <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-navy-500 to-navy-700 transition-all duration-300 ease-out"
              style={{ width: `${pct * 100}%` }}
            />
          </div>
          <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-500 font-mono">
            <span>{progress?.stage ?? "starting"}</span>
            <span>{(pct * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return <div className="text-sm text-slate-500 py-8 text-center">{msg}</div>;
}
