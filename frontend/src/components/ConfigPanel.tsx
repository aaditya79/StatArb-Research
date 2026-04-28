import { useEffect, useMemo, useState } from "react";
import type { BacktestRequest, DefaultsResponse, ModelType, HedgeInstrument } from "../types";

interface Props {
  defaults: DefaultsResponse;
  onRun: (req: BacktestRequest) => void;
  busy: boolean;
}

type AdvTab = "factor" | "ou" | "signal" | "exec" | "ext";

export default function ConfigPanel({ defaults, onRun, busy }: Props) {
  const [tickersText, setTickersText] = useState(defaults.default_tickers.join(", "));
  const [advOpen, setAdvOpen] = useState(false);
  const [tab, setTab] = useState<AdvTab>("factor");

  const [req, setReq] = useState<BacktestRequest>({
    data_source: defaults.data_sources[0] ?? "yfinance",
    tickers: defaults.default_tickers,
    start_date: "1997-01-01",
    end_date: "2007-12-31",
    model_type: "pca",
    pca_lookback: 252,
    pca_n_components: 15,
    explained_variance_threshold: 0.55,
    use_ledoit_wolf: true,
    beta_rolling_window: 252,
    ou_window: 60,
    kappa_min: 8.4,
    mean_center: true,
    s_bo: 1.25, s_so: 1.25, s_sc: 0.50, s_bc: 0.75, s_limit: 4.0,
    vol_enabled: false, vol_window: 10,
    initial_equity: 1_000_000,
    leverage_long: 2.0, leverage_short: 2.0,
    tc_bps: 1.0, hedge_instrument: "SPY",
    pairs_pvalue: 0.05, pairs_max: 20, pairs_min_hl: 1.0, pairs_max_hl: 126.0,
    hmm_enabled: false, hmm_n_states: 2, hmm_training_window: 252,
    hmm_feature_window: 20, hmm_entry_threshold: 0.5,
    hmm_favorable_high_vol: true, hmm_soft_gate: true, hmm_soft_gate_floor: 0.2,
    vol_target_enabled: false, vol_target_floor: 0.2, vol_target_cap: 5.0,
  });

  useEffect(() => {
    const ts = tickersText
      .split(",").map((t) => t.trim().toUpperCase()).filter(Boolean);
    setReq((r) => ({ ...r, tickers: ts }));
  }, [tickersText]);

  function update<K extends keyof BacktestRequest>(k: K, v: BacktestRequest[K]) {
    setReq((r) => ({ ...r, [k]: v }));
  }

  const presetButtons = useMemo(
    () => Object.entries(defaults.ticker_presets),
    [defaults.ticker_presets]
  );

  return (
    <section className="card card-pad">
      {/* ── Strategy picker ── */}
      <div className="flex items-end justify-between gap-6 flex-wrap">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Strategy
          </p>
          <h2 className="text-2xl font-semibold tracking-tight text-navy-950 mt-0.5">
            Configure backtest
          </h2>
        </div>
        <button
          className="btn-primary"
          disabled={busy || req.tickers.length === 0}
          onClick={() => onRun(req)}
        >
          {busy ? <><Spinner /> Running…</> : <>Run Backtest →</>}
        </button>
      </div>

      <div className="mt-5 grid grid-cols-2 lg:grid-cols-4 gap-3">
        {defaults.model_types.map((m) => (
          <button
            key={m.value}
            type="button"
            onClick={() => update("model_type", m.value as ModelType)}
            className={
              "modelcard " + (req.model_type === m.value ? "modelcard-active" : "")
            }
          >
            <div className="flex items-center justify-between">
              <ModelIcon model={m.value as ModelType} active={req.model_type === m.value} />
              {req.model_type === m.value && (
                <span className="text-[10px] font-bold uppercase tracking-wider text-navy-700">
                  Selected
                </span>
              )}
            </div>
            <div className="mt-3 font-semibold text-navy-950">{m.label}</div>
            <p className="text-xs text-slate-500 mt-1 leading-snug">
              {modelBlurb(m.value as ModelType)}
            </p>
          </button>
        ))}
      </div>

      <div className="divider" />

      {/* ── Essentials grid ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Field label="Data source">
          <select className="w-full"
            value={req.data_source}
            onChange={(e) => update("data_source", e.target.value)}>
            {defaults.data_sources.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </Field>
        <Field label="Start date">
          <input type="date" className="w-full"
            value={req.start_date}
            onChange={(e) => update("start_date", e.target.value)} />
        </Field>
        <Field label="End date">
          <input type="date" className="w-full"
            value={req.end_date}
            onChange={(e) => update("end_date", e.target.value)} />
        </Field>
        <Field label="Hedge instrument">
          <select className="w-full"
            value={req.hedge_instrument}
            onChange={(e) => update("hedge_instrument", e.target.value as HedgeInstrument)}>
            {defaults.hedge_instruments.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </Field>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between mb-2">
          <label className="field !mb-0">Universe ({req.tickers.length} tickers)</label>
          <div className="flex gap-2">
            {presetButtons.map(([name]) => (
              <button key={name} className="btn-ghost"
                onClick={() => setTickersText(defaults.ticker_presets[name].join(", "))}>
                {name}
              </button>
            ))}
          </div>
        </div>
        <textarea rows={2}
          className="w-full font-mono text-xs"
          value={tickersText}
          onChange={(e) => setTickersText(e.target.value)} />
      </div>

      {/* ── Advanced ── */}
      <div className="mt-5">
        <button
          onClick={() => setAdvOpen(!advOpen)}
          className="flex items-center gap-2 text-sm font-semibold text-navy-800 hover:text-navy-700"
        >
          <span className={"transition-transform " + (advOpen ? "rotate-90" : "")}>›</span>
          Advanced settings
          <span className="text-xs font-normal text-slate-500">
            (factor model · OU · signal thresholds · execution)
          </span>
        </button>

        {advOpen && (
          <div className="mt-4">
            <div className="tabs">
              {(["factor", "ou", "signal", "exec", "ext"] as AdvTab[]).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={"tab " + (tab === t ? "tab-active" : "")}
                >
                  {tabLabel(t)}
                </button>
              ))}
            </div>

            <div className="mt-4">
              {tab === "factor" && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {(req.model_type === "pca" || req.model_type === "combined") && (
                    <>
                      <NumField label="PCA lookback (days)" v={req.pca_lookback}
                        step={1} min={60} max={504}
                        onChange={(v) => update("pca_lookback", v)} />
                      <NumField label="PCA components" v={req.pca_n_components ?? 15}
                        step={1} min={1} max={30}
                        onChange={(v) => update("pca_n_components", v)} />
                      <Toggle label="Ledoit-Wolf shrinkage"
                        v={req.use_ledoit_wolf}
                        onChange={(v) => update("use_ledoit_wolf", v)} />
                    </>
                  )}
                  {req.model_type === "pairs" && (
                    <>
                      <NumField label="Coint p-value" v={req.pairs_pvalue}
                        step={0.01} min={0.01} max={0.20}
                        onChange={(v) => update("pairs_pvalue", v)} />
                      <NumField label="Max pairs" v={req.pairs_max}
                        step={1} min={5} max={50}
                        onChange={(v) => update("pairs_max", v)} />
                      <NumField label="Min half-life" v={req.pairs_min_hl}
                        step={1} min={1} max={30}
                        onChange={(v) => update("pairs_min_hl", v)} />
                      <NumField label="Max half-life" v={req.pairs_max_hl}
                        step={1} min={10} max={252}
                        onChange={(v) => update("pairs_max_hl", v)} />
                    </>
                  )}
                  <NumField label="Rolling β window" v={req.beta_rolling_window}
                    step={1} min={40} max={504}
                    onChange={(v) => update("beta_rolling_window", v)} />
                </div>
              )}

              {tab === "ou" && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <NumField label="OU window (days)" v={req.ou_window}
                    step={1} min={40} max={120}
                    onChange={(v) => update("ou_window", v)} />
                  <NumField label="Min κ" v={req.kappa_min}
                    step={0.1} min={0} max={50}
                    onChange={(v) => update("kappa_min", v)} />
                  <Toggle label="Mean centering"
                    v={req.mean_center}
                    onChange={(v) => update("mean_center", v)} />
                </div>
              )}

              {tab === "signal" && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <NumField label="Entry s_bo" v={req.s_bo} step={0.05} min={0.5} max={3}
                    onChange={(v) => update("s_bo", v)} />
                  <NumField label="Entry s_so" v={req.s_so} step={0.05} min={0.5} max={3}
                    onChange={(v) => update("s_so", v)} />
                  <NumField label="Exit long s_sc" v={req.s_sc} step={0.05} min={0.1} max={2}
                    onChange={(v) => update("s_sc", v)} />
                  <NumField label="Exit short s_bc" v={req.s_bc} step={0.05} min={0.1} max={2}
                    onChange={(v) => update("s_bc", v)} />
                  <NumField label="Force exit" v={req.s_limit} step={0.5} min={2} max={10}
                    onChange={(v) => update("s_limit", v)} />
                  <Toggle label="Volume-time adjustment"
                    v={req.vol_enabled}
                    onChange={(v) => update("vol_enabled", v)} />
                  {req.vol_enabled && (
                    <NumField label="Vol window" v={req.vol_window}
                      step={1} min={5} max={30}
                      onChange={(v) => update("vol_window", v)} />
                  )}
                </div>
              )}

              {tab === "exec" && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <NumField label="Initial equity ($)" v={req.initial_equity}
                    step={100000} min={10000} max={100000000}
                    onChange={(v) => update("initial_equity", v)} />
                  <NumField label="Long leverage" v={req.leverage_long}
                    step={0.5} min={0} max={5}
                    onChange={(v) => update("leverage_long", v)} />
                  <NumField label="Short leverage" v={req.leverage_short}
                    step={0.5} min={0} max={5}
                    onChange={(v) => update("leverage_short", v)} />
                  <NumField label="Cost (bps/side)" v={req.tc_bps}
                    step={0.5} min={0} max={50}
                    onChange={(v) => update("tc_bps", v)} />
                </div>
              )}

              {tab === "ext" && (
                <div className="space-y-5">
                  {/* HMM regime filter */}
                  <ExtBlock
                    title="HMM regime filter"
                    desc="Fits a 2-state Gaussian HMM on cross-sectional residual vol; gates new entries by P(favorable) computed causally."
                    enabled={req.hmm_enabled}
                    onToggle={(v) => update("hmm_enabled", v)}
                  >
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <NumField label="Hidden states" v={req.hmm_n_states}
                        step={1} min={2} max={4}
                        onChange={(v) => update("hmm_n_states", v)} />
                      <NumField label="Training window" v={req.hmm_training_window}
                        step={1} min={60} max={1000}
                        onChange={(v) => update("hmm_training_window", v)} />
                      <NumField label="Feature window" v={req.hmm_feature_window}
                        step={1} min={5} max={60}
                        onChange={(v) => update("hmm_feature_window", v)} />
                      <NumField label="Entry threshold" v={req.hmm_entry_threshold}
                        step={0.05} min={0.0} max={1.0}
                        onChange={(v) => update("hmm_entry_threshold", v)} />
                      <NumField label="Soft-gate floor" v={req.hmm_soft_gate_floor}
                        step={0.05} min={0.0} max={1.0}
                        onChange={(v) => update("hmm_soft_gate_floor", v)} />
                    </div>
                    <div className="flex flex-wrap gap-4 mt-3 text-xs text-slate-700">
                      <label className="flex items-center gap-2">
                        <input type="checkbox" checked={req.hmm_favorable_high_vol}
                          onChange={(e) => update("hmm_favorable_high_vol", e.target.checked)} />
                        Favorable = high cross-sectional residual vol (mean-reversion harvests dispersion)
                      </label>
                      <label className="flex items-center gap-2">
                        <input type="checkbox" checked={req.hmm_soft_gate}
                          onChange={(e) => update("hmm_soft_gate", e.target.checked)} />
                        Soft gate (scale notional by P(favorable) instead of binary on/off)
                      </label>
                    </div>
                  </ExtBlock>

                  {/* Vol-targeted sizing */}
                  <ExtBlock
                    title="Vol-targeted sizing"
                    desc="Scales each new position's notional by target_sigma / sigma_eq, so high-vol stocks contribute equal risk rather than equal notional."
                    enabled={req.vol_target_enabled}
                    onToggle={(v) => update("vol_target_enabled", v)}
                  >
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      <NumField label="Floor multiplier" v={req.vol_target_floor}
                        step={0.05} min={0.0} max={2.0}
                        onChange={(v) => update("vol_target_floor", v)} />
                      <NumField label="Cap multiplier" v={req.vol_target_cap}
                        step={0.5} min={1.0} max={20.0}
                        onChange={(v) => update("vol_target_cap", v)} />
                    </div>
                  </ExtBlock>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function tabLabel(t: AdvTab) {
  return {
    factor: "Factor model",
    ou: "OU process",
    signal: "Signal",
    exec: "Execution",
    ext: "Extensions",
  }[t];
}

function modelBlurb(m: ModelType) {
  return {
    pca: "Eigenportfolios from rolling 252-day correlation",
    etf: "Stock vs. sector ETF residual",
    combined: "SPY → sector ETF → PCA stack",
    pairs: "Cointegrated stock pairs",
  }[m];
}

function ModelIcon({ model, active }: { model: ModelType; active: boolean }) {
  const cls = active ? "text-navy-700" : "text-slate-400";
  const stroke = "currentColor";
  return (
    <span className={cls}>
      <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke={stroke} strokeWidth="2">
        {model === "pca" && (
          <>
            <circle cx="12" cy="12" r="3" />
            <circle cx="12" cy="12" r="7" strokeDasharray="2 2" />
            <circle cx="12" cy="12" r="11" strokeDasharray="2 4" opacity="0.5" />
          </>
        )}
        {model === "etf" && (
          <>
            <path d="M3 17l5-5 4 3 8-9" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="3" cy="17" r="1.4" fill={stroke} />
            <circle cx="20" cy="6" r="1.4" fill={stroke} />
          </>
        )}
        {model === "combined" && (
          <>
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="8.5" y="14" width="7" height="7" rx="1" />
            <path d="M6.5 10v3M17.5 10v3M12 10v3" />
          </>
        )}
        {model === "pairs" && (
          <>
            <path d="M4 8c4 0 4 8 8 8s4-8 8-8" strokeLinecap="round" />
            <circle cx="4" cy="8" r="1.6" fill={stroke} />
            <circle cx="20" cy="8" r="1.6" fill={stroke} />
          </>
        )}
      </svg>
    </span>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="field">{label}</label>
      {children}
    </div>
  );
}

function NumField({ label, v, onChange, step, min, max }: {
  label: string; v: number; onChange: (v: number) => void;
  step?: number; min?: number; max?: number;
}) {
  return (
    <Field label={label}>
      <input type="number" className="w-full font-mono"
        value={v} step={step} min={min} max={max}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
    </Field>
  );
}

function Toggle({ label, v, onChange }: { label: string; v: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 select-none cursor-pointer mt-6">
      <input type="checkbox" checked={v} onChange={(e) => onChange(e.target.checked)} />
      <span className="text-sm text-navy-950">{label}</span>
    </label>
  );
}

function Spinner() {
  return (
    <span className="inline-block w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
  );
}

function ExtBlock({
  title, desc, enabled, onToggle, children,
}: {
  title: string; desc: string; enabled: boolean;
  onToggle: (v: boolean) => void; children: React.ReactNode;
}) {
  return (
    <div className={
      "rounded-xl border p-4 transition " +
      (enabled
        ? "border-navy-700 bg-navy-50/40 ring-2 ring-navy-700/10"
        : "border-slate-200 bg-white")
    }>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={
              "inline-block w-2 h-2 rounded-full " +
              (enabled ? "bg-emerald-500" : "bg-slate-300")
            } />
            <h4 className="font-semibold text-navy-950">{title}</h4>
            <span className={
              "text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded " +
              (enabled
                ? "bg-emerald-50 text-emerald-700"
                : "bg-slate-100 text-slate-500")
            }>
              {enabled ? "ON" : "OFF"}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1 leading-snug">{desc}</p>
        </div>
        <Switch v={enabled} onChange={onToggle} />
      </div>
      {enabled && <div className="mt-4">{children}</div>}
    </div>
  );
}

function Switch({ v, onChange }: { v: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={v}
      onClick={() => onChange(!v)}
      className={
        "relative inline-flex h-6 w-11 items-center rounded-full transition " +
        (v ? "bg-navy-700" : "bg-slate-300")
      }
    >
      <span className={
        "inline-block h-5 w-5 transform rounded-full bg-white shadow transition " +
        (v ? "translate-x-5" : "translate-x-0.5")
      } />
    </button>
  );
}
