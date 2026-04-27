import {
  AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine,
} from "recharts";

const AXIS = { stroke: "#94a3b8", fontSize: 11 };
const GRID = "#e2e8f0";
const ACCENT = "#1e3a73";        // navy-700
const ACCENT_LIGHT = "#3b6aa5";  // navy-500
const POS = "#059669";
const NEG = "#dc2626";

const TooltipBox = ({ active, payload, label, fmt }: any) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 shadow-card text-xs">
      <div className="text-slate-500 mb-1 font-mono">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-600">{p.name}</span>
          <span className="font-mono text-navy-950 ml-auto font-semibold">
            {fmt ? fmt(p.value) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
};

export function EquityChart({ data }: { data: { date: string; equity: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={ACCENT} stopOpacity={0.35} />
            <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" {...AXIS} minTickGap={40} />
        <YAxis {...AXIS}
          tickFormatter={(v) => `$${(v / 1_000_000).toFixed(1)}M`} />
        <Tooltip content={<TooltipBox fmt={(v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />} />
        <Area type="monotone" dataKey="equity" name="Equity"
          stroke={ACCENT} strokeWidth={2} fill="url(#eq)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function DrawdownChart({ data }: { data: { date: string; drawdown: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="dd" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={NEG} stopOpacity={0} />
            <stop offset="100%" stopColor={NEG} stopOpacity={0.35} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" {...AXIS} minTickGap={40} />
        <YAxis {...AXIS} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip content={<TooltipBox fmt={(v: number) => `${(v * 100).toFixed(2)}%`} />} />
        <ReferenceLine y={0} stroke="#cbd5e1" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="drawdown" name="Drawdown"
          stroke={NEG} strokeWidth={2} fill="url(#dd)" />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ExposureChart({ data }: { data: { date: string; long: number; short: number }[] }) {
  const enriched = data.map((d) => ({ ...d, short: -Math.abs(d.short) }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={enriched} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="date" {...AXIS} minTickGap={40} />
        <YAxis {...AXIS}
          tickFormatter={(v) => `$${(v / 1_000_000).toFixed(1)}M`} />
        <Tooltip content={<TooltipBox fmt={(v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />} />
        <ReferenceLine y={0} stroke="#cbd5e1" />
        <Area type="monotone" dataKey="long" name="Long"
          stackId="1" stroke={POS} fill={POS} fillOpacity={0.25} />
        <Area type="monotone" dataKey="short" name="Short"
          stackId="1" stroke={NEG} fill={NEG} fillOpacity={0.25} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function AnnualReturnsChart({ data }: { data: { year: number; return: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="year" {...AXIS} />
        <YAxis {...AXIS} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip content={<TooltipBox fmt={(v: number) => `${(v * 100).toFixed(2)}%`} />} />
        <ReferenceLine y={0} stroke="#cbd5e1" />
        <Bar dataKey="return" name="Annual return" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.return >= 0 ? ACCENT : NEG} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SScoreChart({ data }: { data: { ticker: string; sscore: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, data.length * 20)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, left: 50, bottom: 0 }}>
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" {...AXIS} />
        <YAxis type="category" dataKey="ticker" {...AXIS} width={60} />
        <Tooltip content={<TooltipBox fmt={(v: number) => v.toFixed(3)} />} />
        <ReferenceLine x={0} stroke="#cbd5e1" />
        <Bar dataKey="sscore" name="S-score" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.sscore < 0 ? POS : d.sscore > 0 ? NEG : ACCENT_LIGHT} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
