import { useEffect, useState } from "react";
import BacktestPage from "./pages/Backtest";
import TradesPage from "./pages/Trades";
import DiagnosticsPage from "./pages/Diagnostics";
import CointegrationPage from "./pages/Cointegration";
import GridSearchPage from "./pages/GridSearch";
import { fetchDefaults, runBacktestStream, type ProgressEvent } from "./api";
import type { BacktestRequest, BacktestResponse, DefaultsResponse } from "./types";

type Page = "backtest" | "trades" | "diagnostics" | "cointegration" | "grid";

const NAV: { id: Page; label: string; icon: string }[] = [
  { id: "backtest", label: "Backtest", icon: "📈" },
  { id: "trades", label: "Trade Analytics", icon: "📒" },
  { id: "diagnostics", label: "Factor Diagnostics", icon: "🧪" },
  { id: "cointegration", label: "Cointegration", icon: "🔗" },
  { id: "grid", label: "Grid Search", icon: "▦" },
];

export default function App() {
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [lastReq, setLastReq] = useState<BacktestRequest | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("backtest");

  useEffect(() => {
    fetchDefaults()
      .then(setDefaults)
      .catch((e) => setError(`Failed to load defaults: ${e.message}`));
  }, []);

  async function onRun(req: BacktestRequest) {
    setBusy(true);
    setError(null);
    setLastReq(req);
    setProgress({ stage: "starting", message: "Sending request…", progress: 0 });
    try {
      const r = await runBacktestStream(req, setProgress);
      setResult(r);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header page={page} setPage={setPage} hasResult={!!result} />
      <main className="flex-1 max-w-[1400px] w-full mx-auto px-6 py-8">
        {!defaults ? (
          <LoadingShell />
        ) : (
          <>
            {page === "backtest" && (
              <BacktestPage
                defaults={defaults}
                result={result}
                busy={busy}
                progress={progress}
                error={error}
                onRun={onRun}
              />
            )}
            {page === "trades" && <TradesPage r={result} />}
            {page === "diagnostics" && <DiagnosticsPage r={result} />}
            {page === "cointegration" && <CointegrationPage baseReq={lastReq} />}
            {page === "grid" && <GridSearchPage baseReq={lastReq} />}
          </>
        )}
      </main>
      <Footer />
    </div>
  );
}

function Header({ page, setPage, hasResult }: {
  page: Page; setPage: (p: Page) => void; hasResult: boolean;
}) {
  return (
    <header className="sticky top-0 z-20 bg-paper/85 backdrop-blur-xl border-b border-slate-200">
      <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Logo />
          <div>
            <h1 className="font-bold leading-tight text-navy-950 text-[15px]">StatArb Research</h1>
            <p className="text-xs text-slate-500">
              Avellaneda &amp; Lee (2010) — PCA / ETF / Pairs
            </p>
          </div>
        </div>
        <nav className="hidden md:flex items-center gap-1">
          {NAV.map((n) => {
            const needsRun = (n.id === "trades" || n.id === "diagnostics") && !hasResult;
            return (
              <button
                key={n.id}
                onClick={() => setPage(n.id)}
                title={needsRun ? "Run a backtest first" : undefined}
                className={
                  "px-3.5 py-2 rounded-lg text-sm font-medium transition flex items-center gap-2 " +
                  (page === n.id
                    ? "bg-navy-800 text-white shadow-card"
                    : needsRun
                      ? "text-slate-400 hover:bg-slate-100"
                      : "text-slate-700 hover:bg-slate-100")
                }
              >
                <span className="opacity-70">{n.icon}</span>
                {n.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-navy-700 to-navy-900 grid place-items-center shadow-card">
      <svg viewBox="0 0 32 32" className="w-5 h-5">
        <path d="M5 22 L11 14 L16 18 L22 9 L27 13"
          stroke="white" strokeWidth="2.5" fill="none"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-slate-200 mt-10">
      <div className="max-w-[1400px] mx-auto px-6 py-6 text-xs text-slate-500 flex items-center justify-between">
        <span>© StatArb Research</span>
        <span>FastAPI · React · Vite</span>
      </div>
    </footer>
  );
}

function LoadingShell() {
  return (
    <div className="grid place-items-center py-32 text-slate-500 text-sm">
      Loading defaults…
    </div>
  );
}
