export type ModelType = "pca" | "etf" | "combined" | "pairs";
export type HedgeInstrument = "SPY" | "sector_etf" | "none";

export interface BacktestRequest {
  data_source: string;
  tickers: string[];
  start_date: string;
  end_date: string;

  model_type: ModelType;
  pca_lookback: number;
  pca_n_components: number | null;
  explained_variance_threshold: number;
  use_ledoit_wolf: boolean;
  beta_rolling_window: number;

  ou_window: number;
  kappa_min: number;
  mean_center: boolean;

  s_bo: number;
  s_so: number;
  s_sc: number;
  s_bc: number;
  s_limit: number;

  vol_enabled: boolean;
  vol_window: number;

  initial_equity: number;
  leverage_long: number;
  leverage_short: number;
  tc_bps: number;
  hedge_instrument: HedgeInstrument;

  pairs_pvalue: number;
  pairs_max: number;
  pairs_min_hl: number;
  pairs_max_hl: number;

  hmm_enabled: boolean;
  hmm_n_states: number;
  hmm_training_window: number;
  hmm_feature_window: number;
  hmm_entry_threshold: number;
  hmm_favorable_high_vol: boolean;
  hmm_soft_gate: boolean;
  hmm_soft_gate_floor: number;

  vol_target_enabled: boolean;
  vol_target_floor: number;
  vol_target_cap: number;
}

export interface Metrics {
  total_return: number;
  annualized_return: number;
  annualized_vol: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trade_win_rate: number;
  profit_factor: number;
  num_trades: number;
  total_costs: number;
  avg_holding_period: number;
}

export interface Trade {
  ticker: string;
  direction: number;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  notional: number;
}

export interface FactorDiagnostics {
  model_type: string;
  eigenvalues: number[];
  all_eigenvalues_top: number[];
  explained_variance_ratio: number;
  n_components: number;
  r_squared: { ticker: string; r2: number }[];
}

export interface OURow {
  ticker: string;
  kappa: number;
  m: number;
  sigma_eq: number;
  half_life: number;
  factor_beta: number;
}

export interface BacktestResponse {
  metrics: Metrics;
  equity_curve: { date: string; equity: number }[];
  drawdown_curve: { date: string; drawdown: number }[];
  exposure_curve: { date: string; long: number; short: number }[];
  annual_returns: { year: number; return: number }[];
  last_sscores: { ticker: string; sscore: number; signal: string }[];
  data_summary: { n_requested: number; n_returned: number; n_dropped: number };
  trades: Trade[];
  diagnostics: FactorDiagnostics;
  ou_last: OURow[];
  regime_curve: { date: string; p_favorable: number }[];
}

export interface CointPair {
  ticker1: string;
  ticker2: string;
  pvalue: number;
  score: number;
  hedge_ratio: number;
  spread_mean: number;
  spread_std: number;
  half_life: number | null;
}

export interface CointResponse {
  pairs: CointPair[];
  spread: { date: string; spread: number; z: number }[];
}

export interface GridCell {
  s_bo: number;
  s_so: number;
  sharpe: number;
  total_return: number;
  max_drawdown: number;
  num_trades: number;
}

export interface GridResponse {
  s_bo_values: number[];
  s_so_values: number[];
  cells: GridCell[];
  best: GridCell | null;
}

export interface DefaultsResponse {
  default_tickers: string[];
  paper_tickers_count: number;
  modern_tickers_count: number;
  data_sources: string[];
  model_types: { value: ModelType; label: string }[];
  hedge_instruments: HedgeInstrument[];
  ticker_presets: Record<string, string[]>;
}
