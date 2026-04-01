from config import FactorConfig
from .base import FactorModel
from .pca import PCAFactorModel
from .etf import ETFFactorModel
from .combined import CombinedFactorModel
from .pairs import PairsFactorModel


def build_factor_model(
    cfg: FactorConfig, sector_mapping: dict[str, str], pairs_cfg=None
) -> FactorModel:
    if cfg.model_type == "pca":
        return PCAFactorModel(
            n_components=cfg.pca_n_components,
            explained_variance_threshold=cfg.explained_variance_threshold,
            use_ledoit_wolf=cfg.use_ledoit_wolf,
            lookback=cfg.pca_lookback,
        )
    elif cfg.model_type == "etf":
        return ETFFactorModel(
            sector_mapping=sector_mapping,
            rolling_window=cfg.beta_rolling_window,
        )
    elif cfg.model_type == "combined":
        return CombinedFactorModel(
            sector_mapping=sector_mapping,
            rolling_window=cfg.beta_rolling_window,
            pca_n_components=cfg.pca_n_components,
            pca_lookback=cfg.pca_lookback,
            use_ledoit_wolf=cfg.use_ledoit_wolf,
        )
    elif cfg.model_type == "pairs":
        pc = pairs_cfg
        return PairsFactorModel(
            pvalue_threshold=pc.pvalue_threshold if pc else 0.05,
            max_pairs=pc.max_pairs if pc else 20,
            min_half_life=pc.min_half_life if pc else 1.0,
            max_half_life=pc.max_half_life if pc else 126.0,
            lookback=pc.lookback_window if pc else 252,
        )
    else:
        raise ValueError(f"Unknown model type: '{cfg.model_type}'. Choose: pca, etf, combined, pairs")
