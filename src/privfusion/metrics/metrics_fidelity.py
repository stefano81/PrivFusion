import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sdmetrics.single_column import KSComplement, TVComplement


def compute_WD(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    wd_col = []
    for col in df_real.columns:
        wd_col.append(wasserstein_distance(df_real[col], df_synth[col]))
    return np.mean(wd_col)


def compute_JSD(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    if len(df_real) != len(df_synth):
        k_sample = min(len(df_real), len(df_synth))
        df_real = df_real.sample(k_sample)
        df_synth = df_synth.sample(k_sample)
    jsd = jensenshannon(df_real, df_synth, axis=1)  # type: ignore[unknown-argument]
    return np.mean(jsd)


def compute_KSComplement(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """
    Computes the similarity of a real column vs. a synthetic column in terms of the marginal distribution of the continuous data.
    """
    ks_col = []
    for col in df_real.columns:
        ks_col.append(
            KSComplement.compute(real_data=df_real[col], synthetic_data=df_synth[col]),
        )
    return np.mean(ks_col)


def compute_TVComplement(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """
    Computes the similarity of a real column vs. a synthetic column in terms of the marginal distribution of the discrete (categorical) data.
    """
    tv_col = []
    for col in df_real.columns:
        tv_col.append(
            TVComplement.compute(real_data=df_real[col], synthetic_data=df_synth[col]),
        )
    return np.mean(tv_col)
