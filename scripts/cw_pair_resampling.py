"""Historical CW label-pair resampling; duplicate selected pairs are merged.

This returns central 95% resampling ranges, not standard bootstrap confidence
intervals. It preserves the original algorithm and RNG order for pair means.
"""
import numpy as np

def historical_pair_range(pair_means, n_resamples=1000, seed=124):
    values = np.asarray(pair_means, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    samples = np.array([
        values[np.unique(rng.choice(len(values), size=len(values), replace=True))].mean()
        for _ in range(n_resamples)
    ])
    return {'method': 'with-replacement pair draws, merged duplicates, equal unique-pair averaging',
            'interpretation': 'central 95% descriptive resampling range; not a standard bootstrap CI',
            'n_pairs': len(values), 'n_resamples': n_resamples, 'seed': seed,
            'lower': float(np.quantile(samples, .025)), 'upper': float(np.quantile(samples, .975))}
