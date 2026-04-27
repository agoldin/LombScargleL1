"""
L1 (robust) Lomb-Scargle periodogram.

A robust alternative to the classical least-squares Lomb-Scargle periodogram,
based on Makarov et al. 2024 (arXiv:2405.12324). Replaces the L2 (least-squares)
objective with L1 (least absolute deviations), making it resistant to outliers
and heavy-tailed noise distributions.

API mirrors astropy.timeseries.LombScargle where possible.
"""

import numpy as np
from scipy.optimize import linprog


class LombScargleL1:
    """Robust L1 Lomb-Scargle periodogram for irregularly sampled time series.

    Parameters
    ----------
    t : array-like
        Observation times.
    y : array-like
        Observation values.
    dy : array-like or float, optional
        Measurement uncertainties. If provided, used as weights (1/dy).
    fit_mean : bool, optional
        If True, include a constant offset in the model. Default True.
    fit_trend : bool, optional
        If True, include a linear trend in the model. Default True.
    nterms : int, optional
        Number of Fourier terms in the model. Default 1 (single sinusoid).
        Higher values fit harmonics at 2f, 3f, etc.
    center_data : bool, optional
        If True, pre-center the data by subtracting the (weighted) mean.
        Default True.
    normalization : str, optional
        Periodogram normalization: 'standard', 'model', 'log', or 'psd'.
        Default 'standard'.
    """

    def __init__(self, t, y, dy=None, fit_mean=True, fit_trend=True,
                 nterms=1, center_data=True, normalization='standard'):
        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)

        if dy is not None:
            self.dy = np.atleast_1d(np.asarray(dy, dtype=float))
            if self.dy.ndim == 0 or self.dy.shape == (1,):
                self.dy = np.full_like(self.y, float(self.dy.ravel()[0]))
        else:
            self.dy = None

        self.fit_mean = fit_mean
        self.fit_trend = fit_trend
        self.nterms = nterms
        self.center_data = center_data
        self.normalization = normalization

        self._validate()

        # Pre-center
        self._y_offset = 0.0
        if center_data:
            if self.dy is not None:
                w = 1.0 / self.dy
                self._y_offset = np.average(self.y, weights=w)
            else:
                self._y_offset = np.mean(self.y)

        self._t_offset = 0.5 * (self.t.min() + self.t.max())

    def _validate(self):
        if self.t.ndim != 1:
            raise ValueError("t must be 1-dimensional")
        if self.y.shape != self.t.shape:
            raise ValueError("t and y must have the same shape")
        if self.dy is not None and self.dy.shape != self.t.shape:
            raise ValueError("dy must have the same shape as t")
        if self.nterms < 1:
            raise ValueError("nterms must be >= 1")
        if self.normalization not in ('standard', 'model', 'log', 'psd'):
            raise ValueError(
                f"normalization must be one of 'standard', 'model', 'log', 'psd', "
                f"got '{self.normalization}'"
            )

    def _centered_data(self):
        return self.y - self._y_offset

    def _centered_times(self):
        return self.t - self._t_offset

    def _design_matrix(self, frequency, t=None):
        """Build design matrix for a single frequency.

        Columns: [mean, trend, cos(f), sin(f), cos(2f), sin(2f), ...].
        The number of cos/sin pairs equals self.nterms.
        """
        if t is None:
            t_c = self._centered_times()
            t_raw = self.t
        else:
            t_raw = np.asarray(t, dtype=float)
            t_c = t_raw - self._t_offset

        cols = []
        if self.fit_mean:
            cols.append(np.ones_like(t_c))
        if self.fit_trend:
            cols.append(t_c)
        for n in range(1, self.nterms + 1):
            phase = 2.0 * np.pi * n * frequency * t_raw
            cols.append(np.cos(phase))
            cols.append(np.sin(phase))
        return np.column_stack(cols)

    def _solve_l1(self, A, d):
        """Solve LAD regression: min Σ|d - A·x| via linear programming.

        Reformulated as:
            min  Σ u_i
            s.t. A·x + u >= d
                 -A·x + u >= -d
                 u >= 0

        Returns the coefficient vector x.
        """
        n, p = A.shape

        # Variables: [x (p), u (n)]
        # Objective: min sum(u) = [0...0, 1...1] · [x, u]
        c = np.concatenate([np.zeros(p), np.ones(n)])

        # Inequality constraints: G · [x, u] <= h
        # From  A·x + u >= d  =>  -A·x - u <= -d
        # From -A·x + u >= -d =>   A·x - u <=  d
        I_n = np.eye(n)
        G_ub = np.vstack([
            np.hstack([-A, -I_n]),
            np.hstack([A, -I_n]),
        ])
        h_ub = np.concatenate([-d, d])

        # u >= 0, x is unbounded
        bounds = [(None, None)] * p + [(0, None)] * n

        result = linprog(c, A_ub=G_ub, b_ub=h_ub, bounds=bounds, method='highs')

        if not result.success:
            raise RuntimeError(f"L1 optimization failed: {result.message}")

        return result.x[:p]

    def _l1_residual_sum(self, A, d, x):
        """Sum of absolute residuals."""
        return np.sum(np.abs(d - A @ x))

    def _reference_l1(self):
        """L1 residual sum for the reference (no sinusoidal) model."""
        d = self._centered_data()
        if self.dy is not None:
            w = 1.0 / self.dy
            d = d * w

        cols = []
        t_c = self._centered_times()
        if self.fit_mean:
            col = np.ones_like(t_c)
            if self.dy is not None:
                col = col * (1.0 / self.dy)
            cols.append(col)
        if self.fit_trend:
            col = t_c.copy()
            if self.dy is not None:
                col = col * (1.0 / self.dy)
            cols.append(col)

        if len(cols) == 0:
            return np.sum(np.abs(d))

        A_ref = np.column_stack(cols)
        x_ref = self._solve_l1(A_ref, d)
        return self._l1_residual_sum(A_ref, d, x_ref)

    def _power_single(self, frequency):
        """Compute raw L1 power at a single frequency.

        Returns (normalized_power, x_coeffs).
        """
        d = self._centered_data()
        A = self._design_matrix(frequency)

        if self.dy is not None:
            w = 1.0 / self.dy
            A = A * w[:, np.newaxis]
            d = d * w

        x = self._solve_l1(A, d)
        l1_freq = self._l1_residual_sum(A, d, x)

        return l1_freq, x

    def _normalize(self, l1_freq, l1_ref):
        """Apply normalization to convert L1 residuals to periodogram power."""
        norm = self.normalization
        if norm == 'standard':
            return (l1_ref - l1_freq) / l1_ref
        elif norm == 'model':
            return (l1_ref - l1_freq) / l1_freq
        elif norm == 'log':
            return np.log(l1_ref / l1_freq)
        elif norm == 'psd':
            # Simplified: classical PSD scaling (Parseval's theorem) relies on
            # statistical properties of L2 that do not carry over to L1.
            return 0.5 * (l1_ref - l1_freq)
        else:
            raise ValueError(f"Unknown normalization: {norm}")

    def power(self, frequency):
        """Compute the L1 periodogram power at the given frequencies.

        Parameters
        ----------
        frequency : array-like
            Frequencies at which to evaluate the periodogram.

        Returns
        -------
        power : ndarray
            Periodogram power values.
        """
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        l1_ref = self._reference_l1()

        power = np.empty(len(frequency))
        for i, f in enumerate(frequency):
            l1_f, _ = self._power_single(f)
            power[i] = self._normalize(l1_f, l1_ref)

        return power

    def autopower(self, minimum_frequency=None, maximum_frequency=None,
                  nyquist_factor=5, samples_per_peak=5):
        """Compute the L1 periodogram on an automatically-determined frequency grid.

        Parameters
        ----------
        minimum_frequency : float, optional
            Minimum frequency. Default: 1 / (time baseline * nyquist_factor).
        maximum_frequency : float, optional
            Maximum frequency. Default: nyquist_factor * pseudo-Nyquist frequency.
        nyquist_factor : float, optional
            Multiplier for the Nyquist frequency estimate. Default 5.
        samples_per_peak : float, optional
            Approximate number of grid points per peak width. Default 5.

        Returns
        -------
        frequency : ndarray
            Frequency grid.
        power : ndarray
            Periodogram power at each frequency.
        """
        frequency = self._autofrequency(
            minimum_frequency, maximum_frequency,
            nyquist_factor, samples_per_peak
        )
        return frequency, self.power(frequency)

    def _autofrequency(self, minimum_frequency, maximum_frequency,
                       nyquist_factor, samples_per_peak):
        """Generate automatic frequency grid, matching astropy convention."""
        baseline = self.t.max() - self.t.min()
        n = len(self.t)

        df = 1.0 / (baseline * samples_per_peak)

        if minimum_frequency is None:
            minimum_frequency = 0.5 * df
        if maximum_frequency is None:
            avg_dt = baseline / (n - 1)
            nyquist = 0.5 / avg_dt
            maximum_frequency = nyquist * nyquist_factor

        nf = int(np.ceil((maximum_frequency - minimum_frequency) / df))
        return minimum_frequency + df * np.arange(nf)

    def model(self, t, frequency):
        """Compute the best-fit L1 model at a given frequency.

        Parameters
        ----------
        t : array-like
            Times at which to evaluate the model.
        frequency : float
            Frequency of the sinusoidal model.

        Returns
        -------
        y_model : ndarray
            Model values at times t.
        """
        params = self.model_parameters(frequency)
        t = np.asarray(t, dtype=float)
        A = self._design_matrix(frequency, t=t)
        return A @ params + self._y_offset

    def model_parameters(self, frequency):
        """Compute best-fit L1 model parameters at a given frequency.

        Parameters
        ----------
        frequency : float
            Frequency of the sinusoidal model.

        Returns
        -------
        params : ndarray
            Model parameters. Order: [mean, trend, cos1, sin1, cos2, sin2, ...],
            where mean and trend are included only if fit_mean/fit_trend are True,
            and there are nterms cos/sin pairs for harmonics 1..nterms.
        """
        _, x = self._power_single(frequency)
        # If weighted, x was solved on weighted system — need to return unweighted params
        # Actually the params x are the same whether weighted or not:
        # W·A·x = W·d => A·x ≈ d in L1 sense weighted by W.
        # x are the model coefficients, applicable to the unweighted model.
        return x

    def _bootstrap_max_powers(self, frequency, n_bootstrap, random_state,
                              n_jobs=1):
        """Compute max power over bootstrap permutations of y."""
        rng = np.random.default_rng(random_state)
        if frequency is None:
            frequency = self._autofrequency(None, None, nyquist_factor=5,
                                            samples_per_peak=5)
        # Pre-generate all shuffled arrays for reproducibility
        y_shuffled_list = [rng.permutation(self.y) for _ in range(n_bootstrap)]

        def _one_bootstrap(y_shuffled):
            ls_boot = LombScargleL1(
                self.t, y_shuffled, dy=self.dy,
                fit_mean=self.fit_mean, fit_trend=self.fit_trend,
                nterms=self.nterms, center_data=self.center_data,
                normalization=self.normalization,
            )
            return np.max(ls_boot.power(frequency))

        if n_jobs == 1:
            max_powers = np.array([_one_bootstrap(ys) for ys in y_shuffled_list])
        else:
            from joblib import Parallel, delayed
            max_powers = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(_one_bootstrap)(ys) for ys in y_shuffled_list
                )
            )
        return max_powers

    def false_alarm_probability(self, power, method='bootstrap', n_bootstrap=1000,
                                frequency=None, random_state=None, n_jobs=1):
        """Estimate the false alarm probability for a given power level.

        Uses bootstrap: shuffle y values (keeping time stamps fixed), recompute
        periodogram maximum, and estimate P(max_power >= power | H0).

        Parameters
        ----------
        power : float or array-like
            Power value(s) to evaluate.
        method : str
            Only 'bootstrap' is supported.
        n_bootstrap : int
            Number of bootstrap iterations.
        frequency : array-like, optional
            Frequency grid for bootstrap periodograms. If None, uses
            _autofrequency defaults. Pass a coarser grid to speed up.
        random_state : int or numpy.random.Generator, optional
            Random state for reproducibility.
        n_jobs : int, optional
            Number of parallel jobs (requires joblib). -1 uses all cores.
            Default 1 (sequential).

        Returns
        -------
        fap : float or ndarray
            False alarm probability.
        """
        if method != 'bootstrap':
            raise ValueError("Only 'bootstrap' method is supported for L1 periodogram")

        power = np.atleast_1d(np.asarray(power, dtype=float))
        max_powers = self._bootstrap_max_powers(frequency, n_bootstrap,
                                                random_state, n_jobs=n_jobs)

        fap = np.array([np.mean(max_powers >= p) for p in power])
        return float(fap[0]) if fap.size == 1 else fap

    def false_alarm_level(self, probability, method='bootstrap', n_bootstrap=1000,
                          frequency=None, random_state=None, n_jobs=1):
        """Compute the power level for a given false alarm probability.

        Parameters
        ----------
        probability : float or array-like
            Target false alarm probability.
        method : str
            Only 'bootstrap' is supported.
        n_bootstrap : int
            Number of bootstrap iterations.
        frequency : array-like, optional
            Frequency grid for bootstrap periodograms. If None, uses
            _autofrequency defaults. Pass a coarser grid to speed up.
        random_state : int or numpy.random.Generator, optional
            Random state for reproducibility.
        n_jobs : int, optional
            Number of parallel jobs (requires joblib). -1 uses all cores.
            Default 1 (sequential).

        Returns
        -------
        power_level : float or ndarray
            Power threshold corresponding to the given FAP.
        """
        if method != 'bootstrap':
            raise ValueError("Only 'bootstrap' method is supported for L1 periodogram")

        probability = np.atleast_1d(np.asarray(probability, dtype=float))
        max_powers = self._bootstrap_max_powers(frequency, n_bootstrap,
                                                random_state, n_jobs=n_jobs)

        levels = np.quantile(max_powers, 1.0 - probability)
        return float(levels[0]) if levels.size == 1 else levels
