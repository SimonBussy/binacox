import numpy as np
import pandas as pd
import statsmodels.api as sm
from sys import stdout
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.inference import CoxRegression
from tick.preprocessing.utils import safe_array


def compute_score(learner, features, features_binarized, times, censoring,
                  blocks_start, blocks_length, boundaries, C=10, n_folds=10,
                  shuffle=True, scoring="log_lik", n_jobs=1,
                  verbose=False):
    scores = cross_val_score(learner, features, features_binarized, times,
                             censoring, blocks_start, blocks_length, boundaries,
                             n_folds=n_folds, shuffle=shuffle, C=C,
                             scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    scores_mean = scores.mean()
    scores_std = scores.std()
    if verbose:
        print("\n%s: score %0.3f (+/- %0.3f)" % (
            scoring, scores_mean, scores_std))
    return scores_mean, scores_std


def cross_val_score(learner, features, features_binarized, times, censoring,
                    blocks_start, blocks_length, boundaries, n_folds, shuffle,
                    C, scoring, n_jobs, verbose):
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    cv_iter = list(cv.split(features))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    learner.C = C
    scores = parallel(
        delayed(fit_and_score)(learner, features, features_binarized, times,
                               censoring, blocks_start, blocks_length,
                               boundaries, idx_train, idx_test, scoring)
        for (idx_train, idx_test) in cv_iter)
    return np.array(scores)


def fit_and_score(learner, features, features_bin, times, censoring,
                  blocks_start, blocks_length, boundaries, idx_train,
                  idx_test, scoring):
    X_train, X_test = features_bin[idx_train], features_bin[idx_test]
    Y_train, Y_test = times[idx_train], times[idx_test]
    delta_train, delta_test = censoring[idx_train], censoring[idx_test]

    learner._solver_obj.linesearch = False
    learner.fit(X_train, Y_train, delta_train)

    if scoring == 'log_lik':
        score = learner.score(X_test, Y_test, delta_test)
    elif scoring == 'log_lik_refit':
        coeffs = learner.coeffs
        cut_points_estimates = {}

        for i, start in enumerate(blocks_start):
            coeffs_ = coeffs[start:start + blocks_length[i]]

            all_zeros = not np.any(coeffs_)
            if all_zeros:
                cut_points_estimate = np.array([-np.inf, np.inf])
            else:
                coeffs_ = np.array([np.arange(len(coeffs_)), coeffs_]).T
                bgm = BGM(n_components=8, covariance_type='full',
                          max_iter=20, n_init=1)
                bgm.fit(coeffs_)
                groups = bgm.predict(coeffs_)
                jump = np.where(groups[1:] - groups[:-1] != 0)[0] + 1

                if len(jump) == 0:
                    cut_points_estimate = np.array([-np.inf, np.inf])
                else:
                    cut_points_estimate = boundaries[str(i)][jump]
                    if cut_points_estimate[0] != -np.inf:
                        cut_points_estimate = np.insert(cut_points_estimate, 0,
                                                        -np.inf)
                    if cut_points_estimate[-1] != np.inf:
                        cut_points_estimate = np.append(cut_points_estimate,
                                                        np.inf)
            cut_points_estimates[str(i)] = cut_points_estimate

        binarizer = FeaturesBinarizer(method='given',
                                      bins_boundaries=cut_points_estimates)
        binarized_features = binarizer.fit_transform(features)
        X_train = binarized_features[idx_train]
        X_test = binarized_features[idx_test]

        solver = 'agd'
        learner_final = CoxRegression(tol=1e-5, solver=solver, verbose=False,
                                      penalty='none')
        learner_final.fit(X_train, Y_train, delta_train)
        score = learner_final.score(X_test, Y_test, delta_test)

    else:
        raise ValueError("scoring ``%s`` not implemented, "
                         "try using 'log_lik' instead" % scoring)

    return score


def get_m_1(hat_K_star, K_star, n_features):
    return (1 / n_features) * np.linalg.norm(hat_K_star - K_star, ord=1)


def get_m_2(cut_points_estimates, cut_points):
    m_2 = 0
    n_features = len(cut_points)
    for j in range(n_features):
        mu_star_j = cut_points[str(j)][1:-1]
        hat_mu_star_j = cut_points_estimates[str(j)][1:-1]
        m_2 += get_H(mu_star_j, hat_mu_star_j)

    return (1 / n_features) * m_2


def get_H(A, B):
    return max(get_E(A, B), get_E(B, A))


def get_E(A, B):
    return max([min([abs(a - b) for a in A]) for b in B])


def get_p_values_j(feature, mu_jk, times, censoring, epsilon=10):
    q1 = np.percentile(feature, epsilon)
    q3 = np.percentile(feature, 100 - epsilon)
    values_to_test = mu_jk[np.where((mu_jk <= q3) & (mu_jk >= q1))]
    p_values = []
    for val in values_to_test:
        feature_bin = feature <= val
        mod = sm.PHReg(endog=times, status=censoring, exog=feature_bin,
                       ties="efron")
        fitted_model = mod.fit()
        p_values.append(fitted_model.pvalues[0])

    p_values_j = pd.DataFrame({'values_to_test': values_to_test,
                               'p_values': p_values})
    p_values_j.sort_values('values_to_test', inplace=True)

    return p_values_j


def auto_cutoff(X, boundaries, Y, delta):
    p_values = Parallel(n_jobs=8)(
        delayed(get_p_values_j)(X[:, j], boundaries[str(j)].copy()[1:-1], Y,
                                delta)
        for j in range(X.shape[1]))

    return p_values


def t_ij(i, j, n):
    return (1 - i * (n - j) / ((n - i) * j)) ** .5


def d_ij(i, j, z, n):
    return (2 / np.pi) ** .5 * norm.pdf(z) * (
        t_ij(i, j, n) - (z ** 2 / 4 - 1) * t_ij(i, j, n) ** 3 / 6)


def p_value_cut(p_values, values_to_test, feature, epsilon=10):
    n_tested = p_values.size
    p_value_min = np.min(p_values)
    l = np.zeros(n_tested)
    l[-1] = n_tested
    D = np.zeros(n_tested - 1)
    z = norm.ppf(1 - p_value_min / 2)
    values_to_test_sorted = np.sort(values_to_test)

    epsilon /= 100
    p_corrected_1 = norm.pdf(1 - p_value_min / 2) * (z - 1 / z) * np.log(
        (1 - epsilon) ** 2 / epsilon ** 2) + 4 * norm.pdf(z) / z

    for i in np.arange(n_tested - 1):
        l[i] = np.count_nonzero(feature <= values_to_test_sorted[i])
        if i >= 1:
            D[i - 1] = d_ij(l[i - 1], l[i], z, feature.shape[0])
    p_corrected_2 = p_value_min + np.sum(D)

    p_corrected = np.min((p_corrected_1, p_corrected_2, 1))
    if np.isnan(p_corrected) or np.isinf(p_corrected):
        p_corrected = p_value_min

    return p_corrected
