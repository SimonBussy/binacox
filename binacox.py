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


def get_m_2(cut_points_estimates, cut_points, S):
    m_2 = 0
    n_features = len(cut_points)
    for j in set(range(n_features)) - set(S):
        mu_star_j = cut_points[str(j)][1:-1]
        hat_mu_star_j = cut_points_estimates[str(j)][1:-1]
        m_2 += get_H(mu_star_j, hat_mu_star_j)

    return (1 / n_features) * m_2


def get_H(A, B):
    return max(get_E(A, B), get_E(B, A))


def get_E(A, B):
    return max([min([abs(a - b) for a in A]) for b in B])


def get_p_values_j(feature, mu_k, times, censoring, values_to_test, epsilon=10):
    if values_to_test is None:
        p1 = np.percentile(feature, epsilon)
        p2 = np.percentile(feature, 100 - epsilon)
        values_to_test = mu_k[np.where((mu_k <= p2) & (mu_k >= p1))]
    p_values, t_values = [], []
    for val in values_to_test:
        feature_bin = feature <= val
        mod = sm.PHReg(endog=times, status=censoring, exog=feature_bin,
                       ties="efron")
        fitted_model = mod.fit()
        p_values.append(fitted_model.pvalues[0])
        t_values.append(fitted_model.tvalues[0])
    p_values = pd.DataFrame({'values_to_test': values_to_test,
                             'p_values': p_values,
                             't_values': t_values})
    p_values.sort_values('values_to_test', inplace=True)

    return p_values


def auto_cutoff(X, boundaries, Y, delta, values_to_test=None,
                features_names=None):
    if values_to_test is None:
        values_to_test = X.shape[1] * [None]
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    X = np.array(X)
    result = Parallel(n_jobs=8)(
        delayed(get_p_values_j)(X[:, j],
                                boundaries[features_names[j]].copy()[1:-1], Y,
                                delta, values_to_test[j])
        for j in range(X.shape[1]))

    return result


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
    p_corr_1 = norm.pdf(1 - p_value_min / 2) * (z - 1 / z) * np.log(
        (1 - epsilon) ** 2 / epsilon ** 2) + 4 * norm.pdf(z) / z

    for i in np.arange(n_tested - 1):
        l[i] = np.count_nonzero(feature <= values_to_test_sorted[i])
        if i >= 1:
            D[i - 1] = d_ij(l[i - 1], l[i], z, feature.shape[0])
    p_corr_2 = p_value_min + np.sum(D)

    p_value_min_corrected = np.min((p_corr_1, p_corr_2, 1))
    if np.isnan(p_value_min_corrected) or np.isinf(p_value_min_corrected):
        p_value_min_corrected = p_value_min

    return p_value_min_corrected


# def bootstrap_cut_max_t(X, boundaries, Y, delta, auto_cutoff_rslt, B=10,
#                         features_names=None, verbose=False):
#     if features_names is None:
#         features_names = [str(j) for j in range(X.shape[1])]
#     n_samples, n_features = X.shape
#     t_values_init, values_to_test_init = [], []
#     for j in range(n_features):
#         t_values_init.append(auto_cutoff_rslt[j].t_values)
#         values_to_test_init.append(auto_cutoff_rslt[j].values_to_test)
#
#     n_tested = t_values_init[0].size
#     t_values_B = n_features * [np.zeros((B, n_tested))]
#
#     for b in np.arange(B):
#         if verbose:
#             stdout.write("\rBootstrap: %d%%" % ((b + 1) * 100 / B))
#             stdout.flush()
#         perm = np.random.choice(n_samples, size=n_samples, replace=True)
#         auto_cutoff_rslt_b = auto_cutoff(X[perm], boundaries, Y[perm],
#                                          delta[perm], values_to_test_init,
#                                          features_names=features_names)
#         for j in range(n_features):
#             t_values_B[j][b, :] = np.abs(auto_cutoff_rslt_b[j].t_values)
#
#     adjusted_p_values = []
#     for j in range(n_features):
#         sd = np.std(t_values_B[j], 0)
#         sd[sd < 1] = 1
#         mean_ = np.repeat(np.mean(t_values_B[j], 0), B).reshape((B, n_tested))
#         sd_ = np.repeat(1 / sd, B).reshape((B, n_tested))
#         t_val_B_H0_j = (t_values_B[j] - mean_) * sd_
#         maxT = np.max(np.abs(t_val_B_H0_j), 0)
#         adjusted_p_values.append(
#             [np.mean(maxT > np.abs(t_k)) for t_k in t_values_init[j]])
#
#     return adjusted_p_values


def bootstrap_cut_max_t(X, boundaries, Y, delta, auto_cutoff_rslt, B=10,
                        features_names=None, verbose=False):
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    n_samples, n_features = X.shape
    t_values_init, values_to_test_init, t_values_B = [], [], []
    for j in range(n_features):
        t_values_init.append(auto_cutoff_rslt[j].t_values)
        values_to_test_j = auto_cutoff_rslt[j].values_to_test
        values_to_test_init.append(values_to_test_j)
        n_tested = values_to_test_j.size
        t_values_B.append(pd.DataFrame(np.zeros((B, n_tested))))

    for b in np.arange(B):
        if verbose:
            stdout.write("\rBootstrap: %d%%" % ((b + 1) * 100 / B))
            stdout.flush()
        perm = np.random.choice(n_samples, size=n_samples, replace=True)
        auto_cutoff_rslt_b = auto_cutoff(X[perm], boundaries, Y[perm],
                                         delta[perm], values_to_test_init,
                                         features_names=features_names)
        for j in range(n_features):
            t_values_B[j].ix[b, :] = auto_cutoff_rslt_b[j].t_values.abs()

    adjusted_p_values = []
    for j in range(n_features):
        sd = t_values_B[j].std(0)
        sd[sd < 1] = 1
        mean = t_values_B[j].mean(0)
        t_val_B_H0_j = (t_values_B[j] - mean) / sd
        maxT = t_val_B_H0_j.abs().max(0)
        adjusted_p_values.append(
            [(maxT > np.abs(t_k)).mean() for t_k in t_values_init[j]])

    return adjusted_p_values
