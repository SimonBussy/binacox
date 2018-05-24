import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.inference import CoxRegression
import statsmodels.api as sm


def compute_score(learner, features, features_binarized, times, censoring,
                  blocks_start, blocks_length, boundaries, C=10, n_folds=10,
                  features_names=None, shuffle=True, n_jobs=1, verbose=False,
                  validation_data=None):
    scores = cross_val_score(learner, features, features_binarized, times,
                             censoring, blocks_start, blocks_length, boundaries,
                             n_folds=n_folds, shuffle=shuffle, C=C,
                             features_names=features_names, n_jobs=n_jobs,
                             verbose=verbose, validation_data=validation_data)
    scores_test = scores[:, 0]
    scores_validation = scores[:, 1]
    if validation_data is not None:
        scores_validation_mean = scores_validation.mean()
        scores_validation_std = scores_validation.std()
    else:
        scores_validation_mean, scores_validation_std = None, None

    scores_mean = scores_test.mean()
    scores_std = scores_test.std()
    if verbose:
        print("\nscore %0.3f (+/- %0.3f)" % (scores_mean, scores_std))
    scores = [scores_mean, scores_std, scores_validation_mean,
              scores_validation_std]
    return scores


def cross_val_score(learner, features, features_binarized, times, censoring,
                    blocks_start, blocks_length, boundaries, n_folds, shuffle,
                    C, features_names, n_jobs, verbose, validation_data):
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    cv_iter = list(cv.split(features))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    learner.C = C
    scores = parallel(
        delayed(fit_and_score)(learner, features, features_binarized, times,
                               censoring, blocks_start, blocks_length,
                               boundaries, features_names, idx_train, idx_test,
                               validation_data)
        for (idx_train, idx_test) in cv_iter)
    return np.array(scores)


def fit_and_score(learner, features, features_bin, times, censoring,
                  blocks_start, blocks_length, boundaries, features_names,
                  idx_train, idx_test, validation_data):
    if features_names is None:
        features_names = [str(j) for j in range(features.shape[1])]
    X_train, X_test = features_bin[idx_train], features_bin[idx_test]
    Y_train, Y_test = times[idx_train], times[idx_test]
    delta_train, delta_test = censoring[idx_train], censoring[idx_test]

    learner._solver_obj.linesearch = False
    learner.fit(X_train, Y_train, delta_train)

    coeffs = learner.coeffs
    cut_points_estimates = {}
    for j, start in enumerate(blocks_start):
        coeffs_j = coeffs[start:start + blocks_length[j]]
        all_zeros = not np.any(coeffs_j)
        if all_zeros:
            cut_points_estimate_j = np.array([-np.inf, np.inf])
        else:
            groups_j = get_groups(coeffs_j)
            jump_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1
            if jump_j.size == 0:
                cut_points_estimate_j = np.array([-np.inf, np.inf])
            else:
                cut_points_estimate_j = boundaries[features_names[j]][
                    jump_j]
                if cut_points_estimate_j[0] != -np.inf:
                    cut_points_estimate_j = np.insert(cut_points_estimate_j,
                                                      0, -np.inf)
                if cut_points_estimate_j[-1] != np.inf:
                    cut_points_estimate_j = np.append(cut_points_estimate_j,
                                                      np.inf)
        cut_points_estimates[features_names[j]] = cut_points_estimate_j
    binarizer = FeaturesBinarizer(method='given',
                                  bins_boundaries=cut_points_estimates)
    binarized_features = binarizer.fit_transform(features)
    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length
    X_bin_train = binarized_features[idx_train]
    X_bin_test = binarized_features[idx_test]
    solver = 'agd'
    learner = CoxRegression(penalty='binarsity', tol=1e-5,
                            solver=solver, verbose=False,
                            max_iter=100, step=0.3,
                            blocks_start=blocks_start,
                            blocks_length=blocks_length,
                            warm_start=True, C=1e10)
    learner._solver_obj.linesearch = False
    learner.fit(X_bin_train, Y_train, delta_train)
    score = learner.score(X_bin_test, Y_test, delta_test)

    if validation_data is not None:
        X_validation = validation_data[0]
        X_bin_validation = binarizer.fit_transform(X_validation)
        Y_validation = validation_data[1]
        delta_validation = validation_data[2]
        score_validation = learner.score(X_bin_validation, Y_validation,
                                         delta_validation)
    else:
        score_validation = None

    return score, score_validation


def get_groups(coeffs):
    n_coeffs = len(coeffs)
    jumps = np.where(coeffs[1:] - coeffs[:-1] != 0)[0] + 1
    jumps = np.insert(jumps, 0, 0)
    jumps = np.append(jumps, n_coeffs)
    groups = np.zeros(n_coeffs)
    for i in range(len(jumps) - 1):
        groups[jumps[i]:jumps[i + 1]] = i
        if jumps[i + 1] - jumps[i] <= 2:
            if i == 0:
                groups[jumps[i]:jumps[i + 1]] = 1
            elif i == len(jumps) - 2:
                groups[jumps[i]:jumps[i + 1]] = groups[jumps[i - 1]]
            else:
                coeff_value = coeffs[jumps[i]]
                group_before = groups[jumps[i - 1]]
                coeff_value_before = coeffs[
                    np.nonzero(groups == group_before)[0][0]]
                try:
                    k = 0
                    while coeffs[jumps[i + 1] + k] != coeffs[
                                        jumps[i + 1] + k + 1]:
                        k += 1
                    coeff_value_after = coeffs[jumps[i + 1] + k]
                except:
                    coeff_value_after = coeffs[jumps[i + 1]]
                if np.abs(coeff_value_before - coeff_value) < np.abs(
                                coeff_value_after - coeff_value):
                    groups[np.where(groups == i)] = group_before
                else:
                    groups[np.where(groups == i)] = i + 1
    return groups.astype(int)


def get_m_1(cut_points_estimates, cut_points, S):
    m_1, d = 0, 0
    n_features = len(cut_points)
    for j in set(range(n_features)) - set(S):
        mu_star_j = cut_points[str(j)][1:-1]
        hat_mu_star_j = cut_points_estimates[str(j)][1:-1]
        if len(hat_mu_star_j) > 0:
            d += 1
            m_1 += get_H(mu_star_j, hat_mu_star_j)
    if d == 0:
        m_1 = np.nan
    else:
        m_1 *= (1 / d)
    return m_1


def get_H(A, B):
    return max(get_E(A, B), get_E(B, A))


def get_E(A, B):
    return max([min([abs(a - b) for a in A]) for b in B])


def get_m_2(hat_K_star, S):
    return (1 / len(S)) * hat_K_star[S].sum()


def get_p_values_j(feature, mu_k, times, censoring, values_to_test, epsilon):
    if values_to_test is None:
        p1 = np.percentile(feature, epsilon)
        p2 = np.percentile(feature, 100 - epsilon)
        values_to_test = mu_k[np.where((mu_k <= p2) & (mu_k >= p1))]
    p_values, t_values = [], []
    for val in values_to_test:
        feature_bin = feature <= val
        mod = sm.PHReg(endog=times, status=censoring,
                       exog=feature_bin.astype(int), ties="efron")
        fitted_model = mod.fit()
        p_values.append(fitted_model.pvalues[0])
        t_values.append(fitted_model.tvalues[0])
    p_values = pd.DataFrame({'values_to_test': values_to_test,
                             'p_values': p_values,
                             't_values': t_values})
    p_values.sort_values('values_to_test', inplace=True)
    return p_values


def auto_cutoff(X, boundaries, Y, delta, values_to_test=None,
                features_names=None, epsilon=5):
    if values_to_test is None:
        values_to_test = X.shape[1] * [None]
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    X = np.array(X)
    result = Parallel(n_jobs=5)(
        delayed(get_p_values_j)(X[:, j],
                                boundaries[features_names[j]].copy()[1:-1], Y,
                                delta, values_to_test[j], epsilon=epsilon)
        for j in range(X.shape[1]))
    return result


def t_ij(i, j, n):
    return (1 - i * (n - j) / ((n - i) * j)) ** .5


def d_ij(i, j, z, n):
    return (2 / np.pi) ** .5 * norm.pdf(z) * (
        t_ij(i, j, n) - (z ** 2 / 4 - 1) * t_ij(i, j, n) ** 3 / 6)


def p_value_cut(p_values, values_to_test, feature, epsilon=5):
    n_tested = p_values.size
    p_value_min = np.min(p_values)
    l = np.zeros(n_tested)
    l[-1] = n_tested
    d = np.zeros(n_tested - 1)
    z = norm.ppf(1 - p_value_min / 2)
    values_to_test_sorted = np.sort(values_to_test)

    epsilon /= 100
    p_corr_1 = norm.pdf(1 - p_value_min / 2) * (z - 1 / z) * np.log(
        (1 - epsilon) ** 2 / epsilon ** 2) + 4 * norm.pdf(z) / z

    for i in np.arange(n_tested - 1):
        l[i] = np.count_nonzero(feature <= values_to_test_sorted[i])
        if i >= 1:
            d[i - 1] = d_ij(l[i - 1], l[i], z, feature.shape[0])
    p_corr_2 = p_value_min + np.sum(d)

    p_value_min_corrected = np.min((p_corr_1, p_corr_2, 1))
    if np.isnan(p_value_min_corrected) or np.isinf(p_value_min_corrected):
        p_value_min_corrected = p_value_min
    return p_value_min_corrected


def auto_cutoff_perm(n_samples, X, boundaries, Y, delta, values_to_test_init,
                     features_names, epsilon):
    np.random.seed()
    perm = np.random.choice(n_samples, size=n_samples, replace=True)
    auto_cutoff_rslt = auto_cutoff(X[perm], boundaries, Y[perm],
                                   delta[perm], values_to_test_init,
                                   features_names=features_names,
                                   epsilon=epsilon)
    return auto_cutoff_rslt


def bootstrap_cut_max_t(X, boundaries, Y, delta, auto_cutoff_rslt, B=10,
                        features_names=None, epsilon=5):
    if features_names is None:
        features_names = [str(j) for j in range(X.shape[1])]
    n_samples, n_features = X.shape
    t_values_init, values_to_test_init, t_values_B = [], [], []
    for j in range(n_features):
        t_values_init.append(auto_cutoff_rslt[j].t_values)
        values_to_test_j = auto_cutoff_rslt[j].values_to_test
        values_to_test_init.append(values_to_test_j)
        n_tested_j = values_to_test_j.size
        t_values_B.append(pd.DataFrame(np.zeros((B, n_tested_j))))

    result = Parallel(n_jobs=10)(
        delayed(auto_cutoff_perm)(n_samples, X, boundaries, Y, delta,
                                  values_to_test_init, features_names, epsilon)
        for _ in np.arange(B))

    for b in np.arange(B):
        for j in range(n_features):
            t_values_B[j].ix[b, :] = result[b][j].t_values

    adjusted_p_values = []
    for j in range(n_features):
        sd = t_values_B[j].std(0)
        sd[sd < 1] = 1
        mean = t_values_B[j].mean(0)
        t_val_B_H0_j = (t_values_B[j] - mean) / sd
        maxT_j = t_val_B_H0_j.abs().max(1)
        adjusted_p_values.append(
            [(maxT_j > np.abs(t_k)).mean() for t_k in t_values_init[j]])
    return adjusted_p_values
