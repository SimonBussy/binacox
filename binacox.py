import numpy as np
import statsmodels.api as sm
from sys import stdout
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.inference import CoxRegression
from tick.preprocessing.utils import safe_array


def _all_safe(features: np.ndarray, times: np.array,
              censoring: np.array):
    if not set(np.unique(censoring)).issubset({0, 1}):
        raise ValueError('``censoring`` must only have values in {0, 1}')
    # All times must be positive
    if not np.all(times >= 0):
        raise ValueError('``times`` array must contain only non-negative '
                         'entries')
    features = safe_array(features)
    times = safe_array(times)
    censoring = safe_array(censoring, np.ushort)
    return features, times, censoring


def compute_score(learner, features, features_binarized, times, censoring,
                  blocks_start, blocks_length, boundaries, C=10, n_folds=10,
                  shuffle=True, scoring="log_lik", n_jobs=1,
                  verbose=False):
    scores = cross_val_score(learner, features, features_binarized, times,
                             censoring, blocks_start, blocks_length, boundaries,
                             n_folds=n_folds, shuffle=shuffle, C=C,
                             scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    scores = np.array(scores)
    train_scores = scores[:, 0]
    test_scores = scores[:, 1]
    train_score_mean = train_scores.mean()
    train_score_std = train_scores.std()
    test_score_mean = test_scores.mean()
    test_score_std = test_scores.std()
    if verbose:
        print("\n%s: train %0.3f (+/- %0.3f), test %0.3f (+/- %0.3f)" %
              (scoring, train_score_mean, 2 * train_score_std, test_score_mean,
               2 * test_score_std))
    return train_score_mean, train_score_std, test_score_mean, test_score_std


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
    return scores


def fit_and_score(learner, features, features_bin, times, censoring,
                  blocks_start, blocks_length, boundaries, idx_train,
                  idx_test, scoring):
    X_train, X_test = features_bin[idx_train], features_bin[idx_test]
    Y_train, Y_test = times[idx_train], times[idx_test]
    delta_train, delta_test = censoring[idx_train], censoring[idx_test]

    learner._solver_obj.linesearch = False
    learner.fit(X_train, Y_train, delta_train)

    if scoring == 'log_lik':
        train_score = learner.score(X_train, Y_train, delta_train)
        test_score = learner.score(X_test, Y_test, delta_test)
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
        learner = CoxRegression(tol=1e-5, solver=solver, verbose=False,
                                penalty='none')
        learner.fit(X_train, Y_train, delta_train)
        train_score = learner.score(X_train, Y_train, delta_train)
        test_score = learner.score(X_test, Y_test, delta_test)

    else:
        raise ValueError("scoring ``%s`` not implemented, "
                         "try using 'log_lik' instead" % scoring)

    return train_score, test_score


def get_p_value(feature, val, times, censoring):
    feature_bin = feature <= val
    mod = sm.PHReg(endog=times, status=censoring, exog=feature_bin,
                   ties="efron")
    fitted_model = mod.fit()
    p_value = fitted_model.pvalues[0]

    return p_value


def auto_cutoff(feature, times, censoring, epsilon=25):
    feature, times, censoring = _all_safe(feature, times, censoring)
    q1 = np.percentile(feature, epsilon)
    q3 = np.percentile(feature, 100 - epsilon)
    values_to_test = feature[np.where((feature <= q3) & (feature >= q1))]
    p_values = Parallel(n_jobs=-1)(delayed(get_p_value)(feature, val,
                                                        times, censoring)
                                   for val in values_to_test)

    return p_values, values_to_test
