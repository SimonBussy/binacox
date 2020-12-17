import os
import sys

sys.path.append('..')
import pandas as pd
import numpy as np
import pylab as pl
from binacox import multiple_testing, p_value_cut, get_groups, \
    plot_screening, refit_and_predict
from sys import stdout
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.survival import CoxRegression
from lifelines.utils import concordance_index
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from tcga.scoring import estim_proba, integrated_brier_score
import warnings

warnings.filterwarnings('ignore')

##
verbose = False

# Load data
cancers = ["GBM", "KIRC", "BRCA"]
data = {}
scaler = StandardScaler()
for cancer in cancers:
    inputdir = "./data/%s/rna/" % cancer
    X = pd.read_csv(inputdir + 'X.csv')
    X = X.fillna(X.mean())
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    Y = np.ravel(pd.read_csv(inputdir + 'T.csv')) + 1
    delta = np.ravel(pd.read_csv(inputdir + 'delta.csv'))
    data[cancer] = {"X": X, "Y": Y, "delta": delta}
    print("%s loaded: n=%s, p=%s" % (cancer, X.shape[0], X.shape[1]))

##
test_size = .3  # 30% for testing
B = 100  # number of runs  100
P = 50  # top-P features
C_chosen = {"GBM": 17, "KIRC": 5, "BRCA": 15.7}  # from external cross-val

# load required R libraries
ro.r('library(CoxBoost)')
ro.r('library(randomForestSRC)')
pandas2ri.activate()

columns = ['Cancer', 'Model', 'C-index', 'ibs']
result = pd.DataFrame(columns=columns)
for cancer in cancers:
    X = data[cancer]["X"]
    Y = data[cancer]["Y"]
    delta = data[cancer]["delta"]

    for b in range(B):
        print("\n%s: %s/%s" % (cancer, b + 1, B))

        # 1) randomly split data into training and test sets
        rs = ShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in rs.split(X):
            X_test = X.iloc[test_index, :]
            Y_test = Y[test_index]
            delta_test = delta[test_index]

            X_train = X.iloc[train_index, :]
            Y_train = Y[train_index]
            delta_train = delta[train_index]

        # 2) screening cox, top-P features
        n_features = X_train.shape[1]
        screening_cox = pd.Series(index=X_train.columns)
        learner = CoxRegression(tol=1e-5, solver='agd', verbose=False,
                                penalty='none', max_iter=100)

        for j in range(n_features):
            stdout.write("\rscreening: %d/%s" % (j + 1, n_features))
            stdout.flush()

            feat_name = X_train.columns[j]
            X_j = X_train[[feat_name]]
            learner.fit(X_j, Y_train, delta_train)
            coeffs = learner.coeffs
            marker = X_j.dot(coeffs)
            c_index = concordance_index(Y_train, marker, delta_train)
            c_index = max(c_index, 1 - c_index)
            screening_cox[feat_name] = c_index

        screening_cox.sort_values(ascending=False, inplace=True)

        if verbose:
            plot_screening("cox", screening_cox, cancer, P)

        screening_cox_topP = screening_cox[:P].index

        # 3) train models
        # Cox PH on original data
        print("\nTrain Cox PH...")
        X_train_ = X_train[screening_cox_topP]
        X_test_ = X_test[screening_cox_topP]

        learner = CoxRegression(tol=1e-5, solver='agd', verbose=False,
                                penalty='none', max_iter=100)
        learner.fit(X_train_, Y_train, delta_train)
        coeffs = learner.coeffs
        marker_cox = np.array(X_test_.dot(coeffs))
        lp_train = np.array(X_train_.dot(coeffs))
        c_index = concordance_index(Y_test, marker_cox, delta_test)
        c_index_continuous = max(c_index, 1 - c_index)

        predictions = estim_proba(marker_cox, lp_train, Y_train, delta_train)
        ibs_cox = integrated_brier_score(predictions['values'], predictions['times'],
                               Y_test, delta_test, Y_train, delta_train)

        # Binacox
        print("Train Binacox screening_cox_topP...")

        X_train_ = X_train[screening_cox_topP]
        X_test_ = X_test[screening_cox_topP]

        # binarize feature
        n_cuts = 50
        binarizer = FeaturesBinarizer(n_cuts=n_cuts)

        binarizer.fit(pd.concat([X_train_, X_test_]))
        X_bin_train = binarizer.transform(X_train_)
        blocks_start = binarizer.blocks_start
        blocks_length = binarizer.blocks_length
        boundaries = binarizer.boundaries

        # fit binacox
        learner = CoxRegression(penalty='binarsity', tol=1e-5,
                                solver='agd', verbose=False,
                                max_iter=100, step=0.3,
                                blocks_start=blocks_start,
                                blocks_length=blocks_length,
                                C=C_chosen[cancer], warm_start=True)
        learner._solver_obj.linesearch = False
        learner.fit(X_bin_train, Y_train, delta_train)
        coeffs = learner.coeffs

        X_bin_test = binarizer.transform(X_test_)
        marker2 = X_bin_test.dot(coeffs)
        c_index = concordance_index(Y_test, marker2, delta_test)
        c_index_bina1 = max(c_index, 1 - c_index)

        if verbose:
            fig = pl.figure(figsize=(50, 3))
            ax = fig.add_subplot(111)
            ax.stem(coeffs, linefmt='b', markerfmt='bo',
                    label=r"$\beta^\star_{j,k}$")
            ax.set_xlim([-5, len(coeffs) + 5])
            n_coeffs_cum = 0
            for i in range(50 - 1):
                n_coeffs = blocks_length[i]
                label = ''
                if i == 0:
                    label = r'$j$-blocks'
                ax.axvline(n_coeffs_cum + n_coeffs - .5, c='m',
                           ls='--', alpha=.8, lw=1, label=label)
                n_coeffs_cum += n_coeffs
            pl.tight_layout()
            pl.show()

        # get cut points estimates
        all_groups = list()
        cut_points_estimates = {}
        for j, start in enumerate(blocks_start):
            coeffs_j = coeffs[start:start + blocks_length[j]]
            all_zeros = not np.any(coeffs_j)
            if all_zeros:
                cut_points_estimate_j = np.array([-np.inf, np.inf])
                groups_j = blocks_length[j] * [0]
            else:
                groups_j = get_groups(coeffs_j)
                jumps_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1

                if len(jumps_j) == 0:
                    cut_points_estimate_j = np.array([-np.inf, np.inf])
                else:
                    cut_points_estimate_j = boundaries[X_train_.columns[j]][
                        jumps_j]
                    if cut_points_estimate_j[0] != -np.inf:
                        cut_points_estimate_j = np.insert(cut_points_estimate_j,
                                                          0,
                                                          -np.inf)
                    if cut_points_estimate_j[-1] != np.inf:
                        cut_points_estimate_j = np.append(cut_points_estimate_j,
                                                          np.inf)
            cut_points_estimates[X_train_.columns[j]] = cut_points_estimate_j
            if j > 0:
                groups_j += np.max(all_groups) + 1
            all_groups += list(groups_j)

        # final binacox refit
        c_index_bina, marker_bina, lp_train = refit_and_predict(cut_points_estimates, X_train_,
                                         X_test_, Y_train, delta_train, Y_test,
                                         delta_test)
        predictions = estim_proba(marker_bina, lp_train, Y_train, delta_train)
        ibs_bina = integrated_brier_score(predictions['values'],
                                         predictions['times'],
                                         Y_test, delta_test, Y_train,
                                         delta_train)

        # Multiple testing method
        print("Train Multiple testing method...")
        epsilon = 10
        multiple_testing_rslt = multiple_testing(X_train_, boundaries, Y_train,
                                                 delta_train, epsilon=epsilon,
                                                 features_names=screening_cox_topP)

        # Lausen & Schumacher correction
        p_values_corr = []
        X_ = np.array(X_train_)
        n_features = X_train_.shape[1]
        for j in range(n_features):
            p_values_j = multiple_testing_rslt[j]
            p_values_corr.append(
                p_value_cut(p_values_j.p_values, p_values_j.values_to_test,
                            X_[:, j], epsilon=epsilon))

        # Get estimated cut-points
        cut_points_estimates_MT_B, cut_points_estimates_MT_LS = {}, {}
        for j in range(n_features):
            p_values_j = multiple_testing_rslt[j]
            p_values_j_min = p_values_j.p_values.min()
            p_values_j_argmin = p_values_j.p_values.argmin()
            cut_pts_j = p_values_j.values_to_test[p_values_j_argmin]

            alpha = .05
            # Bonferroni detection
            n_tested = len(p_values_j.values_to_test)
            est_j = [-np.inf, np.inf]
            if p_values_j_min < alpha / n_tested:
                est_j.insert(1, cut_pts_j)
            cut_points_estimates_MT_B[X_train_.columns[j]] = np.array(est_j)

            # Lausen & Schumacher detection
            est_j = [-np.inf, np.inf]
            if p_values_corr[j] < alpha:
                est_j.insert(1, cut_pts_j)
            cut_points_estimates_MT_LS[X_train_.columns[j]] = np.array(est_j)

        c_index_MT_B, marker_MT_B, lp_train = refit_and_predict(cut_points_estimates_MT_B,
                                                      X_train_,
                                                      X_test_, Y_train,
                                                      delta_train, Y_test,
                                                      delta_test)
        predictions = estim_proba(marker_MT_B, lp_train, Y_train, delta_train)
        ibs_MT_B = integrated_brier_score(predictions['values'],
                                          predictions['times'],
                                          Y_test, delta_test, Y_train,
                                          delta_train)

        c_index_MT_LS, marker_MT_LS, lp_train = refit_and_predict(
            cut_points_estimates_MT_LS, X_train_,
            X_test_, Y_train, delta_train, Y_test,
            delta_test)
        predictions = estim_proba(marker_MT_LS, lp_train, Y_train, delta_train)
        ibs_MT_LS = integrated_brier_score(predictions['values'],
                                          predictions['times'],
                                          Y_test, delta_test, Y_train,
                                          delta_train)

        # Add CoxBoost and RSF
        print("Train CoxBoost...")
        ro.globalenv['X'] = X_train_
        ro.globalenv['Y'] = Y_train
        ro.globalenv['delta'] = delta_train
        ro.globalenv['X_test'] = X_test_
        ro.globalenv['Y_test'] = Y_test
        ro.globalenv['delta_test'] = delta_test
        ro.r('cbfit <- CoxBoost(Y, delta, as.matrix(X), stepno=300, '
             'penalty=100)')
        lp_train = ro.r('predict(cbfit, type="lp")')
        marker_coxBoost = ro.r('predict(cbfit, X_test, '
                               'Y_test, delta_test, type="lp")')

        c_index_coxBoost = concordance_index(Y_test, marker_coxBoost[0],
                                             delta_test)
        c_index_coxBoost = max(c_index_coxBoost, 1 - c_index_coxBoost)

        predictions = estim_proba(marker_coxBoost[0], lp_train[0], Y_train, delta_train)
        ibs_coxBoost = integrated_brier_score(predictions['values'],
                                           predictions['times'],
                                           Y_test, delta_test, Y_train,
                                           delta_train)

        print("Train RSF...")
        ro.r('data_train <- as.data.frame(X)')
        ro.r('data_train["time"] <- Y')
        ro.r('data_train["status"] <- delta')
        ro.r('data_test <- as.data.frame(X_test)')
        ro.r('data_test["time"] <- Y_test')
        ro.r('data_test["status"] <- delta_test')
        ro.r('rsf <- rfsrc(Surv(time, status) ~ ., data=data_train, '
             'ntree=200)')
        ro.r('rsf.pred <- predict(rsf, data_test)')
        marker_rsf = ro.r('rsf.pred$predicted')
        c_index_rsf = concordance_index(Y_test, marker_rsf, delta_test)
        c_index_rsf = max(c_index_rsf, 1 - c_index_rsf)

        pred_survival = ro.r('rsf.pred$survival')
        pred_times = ro.r('rsf.pred$time.interest')
        ibs_rsf = integrated_brier_score(pred_survival.T, pred_times, Y_test, delta_test,
                               Y_train, delta_train)

        tmp = np.array(
            [[cancer, 'Continuous data', c_index_continuous, ibs_cox],
             [cancer, 'Binacox', c_index_bina, ibs_bina],
             [cancer, 'MT-B', c_index_MT_B, ibs_MT_B],
             [cancer, 'MT-LS', c_index_MT_LS, ibs_MT_LS],
             [cancer, 'CoxBoost', c_index_coxBoost, ibs_coxBoost],
             [cancer, 'RSF', c_index_rsf, ibs_rsf]]
            )

        result = result.append(pd.DataFrame(tmp, columns=columns))

        if b % 10 == 0:
            result.to_excel("./result.xlsx", index=False)

os.system('say "computation finished"')

#################
# get final result

result.to_excel("./result.xlsx", index=False)

mean_res = result.groupby(['Cancer', 'Model']).mean()
mean_res.columns = ['C-index mean']
std_res = result.groupby(['Cancer', 'Model']).std()
std_res.columns = ['C-index std']

final_result = pd.concat((mean_res, std_res), axis=1)
final_result.to_excel("./risk_prediction_results.xlsx", index=False)
