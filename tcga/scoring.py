import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def breslow(lp_train, Y_train, delta_train):
    """
    Breslow estimator of the cumulative baseline hazard
    lp_train array of linear predictor on the train
    Y censored times
    delta censoring indicator
    """
    data = pd.DataFrame.from_dict(
        {'Y': Y_train, 'delta': delta_train, 'exp_lp': np.exp(lp_train)})
    data = data.sort_values(by=['Y'])
    data_unique = data.groupby(['Y']).sum()
    denom = np.flip(np.cumsum(np.flip(data_unique['exp_lp'])))
    breslow = np.cumsum(data_unique['delta'] / denom)
    return ({'times': data_unique.index, 'values': breslow})


def estim_proba(lp_test, lp_train, Y_train, delta_train):
    """
    estimator of the proba of being alive at t on test
    lp_test array of linear predictor on the test
    lp_train array of linear predictor on the train
    Y censored times
    delta censoring indicator
    """
    breslow_estim = breslow(lp_train, Y_train, delta_train)
    n_times = len(breslow_estim['times'])
    n_ind = len(lp_test)
    values = np.exp(-np.kron(breslow_estim['values'],
                             np.exp(lp_test)).reshape(
        (n_times, n_ind)))  # times in rows, individuals in columns
    return ({'times': np.sort(breslow_estim['times']), 'values': values})


def brier_score(t, probas_test, times, Y_test, delta_test, Y_train,
                delta_train):
    """
    estimator of the Brier score at t on test
    probas_test is a n_times x n_test   array of predicted probas
    times array of n_times where the probas are evaluated
    Y_test censored train times
    delta_test censoring train indicator
    Y_train censored train times
    delta_train censoring train indicator
    """
    data = pd.DataFrame.from_dict({'Y_test': Y_test, 'delta_test': delta_test})
    times = np.sort(np.unique(times))
    place_of_t = np.max(np.where((t >= times)))
    KM_censoring = KaplanMeierFitter().fit(Y_train,
                                           1 - delta_train).survival_function_
    data['predict'] = probas_test[place_of_t, :]

    data['value_uncensored'] = np.power(data['predict'], 2) * (
                data['Y_test'] <= t) * data['delta_test']
    data['value_censored'] = np.power(1 - data['predict'], 2) * (
                data['Y_test'] > t)

    data_uncensored = data.groupby(['Y_test']).sum()
    data_uncensored = data_uncensored.sort_values(by=['Y_test'])
    data_uncensored['value_uncensored'] = data_uncensored['value_uncensored'] / \
                                          KM_censoring['KM_estimate']
    data_uncensored['value_censored'] = data_uncensored['value_censored'] / \
                                        np.array(KM_censoring['KM_estimate'])[
                                            place_of_t]
    brier_score = np.sum(np.mean(data_uncensored['value_uncensored']) + np.mean(
        data_uncensored['value_censored']))
    return brier_score


def integrated_brier_score(probas_test, times, Y_test, delta_test, Y_train,
                           delta_train, tau=None, decile=0.9, grid_size=10):
    """
    estimator of the integrated Brier score
    probas is a n_times x n_test   array of predicted probas
    times array of n_times where the probas are evaluated
    Y censored times
    delta censoring indicator
    tau : time horizon if missing
    decile : decile of np.unique(Y)is taken as tau
    grid_size : time grid for the approximation of the integral
    """
    unique_times = np.sort(np.unique(times))
    n_obs = len(unique_times)
    if tau is None:
        tau = unique_times[int(np.ceil(n_obs * decile))]
    briers = np.zeros(grid_size)
    k = 0
    for t in np.linspace(np.min(unique_times), tau, grid_size):
        briers[k] = brier_score(t, probas_test, times, Y_test, delta_test,
                                Y_train, delta_train)
        k += 1
    intergrated_brier_score = (tau / grid_size) * np.sum(briers)
    return intergrated_brier_score / n_obs
