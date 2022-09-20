'''
Fit the local saturation data from individual interaction regions to the logistic growth
function. Record the avg logistic fit parameters from all of the fits into one DataFrame.
'''
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
import glob

def fit_logistic(t, rate, capacity):
    '''
    For use with scipy's curve_fit. Assumes initial population of 1.
    '''
    return capacity / (1 + (capacity - 1) * np.exp(-rate * t))

def calc_tsat(rate, capacity):
    '''
    Compute how long it will take for a region to fill up to one less than its fitted carrying capacity.
    Based on algebraic manipulation of logistic growth function when initial population is 1.
    '''
    return (-1 / rate) * np.log(1 / (capacity - 1)**2)

def logistic_fits(pvals, muvals, max_nums, return_aggregated=True):
    '''
    Fit the deme fillup data to the logistic growth curve. Return a DataFrame with the logistic growth
    parameters and an R^2 value for the fit.
    Inputs: pvals- the LIST of percentages of dispersal attempts within the local region (distance <= 1)
            muvals- the LIST of kernel exponents
            max_nums- the LIST of maximum numbers of individuals allowed within an interaction region (as
                      used in my density regulation scheme in SLiM)
            return_aggregated- return a data frame containing averaged quantities?

    Returns: df- a DataFrame that records simulation parameters and the fitted logistic growth parameters
    '''
    if return_aggregated:
        cols = ['percentage', 'mu', 'max_num', 'avg_rate', 'avg_capacity', 'avg_saturation',
                'avg_R2', 'number']
    else:
        cols = ['percentage', 'mu', 'max_num', 'rate', 'capacity', 'saturation', 'R2',
                't_sat']

    df = pd.DataFrame(columns=cols)

    for max_num in max_nums:
        for p in pvals:
            for mu in muvals:
                d = 'hets_{}/mu{}_{}local/'.format(max_num, mu, p)
                rates = []
                capacities = []
                R2vals = []
                countfiles = glob.glob(d + 'counts*.txt')
                if len(countfiles) == 0: continue
                for cfile in countfiles:
                    counts = np.loadtxt(cfile)
                    if len(counts.shape) == 1: continue # started tracking in final gen?
                    tvals = np.arange(counts.shape[0])
                    for i in range(counts.shape[1]):
                        if counts[-1, i] < 0.6 * max_num: continue # satellite must be at least pretty full
                        fitpars = curve_fit(fit_logistic, xdata=tvals, ydata=counts[:,i], p0=[1, 80])[0]

                        ss_res = np.sum((counts[:,i] - fit_logistic(tvals, *fitpars))**2)
                        ss_tot = np.sum((counts[:,i] - np.mean(counts[:,i]))**2)
                        R2 = 1 - ss_res/ss_tot

                        rates.append(fitpars[0])
                        capacities.append(fitpars[1])
                        R2vals.append(R2)

                        if not return_aggregated:
                            thisrow = pd.DataFrame(data=[[p, mu, max_num, rates[-1], capacities[-1],
                                                          capacities[-1]/max_num, R2,
                                                          calc_tsat(rates[-1], capacities[-1])]],
                                                   columns=cols)
                            df = df.append(thisrow)

                if return_aggregated:
                    df = df.append(pd.DataFrame(data=[[p, mu, max_num, np.mean(rates), np.mean(capacities),
                                                   np.mean(capacities)/max_num, np.mean(R2vals),
                                                   len(rates)]],
                                                columns=cols))

    if return_aggregated: df['t_sat'] = calc_tsat(df.avg_rate, df.avg_capacity)
    df.sort_values(by=['max_num', 'mu', 'percentage'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

if __name__ == 'main':
    ps = list(range(0, 96, 5)) + [99]
    mus = [2.5, 2.0, 1.5, 1.0]
    max_nums = [10, 50, 100]
    logistic_df = logistic_fits(ps, mus, max_nums, return_aggregated=False)
    logistic_df.to_csv('indiv_logistic_fits.csv', index=False)
