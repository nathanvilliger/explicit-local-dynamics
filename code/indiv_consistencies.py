'''
Make fits to the consistency condition and infer the kernel exponent for all available
*individual* simulations. Fitting for the power that matches l(t)^<power> ~ t l(t/2)^(2d).
The power should be (d + \mu) according to consistency condition.

Consistency condition is only valid at longer times, so the t values I use in the
fits begin halfway through the growth curves. t/2 then goes from 1/4 to 1/2 through
the duration of the growth curve.

Record all inferred kernel exponents and relevant parameters to a data frame.
Write to csv. Also perform bootstrap resampling of inferred kernel exponents to get
a better understanding of the distribution of inferred kernel exponents at each set
of parameters.
'''
import numpy as np
import pandas as pd
import glob
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os

def power_fit(x, a, b):
    return a * x**b

def infer_mus(percentages, mus=[2.5, 2.0, 1.5], max_nums=[10, 100]):
    '''
    Infer and record kernel exponent for every single growth curve I have available at each
    combination of parameters.
    '''
    cols = ['max_num', 'mu', 'percent_local', 'filename', 'inferred', 'R2', 'lm_inferred', 'lm_R2',
            'max generation', 'number of x values in the fit']
    results = pd.DataFrame(columns=cols)
    Nbad = 0

    for max_num in max_nums:
        d0 = f'consistency_{max_num}/'

        for p in percentages:
            for mu in mus:
                d = d0 + f'mu{mu}_{p}local/'

                # or longer sims?
                ref_version = f'consistency_{max_num}_longer/mu{mu}_{p}local/'
                if os.path.exists(ref_version):
                    d = ref_version
                    cutoff_threshold = 15e6
                else:
                    cutoff_threshold = 5e5

                pop_datas = glob.glob(d + 'pop_data*.txt')
                for f in pop_datas:
                    # pop_data = pd.read_csv(f)
                    try:
                        pop_data = pd.read_csv(f)
                        # print('good')
                    except pd.errors.EmptyDataError:
                        # print('bad -', f)
                        Nbad += 1
                        continue

                    if np.max(pop_data.num_individuals) < cutoff_threshold: continue # sim got cut off
                    times = np.array(pop_data.generation)
                    sizes = np.array(pop_data.ell)
                    # sizes = np.sqrt(np.array(pop_data.num_individuals) / np.pi)
                    # if len(times) == 1: continue

                    even_gens = (times % 2 == 0)
                    later = (times >= times.max()/2)
                    later_even_times = np.array(times[even_gens & later], dtype=int)
                    whole_args = even_gens & later

                    half_times = np.array(later_even_times / 2, dtype=int)
                    half_args = np.squeeze(np.concatenate([np.argwhere(times == half_time) for half_time in half_times]))

                    xvals = sizes[whole_args]
                    yvals = later_even_times * sizes[half_args]**4
                    fit_pars = curve_fit(f=power_fit, xdata=xvals, ydata=yvals, maxfev=400000)[0]
                    inferred = fit_pars[1] - 2 # because we fit for mu + d

                    ypred = power_fit(xvals, *fit_pars)
                    SS_res = np.sum((yvals - ypred)**2)
                    SS_tot = np.sum((yvals - np.mean(yvals))**2)
                    R2 = 1 - SS_res / SS_tot

                    lm = linregress(np.log(xvals), np.log(yvals))
                    lm_inferred = lm.slope - 2 # because we fit for mu + d
                    lm_R2 = lm.rvalue**2

                    df = pd.DataFrame(data=[[max_num, mu, p, f, inferred, R2, lm_inferred, lm_R2,
                                             np.max(times), len(xvals)]],
                                      columns=cols)
                    results = results.append(df)

    results.sort_values(by=['max_num', 'mu', 'percent_local'], inplace=True)
    results.reset_index(drop=True, inplace=True)
    # print(f'The number of empty pop_data files was {Nbad}')
    return results

# ps = sorted([1] + list(range(0, 81, 5)) + [82, 85, 88] + list(range(90, 100)) + [98.5, 99.5, 99.7])
ps = list(range(0, 96, 5)) + [99, 99.5, 99.7]
max_nums = [10, 100]

# do the individual inferences
df = infer_mus(ps, max_nums=max_nums)
df.to_csv('indiv_consistencies_w_lm_and_longer.csv', index=False)
