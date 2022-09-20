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

def power_fit(x, a, b):
    return a * x**b

def infer_mus(percentages, mus=[2.5, 2.0, 1.5], max_nums=[10, 100]):
    '''
    Infer and record kernel exponent for every single growth curve I have available at each
    combination of parameters.
    '''
    cols = ['max_num', 'mu', 'percent_local', 'inferred']
    results = pd.DataFrame(columns=cols)
    Nbad = 0

    for max_num in max_nums:
        d0 = f'consistency_{max_num}/'

        for p in percentages:
            for mu in mus:
                d = d0 + f'mu{mu}_{p}local/'

                pop_datas = glob.glob(d + 'pop_data*.txt')
                for f in pop_datas:
                    # pop_data = pd.read_csv(f)
                    try:
                        pop_data = pd.read_csv(f)
                        # print('good')
                    except pd.errors.EmptyDataError:
                        print('bad -', f)
                        Nbad += 1
                        continue

                    if np.max(pop_data.num_individuals) < 5e5: continue # sim got cut off
                    times = np.array(pop_data.generation)
                    sizes = np.array(pop_data.ell)

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

                    df = pd.DataFrame(data=[[max_num, mu, p, inferred]], columns=cols)
                    results = results.append(df)

    results.sort_values(by=['max_num', 'mu', 'percent_local'], inplace=True)
    results.reset_index(drop=True, inplace=True)
    print(f'The number of empty pop_data files was {Nbad}')
    return results

def boots(x, n_sets=10000, col_name='inferred'):
    '''
    Perform bootstrap resampling of data. Do n_sets of n_samples samples with replacement
    from my data set, where n_samples is the same as the number of samples in x.
    Return mean and standard deviation of set means.
    To be used as DataFrame.groupby().apply(boots) with appropriately specified keys within groupby().

    **x is a data frame of group values.** x is automatically fed in when function used as above.
    '''
    n_samples = len(x[col_name])
    samps = np.random.choice(x[col_name], size=(n_samples, n_sets), replace=True)
    boot_means = np.mean(samps, axis=0)
    x['boot_mean'] = np.mean(boot_means)
    x['boot_se'] = np.std(boot_means)

    # end positions of 95% confidence interval for error bars
    x['boot_lower'] = np.percentile(boot_means, 2.5)
    x['boot_upper'] = np.percentile(boot_means, 97.5)

    return x

ps = list(range(0, 96, 5)) + [99, 99.5, 99.7]
max_nums = [10, 100]

# do the individual inferences
df = infer_mus(ps, max_nums=max_nums)
df.to_csv('indiv_consistencies.csv', index=False)

# now do the bootstrap resampling
indiv_boots = df.groupby(['max_num', 'mu', 'percent_local']).apply(boots)

# and collapse to simple data frame with one entry per parameter group.
# compute mean and std of indiv inferences.
columns = list(indiv_boots.columns)
columns.remove('inferred')
boot_means = indiv_boots.groupby(columns).aggregate(['count', 'mean', 'median', 'std'])['inferred'].reset_index()
boot_means['SEM'] = boot_means['std'] / np.sqrt(boot_means['count'])

# compute length of lower and upper portions of error bars from mean of individual inferences
boot_means['low_ebar'] = np.abs(boot_means['mean'] - boot_means['boot_lower'])
boot_means['high_ebar'] = np.abs(boot_means['mean'] - boot_means['boot_upper'])

boot_means.to_csv('boot_consistencies.csv', index=False)
