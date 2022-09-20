'''
Collect average satellite population growth data into one large data frame.
Optionally perform logistic fits on the average local saturation data.
'''

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import glob
from all_logistic_fits import fit_logistic # args: t, rate, capacity

def collect_data(capacities, mus, ps):
    '''
    Collect all satellite population growth data into one large data frame.
    Average across satellites to get the average growth trajectory once a
    satellite has been seeded by long-range dispersal. Record time since first
    arrival, avg satellite population, number of satellites averaged over, and
    parameters.

    Filter the satellites to only keep ones that appear "full" (i.e. have
    stopped growing)

    Inputs: capacities- LIST of local carrying capacities (integers)
            mus- LIST of kernel exponents
            ps- LIST of percentages of local dispersal
    '''
    cols = ['time', 'avg_occupancy']
    ordered = ['capacity', 'mu', 'percent_local', 'time', 'avg_occupancy', 'N_satellites']
    all_df = pd.DataFrame(columns=cols)
    for cap in capacities:
        for mu in mus:
            for p in ps:
                d0 = f'hets_{cap}/mu{mu:.1f}_{p}local/'

                # filter columns
                data_list = [np.loadtxt(f) for f in glob.glob(d0 + 'counts*.txt')]
                for i in range(len(data_list)):
                    sel = []
                    if len(data_list[i].shape) == 1:
                        del data_list[i] # or remove entries as necessary
                        continue # found satellites in final gen
                    for c in range(data_list[i].shape[1]):
                        # HARD CODED LEVELLING OFF THRESHOLD TIME - same value final 4 generations
                        sel.append(np.all(data_list[i][-1, c] == data_list[i][-4:-1, c]) \
                                    & (data_list[i][-1, c] >= 0.5*cap))
                    # keep only columns (int. regions) that have levelled off at "full enough" saturation
                    data_list[i] = data_list[i][:, sel]

                # pad rows if necessary so data can be combined
                max_len = np.max([d.shape[0] for d in data_list])
                for i in range(len(data_list)):
                    pad_len = max_len - data_list[i].shape[0]
                    data_list[i] = np.pad(data_list[i], pad_width=[(0, pad_len), (0, 0)],
                                          mode='constant', constant_values=np.nan)

                stacked = np.hstack(data_list)

                # now average and add to data frame
                avg = np.mean(stacked, axis=1)
                avg = avg[~np.isnan(avg)] # drop padded rows CAREFUL

                # TO DO: truncate data once avg once it levels off for N generations?

                times = np.arange(1, len(avg)+1)
                data = np.stack([times, avg], axis=1)
                df = pd.DataFrame(data=data, columns=['time', 'avg_occupancy'])
                df['N_satellites'] = stacked.shape[1]
                df['capacity'] = cap
                df['mu'] = mu
                df['percent_local'] = p

                all_df = all_df.append(df)

    all_df = all_df[ordered]

    return all_df

def fits_to_avg(avg_df):
    '''
    Perform logistic fits to the average local saturation data stored in the
    data frame avg_df. Return a data frame with one row per set of parameters
    that contains the fit parameters and estimated std deviations from the fits.
    '''
    cols = ['capacity', 'mu', 'percent_local', 'fitted_rate', 'std_rate',
            'fitted_capacity', 'std_capacity', 'N_averaged']
    res = pd.DataFrame(columns=cols)

    for cap in np.unique(avg_df.capacity):
        for mu in np.unique(avg_df.mu):
            for per in np.unique(avg_df.percent_local):
                sel = (avg_df.capacity == cap) & (avg_df.mu == mu) & (avg_df.percent_local == per)
                data = avg_df[sel]
                times = data.time - 1
                pars, cov = curve_fit(fit_logistic, xdata=times, ydata=data.avg_occupancy, p0=[1, 80])
                stds = np.sqrt(np.diag(cov))
                this = pd.DataFrame(data=[[cap, mu, per, pars[0], stds[0], pars[1],
                                           stds[1], np.min(data.N_satellites)]],
                                    columns=cols)
                res = res.append(this)

    return res

record_avg = True
fit_to_avg = False

capacities = [10, 100]#, 50]
mus = [1.5, 2, 2.5]
# ps = list(range(0, 96, 5)) + [99]
ps = [0, 50, 99]
df = collect_data(capacities, mus, ps)
if record_avg:
    df.to_csv('logistic_data.csv', index=False)

if fit_to_avg:
    fits = fits_to_avg(df)
    fits.to_csv('logistic_fits_to_avg.csv', index=False)
