'''
Compute average growth curves and collect them all into a data frame. Write the data frame
to a csv. The growth curves as lists in the data frame are kind of heinous to work with, but
the convenience of having them all in one place might make it worthwhile. Once you isolate
the single-element series that contains the average growth curve, do procedure outlined in
comments variable.
'''

comments = [
'# working with the array entries in df can be a nuisance. procedure after isolating series with one growth curve:',
'# ell_list = series.iloc[0].split()',
"# ell_list.remove('[')        # (it has brackets on both ends)",
"# ell_list[-1] = ell_list[-1].replace(']', '')",
"# if ell_list[-1] == '': ell_list.pop()",
'# ell_arr = np.asarray(ell_list, dtype=float)     # there you have it!'
]
import numpy as np
import pandas as pd
import glob

def compute_collect_growth_curves(max_nums, mus, ps, return_tidy=True):
    '''
    Compute the average growth curve using all available simulations at each
    set of parameters. Place average growth curve into entry of data frame.

    Using `return_tidy` leads to a tidy data frame instead of a data frame with
    one row per parameter set and the average growth curve as an array within
    that row. Using `return_tidy=True` is the nicer way; refer to the comments
    above/at top of output file if not using it.
    '''
    if return_tidy:
        cols = ['max_num', 'mu', 'percent_local', 'N_sims', 'gen', 'avg_ell']
    else:
        cols = ['max_num', 'mu', 'percent_local', 'N_sims', 'avg_growth_curve']
    df = pd.DataFrame(columns=cols)

    for max_num in max_nums:
        d0 = f'consistency_{max_num}/'

        for mu in mus:
            for p in ps:
                d = d0 + f'mu{mu}_{p}local/'
                files = glob.glob(d+'pop_data*.txt')
                if len(files) == 0: continue # nothing at this set of parameters

                ells = []
                for f in files:
                    try:
                        pop_data = pd.read_csv(f)
                    except pd.errors.EmptyDataError:
                        continue
                    # skip those that didn't finish. arbitrary cutoff but works:
                    if np.max(pop_data.num_individuals) < 5e5: continue
                    ells.append(np.array(pop_data.ell))

                # standardize length of all the data sets
                min_duration = np.min([len(ell) for ell in ells])
                for i in range(len(ells)):
                    ells[i] = ells[i][:min_duration]

                ells_arr = np.vstack(ells)
                mean_ell = np.mean(ells_arr, axis=0).reshape(-1, 1)
                if return_tidy:
                    times = np.arange(1, len(mean_ell) + 1).reshape(-1, 1)
                    pars = np.array([max_num, mu, p, len(ells)]).reshape(1, -1)
                    rep_pars = np.repeat(pars, repeats=len(mean_ell), axis=0)
                    data = np.hstack([rep_pars, times, mean_ell])
                    df = df.append(pd.DataFrame(data=data, columns=cols))
                else:
                    df = df.append(pd.DataFrame(data=[[max_num, mu, p, len(ells), mean_ell]],
                                                columns=cols))

    return df

max_nums = [10, 100]
mus = [1.5, 2.0, 2.5]
ps = [0, 75, 95, 99, 99.5, 100]
df = compute_collect_growth_curves(max_nums, mus, ps, return_tidy=True)
with open('avg_growth_curves.csv', 'w') as f:
    for comm in comments:
        f.write(comm + '\n')
    df.to_csv(f, index=False)
