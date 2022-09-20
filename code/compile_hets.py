'''
Compute and record the average heterozygosity in the distant satellites I'm tracking for
logistic growth. In these simulations, I gave everyone in the initial population a *distinct*
neutral mutation, so that if a satellite is seeded a second time it is likely to be by an
individual with a different allele than the initial pioneer.
'''

import numpy as np
import pandas as pd
import glob

def compile_hets(percentages, mus, max_nums, req_full_enough=False):
    '''
    Compute the average heterozygosity over all of the distant satellites I'm tracking
    in the final generation of simulations.
    Inputs: percentages- the list of percentages of local dispersal
            mus- the list of kernel exponents
            max_nums- the list of maximum numbers allowed in the interaction region of a
                      new offspring (SLiM density scheme). This also tells where the files
                      are stored.
            req_full_enough- True or False: I should only record heterozygosity from satellites
                             that are at least 60% full?
    Outputs: het_df- a DataFrame that collects the relevant parameters and data from all sims
    '''
    cols = ['max_num', 'max_density', 'percent local', '$\mu$', 'avg H', 'normed', 'max_normed',
            'num. satellites', 'avg occupancy (in full enough)', 'num satellites (full enough)']
    het_df = pd.DataFrame(columns=cols)

    for max_num in max_nums:
        d0 = f'hets_{max_num}/'
        if type(max_num) is int: N_c = max_num
        else: N_c = int(max_num.split('_')[0])
        H_norm = 2 * (1 / N_c) * (1 - 1 / N_c)

        for mu in mus:
            for p in percentages:
                d1 = f'mu{mu}_{p}local/'
                N_sats = 0
                all_hets = np.array([])
                counts = []
                hetfiles = sorted(glob.glob(d0+d1+'hets*.txt'))
                countfiles = sorted(glob.glob(d0+d1+'counts*.txt'))
                for hfile, cfile in zip(hetfiles, countfiles): # sim setup forces these to be same length
                    het = np.loadtxt(hfile)
                    final_counts = np.loadtxt(cfile)[-1]
                    if len(het.shape) == 1: continue # found the satellites in final generation
                    N_sats += het.shape[1]
                    final_hets = het[-1]
                    if req_full_enough:
                        full_enough = final_counts >= 0.6 * N_c
                        final_hets = final_hets[full_enough]
                        final_counts = final_counts[full_enough]

                    all_hets = np.concatenate([all_hets, final_hets])
                    counts.append(final_counts)
                    counts_arr = np.concatenate(counts)

                if len(hetfiles) > 0:
                    params = pd.read_csv(glob.glob(d0+d1+'parameters*.txt')[0])
                    density = float(params.max_density)
                    avg_H = np.mean(all_hets)
                    normed = avg_H / H_norm
                    max_normed = avg_H / (1 - 1/N_c)
                    df = pd.DataFrame(data=[[N_c, density, p, mu, avg_H, normed, max_normed, N_sats,
                                             np.mean(counts_arr), len(counts_arr)]],
                                      columns=cols)
                    het_df = het_df.append(df)

    het_df.dropna(inplace=True) # NaN introduced if I average over an empty all_hets
    return het_df

ps = list(range(0, 96, 5)) + [99]
mus = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
max_nums = [10, 50, 100]
het_df = compile_hets(ps, mus, max_nums, req_full_enough=False)
comments = ['# "normed" heterozygosity normalized by value at which full satellite has one '\
            + 'individual with a different allele than everyone else: 2(1/max_num)(1-1/max_num)',
            '# "max_normed" normalized against maximum possible H in full satellite: 1 - 1/max_num']
with open('het_df.csv', 'w') as f:
    for comm in comments:
        f.write(comm + '\n')
    het_df.to_csv(f, index=False)
