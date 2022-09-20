'''
Record the population heterozygosity over the course of expansions, along with other
parameters. Collect the data from all relevant expansions into a single
data frame. Consider using groupby operations to 'aggregate' data from multiple
expansions at the same set of parameters or seaborn to easily plot tidy data while
assigning aesthetic properties to data frame variables (such as kernel exponent).
'''
import pandas as pd
import glob

def collect_hets(capacities, ps, mus):
    '''
    Inputs: capacities- the list of local carrying capacities (AKA max_nums)
            ps- the list of percentages of local dispersal
            mus- the list of kernel exponents
    '''
    cols = ['percent_local', 'mu', 'generation', 'num_individuals', 'pop_heterozygosity']
    df = pd.DataFrame(columns=cols)

    for cap in capacities:
        if cap == 10: ps = [50] # be careful about order of capacities!
        for p in ps:
            for mu in mus:
                d = f'hets_{cap}_biallelic_hettraj/mu{mu}_{p}local/'
                for file in glob.glob(d+'pop_data*.txt'):
                    pop_data = pd.read_csv(file)
                    if (mu != 4) and (pop_data.num_individuals.iloc[-1] < 1e7): continue # didn't finish simulation
                    elif (mu == 4) and (pop_data.num_individuals.iloc[-1] < 5e6): continue

                    relevant = pop_data[['generation', 'num_individuals', 'pop_heterozygosity']].copy()
                    relevant['capacity'] = cap
                    relevant['percent_local'] = p
                    relevant['mu'] = mu
                    relevant = relevant[['capacity', 'percent_local', 'mu', 'generation',
                                         'num_individuals', 'pop_heterozygosity']]

                    df = df.append(relevant)

    df.reset_index(drop=True, inplace=True)
    return df

caps = [100, 10]
ps = [0, 50, 80]
mus = [1.0, 1.5, 2.0, 2.5, 4.0]
df = collect_hets(caps, ps, mus)
df.to_csv('het_trajectories.csv', index=False)
