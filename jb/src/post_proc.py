import pandas as pd
from scipy.stats import binom
from scipy.optimize import curve_fit
from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


burnin = 1000
def get_wavelet_power_peak():
    # setup and execute wavelet transform
    def wavelet(M,s):
         return signal.morlet2(M, s, w=6)

    def pad_data(x):
        """
        Pad data to the next power of 2
        """
        nx = len(x) # number of samples
        nx2 = (2**np.ceil(np.log(nx)/np.log(2))).astype(int) # next power of 2
        x2 = np.zeros(nx2, dtype=x.dtype) # pad to next power of 2
        offset = (nx2-nx)//2 # offset
        x2[offset:(offset+nx)] = x # copy
        return x2

    def coi_mask(b, T, min_period, max_period):
        """
        Cone of influence mask
        """
        coi = np.tile((np.ptp(T)/2 - np.abs(T - np.mean(T))) / np.sqrt(2), (b.shape[0], 1))
        s = np.tile(np.linspace(min_period, max_period, b.shape[0]), (b.shape[1], 1)).T
        return s >= coi

    def get_cases(node_id=0):
        df = pd.read_csv('simulation_output.csv')

        df_filtered = df[df['Node'] == node_id]

        # Calculate the week number based on Timestep
        df_filtered['Week'] = df_filtered['Timestep'] // 7

        # Group by Week and sum the New Infections for each week
        weekly_new_infections = df_filtered.groupby('Week')['New Infections'].sum().reset_index()
        return weekly_new_infections["New Infections"].to_numpy()

    def log_transform(x, debug=1):
        """
        Log transform for case data
        """ 
        # add one and take log
        x = np.log(x+1)
        # set mean=0 and std=1
        m = np.mean(x)
        s = np.std(x)
        x = (x - m)/s
        return x

    MAX_PERIOD = 7*52 # in weeks
    widths = np.logspace(np.log10(1), np.log10(MAX_PERIOD), int(MAX_PERIOD))
    y = widths / 52

    cases = get_cases(0)
    log_cases = pad_data(log_transform(cases))
    cwt = signal.cwt(log_cases, wavelet, widths)  # (M x N)
    # Number of time steps in padded time series
    nt = len(cases)
    # trim matrix
    offset = (cwt.shape[1] - nt) // 2
    cwt = cwt[:, offset:offset + nt]
    cwt2 = np.real(cwt * np.conj(cwt))
    power_peak_period = y[np.argmax(cwt2.mean(axis=1))]
    return power_peak_period 

def analyze_ccs():
    # Load the CSV file
    cases_df = pd.read_csv('simulation_output.csv')

    cases_df = cases_df[cases_df["Timestep"] > burnin]

    # Set the parameters for the binomial distribution
    num_trials = cases_df['New Infections']  # Number of trials (events)
    prob_success = 0.5  # Probability of success (observation)

    cases_df['Observed Infections'] = binom.rvs(num_trials, prob_success)

    cases_df['Weeks'] = cases_df['Timestep']//7
   
    # Load the cities.csv file into a DataFrame
    cities_df = pd.read_csv('cities.csv')

    # Merge the cases_df with cities_df based on the 'node' column
    merged_df = pd.merge(cases_df, cities_df, left_on='Node', right_on='ID')

    # Filter the merged DataFrame based on the condition
    #filtered_df = merged_df[(merged_df['Timestep'] == 1954)]

    # Group by 'Node', 'Name', and 'Weeks', and sum the 'Observed Infections' for each group
    grouped_df = merged_df.groupby(['Node', 'Name', 'Weeks'])['Observed Infections'].sum().reset_index()

    # Sort the DataFrame by 'Weeks' and then 'Node'
    sorted_df = grouped_df.sort_values(by=['Weeks', 'Node'])

    # Group by 'ID' and calculate the fraction of time Observed Infections is 0
    #fraction_nonzero = filtered_df.groupby('ID')['Observed Infections'].apply(lambda x: (x == 0).mean()).reset_index()
    fraction_nonzero = sorted_df.groupby('Node').apply(lambda group: (group['Observed Infections'] == 0).mean()).reset_index()
    merged_df = fraction_nonzero.merge(sorted_df[['Node', 'Name']], on='Node')
    merged_df = merged_df.rename(columns={0: 'Fraction_NonZero_New_Infections'})

    #weekly_infections = filtered_df.groupby(['ID', 'Weeks'])['Observed Infections'].sum().reset_index()

    # Rename the column to 'Fraction_NonZero_New_Infections'
    #fraction_nonzero.columns = ['ID', 'Fraction_NonZero_New_Infections']
    #fraction_nonzero = weekly_infections.groupby('ID').apply(lambda group: (group['Observed Infections'] != 0).mean()).reset_index()

    # Load the pops.csv file into a DataFrame
    pops_df = pd.read_csv('pops.csv')
    pops_df = pops_df[pops_df['Timestep'] == 1954]
    pops_df['ID'] = pops_df.reset_index().index

    # Sort the DataFrame by 'Fraction_NonZero_New_Infections'
    sorted_df = pd.merge(merged_df, pops_df[['ID', 'Population']], left_on='Node', right_on='ID')
    #sorted_df = fraction_nonzero.sort_values(by='Fraction_NonZero_New_Infections')
    #sorted_df = pd.merge(sorted_df, pops_df, on='ID')
    #print( sorted_df )

    # Select the rows corresponding to the specified cities
    cities = ['London', 'Birmingham', 'Liverpool', 'Manchester', 'Leeds']
    city_rows = sorted_df.loc[sorted_df['Name'].isin(cities)]

    # Calculate the mean of 'Fraction_NonZero_New_Infections' for the selected cities
    mean_fraction = city_rows['Fraction_NonZero_New_Infections'].mean()

    def get_median( population, fraction ):
        x=np.log10(population)
        y=fraction
        #slope, intercept, _, _, _ = linregress(x, y)
        median_point = np.median(x), np.median(y)
        return median_point

    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    initial_guess = [ 1.10907949, -1.78066486, 4.56063481, -0.08648216]

    try:
        popt, pcov = curve_fit(sigmoid, np.log10(sorted_df["Population"]), sorted_df['Fraction_NonZero_New_Infections'], p0=initial_guess)
        sig_slope = popt[1]
    except Exception as ex:
        print( str( ex ) )
        sig_slope = -5

    median = get_median( pops_df["Population"], sorted_df['Fraction_NonZero_New_Infections'] )
    def save_and_plot():
        sorted_df.to_csv( "logpop_vs_fractionzero.csv" )
        x = np.log10(sorted_df['Population'])
        y=sorted_df['Fraction_NonZero_New_Infections']
        plt.scatter(x,y)
        plt.show()
    #save_and_plot()
    return mean_fraction, median[1], sig_slope


def analyze():
    # Read the CSV file into a DataFrame
    raw_df = pd.read_csv("simulation_output.csv")

    df = raw_df[raw_df["Timestep"] > burnin]

    # Measure some things based on simulation_output.csv
    # 1) Total new infections per year...

    # Calculate the total number of new infections
    total_new_infections = df["New Infections"].sum()

    # Calculate the number of years
    num_years = ( df["Timestep"].max()-burnin) / 365  # Assuming 365 timesteps per year

    # Calculate the average number of new infections per year
    average_new_infections_per_year = total_new_infections / num_years

    # Filter the DataFrame to include only rows where Node is 507 (London)
    df_london = df[df["Node"] == 0]

    # Calculate the total number of new infections in London
    total_new_infections_london = df_london["New Infections"].sum()

    # Calculate the average number of new infections in London per year
    average_new_infections_per_year_london = total_new_infections_london / num_years

    """
    ccs_bigcity_mean, ccs_median, sig_slope = analyze_ccs()
    """

    # Create a DataFrame with the metric and its value
    data = {
        "metric": [
            "mean_new_infs_per_year",
            "mean_new_infs_per_year_london",
            "mean_ccs_fraction_big_cities",
            "ccs_median_fraction",
            "sigmoid_slope",
            "max_wavelet_power_period"
            ],
        "value": [
            average_new_infections_per_year,
            average_new_infections_per_year_london,
            0,
            0,
            0,
            get_wavelet_power_peak()
        ]
    }
    report_df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    report_df.to_csv("metrics.csv", index=False)

