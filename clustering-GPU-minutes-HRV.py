import argparse
from datetime import timedelta
import sys
from pathlib import Path
import cudf
import hdbscan
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from itertools import product

def save_fig(pt_id, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH = Path() / "results-GPU-minutes-HRV" / (pt_id + "-all-files")
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()

def read_data(pt_id):
    pt_pain_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_pain_filtered.xlsx")
    pt_hr_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_heartrate_1min_20190101_20230611.csv")
    pt_hr_seconds_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_heartrate_seconds_20190101_20230611.csv")
    pt_steps_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_minuteStepsNarrow_20190101_20230611.csv")
    pt_calories_path = Path() / "data" / (pt_id + "-all-files") / (
                pt_id + "_minuteCaloriesNarrow_20190101_20230611.csv")
    pt_sleep_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_sleepStageLogInfo_20190101_20230611.csv")
    pt_wear_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_fitbitWearTimeViaHR_20190101_20230611.csv")
    pt_sync_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_syncEvents_20190101_20230611.csv")
    pt_clinic_path = Path() / "data" / "pt-s0-s1-clinic-visits.xlsx"

    pt_pain = cudf.from_pandas(pd.read_excel(pt_pain_path))
    pt_hr = cudf.read_csv(pt_hr_path)
    pt_hr_seconds = cudf.read_csv(pt_hr_seconds_path)
    pt_steps = cudf.read_csv(pt_steps_path)
    pt_calories = cudf.read_csv(pt_calories_path)
    pt_sleep = cudf.read_csv(pt_sleep_path)
    pt_wear = cudf.read_csv(pt_wear_path)
    pt_sync = cudf.read_csv(pt_sync_path)
    pt_clinic = pd.read_excel(pt_clinic_path)

    return pt_pain, pt_hr, pt_hr_seconds, pt_steps, pt_calories, pt_sleep, pt_wear, pt_sync, pt_clinic

def process_data(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction):
    pt_pain, pt_hr, pt_hr_seconds, pt_steps, pt_calories, pt_sleep, pt_wear, pt_sync, pt_clinic = read_data(pt_id)

    # Convert date strings to datetime objects
    pt_sync['DateTime'] = cudf.to_datetime(pt_sync['DateTime'], format='%m/%d/%Y %I:%M:%S %p')
    pt_sync['SyncDateUTC'] = cudf.to_datetime(pt_sync['SyncDateUTC'], format='%m/%d/%Y %I:%M:%S %p')

    # Calculate the time difference and extract the time zone
    pt_sync['TimeZone'] = (pt_sync['DateTime'] - pt_sync['SyncDateUTC'])

    # Convert the DataFrame to pandas for the time zone adjustment
    pt_sync_pd = pt_sync.to_pandas()

    # Convert time zone difference to hours
    pt_sync_pd['TimeZone'] = pt_sync_pd['TimeZone'].apply(lambda x: x.total_seconds() / 3600)

    # Find the most common TimeZone
    most_common_timezone = pt_sync_pd['TimeZone'].mode()[0]

    # Calculate the adjustment needed to get to -7.0
    adjustment = -7.0 - most_common_timezone

    # Let's process the pain data
    pt_pain_filtered = pt_pain[["time", "MPQtotal", "painVAS", "mayoNRS"]]
    pt_pain_filtered = pt_pain_filtered.dropna()

    # Convert "time" column to datetime
    pt_pain_filtered['time'] = cudf.to_datetime(pt_pain_filtered['time'])

    # Convert time column which is in PST to UTC using pandas
    pt_pain_filtered_pd = pt_pain_filtered.to_pandas()
    pt_pain_filtered_pd['time'] += pd.to_timedelta(adjustment, unit='h')

    # Convert back to cudf DataFrame
    pt_pain_filtered = cudf.from_pandas(pt_pain_filtered_pd)

    # Extract only the date component as a string
    pt_pain_filtered['day'] = pt_pain_filtered['time'].dt.strftime('%Y-%m-%d')

    pt_pain_filtered_original = pt_pain_filtered.copy().to_pandas()

    # Sort the DataFrame based on the 'time' column in ascending order
    pt_pain_filtered = pt_pain_filtered.sort_values('time')

    # Calculate the time difference between consecutive surveys
    pt_pain_filtered['time_diff'] = pt_pain_filtered['time'].diff()

    # Keep the surveys where the time difference is greater than or equal washout_period minutes (washout period)
    pt_pain_filtered = pt_pain_filtered[pt_pain_filtered['time_diff'] >= timedelta(minutes=washout_period)]

    # Let's process the activity data (heart rate, steps, calories)
    # and compute the activity_bin_before_pain_survey to activity_bin_after_pain_survey minute averages with respect to the pain survey time stamp

    # Convert the "Time" column to datetime type
    pt_hr['Time'] = cudf.to_datetime(pt_hr['Time'], format='%m/%d/%Y %I:%M:%S %p')
    pt_hr_seconds['Time'] = cudf.to_datetime(pt_hr_seconds['Time'], format='%m/%d/%Y %I:%M:%S %p')
    pt_steps['Time'] = cudf.to_datetime(pt_steps['ActivityMinute'], format='%m/%d/%Y %I:%M:%S %p')
    pt_calories['Time'] = cudf.to_datetime(pt_calories['ActivityMinute'], format='%m/%d/%Y %I:%M:%S %p')

    # Create new DataFrame to store results
    results = pt_pain_filtered.copy()

    # Create an empty list to store the maximum time differences
    max_time_diffs = []

    # Create empty lists to store excluded time differences and their corresponding probabilities
    excluded_time_diffs = []
    excluded_probabilities = []

    for df, value_col, new_col in [(pt_hr, 'Value', 'hr'),
                                   (pt_hr_seconds, 'Value', 'rmssd'),
                                   (pt_steps, 'Steps', 'steps'),
                                   (pt_calories, 'Calories', 'calories')]:

        # Create empty series for results
        results_df = cudf.Series([np.nan] * len(results), index=results.index, dtype='float64')

        # Convert the cuDF DataFrame to a Pandas DataFrame for iteration
        results_pd = results.to_pandas()

        # Loop over the results DataFrame
        for i, row in results_pd.iterrows():
            # Filter df where the Time is within the interval BOTH BEFORE AND AFTER the survey
            mask = (df['Time'] >= row['time'] - pd.Timedelta(minutes=activity_bin_before_pain_survey)) & (
                        df['Time'] <= row['time'] + pd.Timedelta(minutes=activity_bin_after_pain_survey))
            df_filtered = df.loc[mask]

            # If there are matching rows in df_filtered, calculate the mean
            if len(df_filtered) > 0:
                if new_col == 'rmssd':
                    df_filtered = df_filtered.to_pandas()

                    # Calculate the time differences between consecutive timestamps
                    df_filtered['TimeDiff'] = (df_filtered['Time'].diff() / np.timedelta64(1, 's'))

                    # Calculate the squared differences between consecutive heart rate values
                    df_filtered['ValueDiffSq'] = (df_filtered['Value'].diff() ** 2)

                    # Calculate RMSSD using NumPy
                    rmssd = np.sqrt(np.mean(df_filtered['ValueDiffSq'] / df_filtered['TimeDiff']))

                    # Find the maximum TimeDiff value
                    max_time_diff = df_filtered['TimeDiff'].max()

                    if max_time_diff <= 240:
                        results_df[i] = rmssd
                        max_time_diffs.append(max_time_diff)
                    else:
                        # Exclude the time difference and store it in the excluded lists
                        excluded_time_diffs.append(max_time_diff)
                        excluded_probabilities.append(len(excluded_time_diffs) / len(max_time_diffs))
                else:
                    results_df[i] = df_filtered[value_col].mean()

        # Assign the results to the appropriate column in results
        results[new_col] = results_df

    # Drop rows with null values in 'hr', 'steps', and 'calories' columns
    results = results.dropna(subset=['hr', 'rmssd', 'steps', 'calories'])

    results['hr'] = results['hr'].astype('float64')
    results['rmssd'] = results['rmssd'].astype('float64')
    results['steps'] = results['steps'].astype('float64')
    results['calories'] = results['calories'].astype('float64')

    # Plot the CDF of the maximum time differences
    plt.figure(figsize=(8, 6))
    sorted_time_diffs = sorted(max_time_diffs)
    cumulative_prob = np.arange(len(sorted_time_diffs)) / len(sorted_time_diffs)
    plt.plot(sorted_time_diffs, cumulative_prob, marker='o', label='Included')
    plt.scatter(excluded_time_diffs, excluded_probabilities, color='red', marker='x', label='Excluded')
    plt.xlabel('Max TimeDiff')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Max TimeDiff (Excluded: {})'.format(len(excluded_time_diffs)))
    plt.grid(True)
    plt.legend()
    save_fig(pt_id, (pt_id + "_cdf_hrv_plot"))

    pt_sleep = pt_sleep[pt_sleep['IsMainSleep'] == 1]
    pt_sleep = pt_sleep[['StartTime', 'Efficiency', 'MinutesAsleep']]

    # Convert the "StartTime" column to datetime type
    pt_sleep['StartTime'] = cudf.to_datetime(pt_sleep['StartTime'], format='%m/%d/%Y %I:%M:%S %p')

    # Convert times in StartTime to relative start times
    pt_sleep['RelativeStartTime'] = pt_sleep['StartTime'].dt.hour + pt_sleep['StartTime'].dt.minute / 60 + pt_sleep[
        'StartTime'].dt.second / 3600

    # Extract just the date from the StartTime field and store it in a separate column
    pt_sleep['Day'] = pt_sleep['StartTime'].dt.strftime('%Y-%m-%d')

    # Get rid of days with 0 percent wear time
    pt_wear = pt_wear[pt_wear['PercentageWearTime'] != 0]

    # Convert 'Day' column in pt_wear to datetime type
    pt_wear['Day'] = cudf.to_datetime(pt_wear['Day'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

    # Merge sleep and wear dataframes with SQL-like inner join
    pt_sleep_wear = cudf.merge(pt_sleep, pt_wear, on='Day', how='inner')

    # Shorten some column names to improve plotting aesthetics
    pt_sleep_wear.columns = pt_sleep_wear.columns.str.replace('Classic', 'C-')
    pt_sleep_wear.columns = pt_sleep_wear.columns.str.replace('Stages', 'S-')

    pt_sleep_daily = pt_sleep_wear[["StartTime", "Efficiency", "MinutesAsleep", "RelativeStartTime", "Day"]].copy()

    # Convert cuDF DataFrames to pandas DataFrames
    results = results.to_pandas()
    pt_sleep_daily = pt_sleep_daily.to_pandas()

    # Sort pt_sleep_daily DataFrame by StartTime
    pt_sleep_daily = pt_sleep_daily.sort_values('StartTime')

    # Perform merge_asof with sorted DataFrames
    results = pd.merge_asof(results, pt_sleep_daily, left_on='time', right_on='StartTime', direction=sleep_binning_direction,
                            tolerance=timedelta(days=1))

    # Convert 'day' column to datetime type
    results['day'] = pd.to_datetime(results['day'])

    # Drop rows with missing values
    results = results.dropna(how='any')

    # Convert date columns to datetime
    pt_clinic['Stage 0 Implant'] = pd.to_datetime(pt_clinic['Stage 0 Implant'])
    pt_clinic['Stage 1 Implant'] = pd.to_datetime(pt_clinic['Stage 1 Implant'])
    pt_clinic['Clinic Visit 1'] = pd.to_datetime(pt_clinic['Clinic Visit 1'])
    pt_clinic['Clinic Visit 2'] = pd.to_datetime(pt_clinic['Clinic Visit 2'])
    pt_clinic['Clinic Visit 3'] = pd.to_datetime(pt_clinic['Clinic Visit 3'])
    pt_clinic['Clinic Visit 4'] = pd.to_datetime(pt_clinic['Clinic Visit 4'])
    pt_clinic['Clinic Visit 5'] = pd.to_datetime(pt_clinic['Clinic Visit 5'])
    pt_clinic['Clinic Visit 6'] = pd.to_datetime(pt_clinic['Clinic Visit 6'])
    pt_clinic['Clinic Visit 7'] = pd.to_datetime(pt_clinic['Clinic Visit 7'])
    pt_clinic['Clinic Visit 8'] = pd.to_datetime(pt_clinic['Clinic Visit 8'])
    pt_clinic['Clinic Visit 9'] = pd.to_datetime(pt_clinic['Clinic Visit 9'])
    pt_clinic['Clinic Visit 10'] = pd.to_datetime(pt_clinic['Clinic Visit 10'])

    # Calculate s0_exclusion
    pt_clinic['s0_exclusion_start'] = pt_clinic['Stage 0 Implant']
    pt_clinic['s0_exclusion_end'] = pt_clinic['Stage 0 Implant'] + pd.DateOffset(days=14)

    # Calculate s1_exclusion
    pt_clinic['s1_exclusion_start'] = pt_clinic['Stage 1 Implant'] - pd.DateOffset(days=2)
    pt_clinic['s1_exclusion_end'] = pt_clinic['Stage 1 Implant'] + pd.DateOffset(days=5)

    # Calculate clinic_exclusion for each visit
    visit_columns = ['Clinic Visit 1', 'Clinic Visit 2', 'Clinic Visit 3', 'Clinic Visit 4',
                     'Clinic Visit 5', 'Clinic Visit 6', 'Clinic Visit 7', 'Clinic Visit 8',
                     'Clinic Visit 9', 'Clinic Visit 10']

    for col in visit_columns:
        pt_clinic[col + '_exclusion_start'] = pt_clinic[col]
        pt_clinic[col + '_exclusion_end'] = pt_clinic[col] + pd.DateOffset(days=3)

    # Reset the index
    results = results.reset_index(drop=True)

    # Convert 'day' column to datetime type
    results['day'] = pd.to_datetime(results['day'])

    # Filter the row in pt_clinic based on the patient ID
    pt_row = pt_clinic[pt_clinic['Patient ID'] == pt_id]

    # Extract the exclusion columns from pt_row
    exclusion_cols = pt_row.columns

    # Create a mask to identify the rows within the exclusion date ranges
    mask = pd.Series(True, index=results.index)
    for col in range(13, len(exclusion_cols) - 1, 2):
        exclusion_start = pd.to_datetime(pt_row[exclusion_cols[col]].iloc[0]).date()
        exclusion_end = pd.to_datetime(pt_row[exclusion_cols[col + 1]].iloc[0]).date()

        # Exclude the data within the exclusion date range
        mask &= ~((results['day'].dt.date >= exclusion_start) & (results['day'].dt.date <= exclusion_end))

    # Apply the mask to filter out the data
    results = results[mask]

    # Reset the index
    results = results.reset_index(drop=True)

    results_filtered = results[["MPQtotal", "painVAS", "mayoNRS",
                                "hr", "rmssd", "steps", "calories",
                                "Efficiency", "MinutesAsleep", "RelativeStartTime"]].copy()

    return results, results_filtered, pt_pain_filtered_original, adjustment


# code adapted from https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
# we will use kendall tau rank correlation as it more robust to outliers than spearman's rank correlation

# Define the corrdot and corrfunc functions
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'kendall')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5, ], xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

def corrfunc(x, y, **kws):
    r, p = stats.kendalltau(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=70)

# Define the plotcorr function
def plotcorr(data):
    sns.set(style='white', font_scale=1.6)
    g = sns.PairGrid(data, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.histplot, kde=True, color='black')
    g.map_upper(corrdot)
    g.map_upper(corrfunc)

def explore_processed_data(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction):
    results, results_filtered, pt_pain_filtered_original, adjustment = process_data(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction)

    # let's check if data is normally distributed
    for column in results_filtered.columns:
        data = results_filtered[column]
        statistic, p_value = stats.shapiro(data)

        print("Column:", column)
        print("Shapiro-Wilk Test:")
        print("Statistic:", statistic)
        print("p-value:", p_value)
        print("\n")

    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, column in enumerate(results_filtered.columns):
        data = results_filtered[column]
        ax = axes[i]

        # Generate Q-Q plot
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot for {}".format(column))

    plt.tight_layout()

    save_fig(pt_id, (pt_id + "_Q_Q_plot"))

    plotcorr(results_filtered)

    plt.tight_layout()

    save_fig(pt_id, (pt_id + "_corr_plot"))

    return results, results_filtered, pt_pain_filtered_original, adjustment

# Function to fit and score a clusterer for a given parameter combination
def fit_and_score_clusterer(params, results_filtered_clustered):
    # Unpack the params tuple into individual variables
    min_cluster_size, min_samples, cluster_selection_epsilon, metric, cluster_selection_method, alpha, allow_single_cluster, gen_min_span_tree = params

    # Create an instance of the clusterer with current parameter settings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                cluster_selection_epsilon=cluster_selection_epsilon,
                                metric=metric,
                                cluster_selection_method=cluster_selection_method,
                                alpha=alpha,
                                allow_single_cluster=allow_single_cluster,
                                gen_min_span_tree=gen_min_span_tree)

    # Fit the clusterer to the data
    clusterer.fit(results_filtered_clustered)

    # Calculate the validity score
    validity_score = clusterer.relative_validity_

    return validity_score, {'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': cluster_selection_epsilon,
                            'metric': metric,
                            'cluster_selection_method': cluster_selection_method,
                            'alpha': alpha,
                            'allow_single_cluster': allow_single_cluster,
                            'gen_min_span_tree': gen_min_span_tree}

def clustering(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction, n_jobs):
    results, results_filtered, pt_pain_filtered_original, adjustment = explore_processed_data(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction)

    results_filtered_clustered = results_filtered.copy()

    results_filtered_clustered = results_filtered_clustered[["MPQtotal", "rmssd", "MinutesAsleep"]]

    # Create an instance of StandardScaler
    scaler = StandardScaler()

    # Z-score normalization for each column
    results_filtered_clustered = pd.DataFrame(scaler.fit_transform(results_filtered_clustered),
                                              columns=results_filtered_clustered.columns)

    # Define the parameter grid
    param_grid = {'min_cluster_size': [i for i in range(50, 101, 1)],
                  'min_samples': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'cluster_selection_epsilon': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'metric': ['euclidean', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean'],
                  'cluster_selection_method': ['eom', 'leaf'],
                  'alpha': [1.0],
                  'allow_single_cluster': [False],
                  'gen_min_span_tree': [True]}

    best_score = -np.inf
    best_params = {}

    # Generate parameter combinations
    param_combinations = list(product(*param_grid.values()))

    # Perform grid search in parallel
    results_search = Parallel(n_jobs=n_jobs)(delayed(fit_and_score_clusterer)(params, results_filtered_clustered) for params in param_combinations)

    # Find the best parameters and validity score
    for validity_score, params in results_search:
        if validity_score > best_score:
            best_score = validity_score
            best_params = params

    # Fit the clusterer with the best parameters
    best_clusterer = hdbscan.HDBSCAN(**best_params)
    best_clusterer.fit(results_filtered_clustered)

    # Assign clusters to the data
    results_filtered_clustered["cluster"] = best_clusterer.labels_

    # Count the number of instances in each cluster
    print("Cluster counts: ")
    print(results_filtered_clustered["cluster"].value_counts())

    results_filtered_clustered = results_filtered.copy()

    results_filtered_clustered = results_filtered_clustered[["MPQtotal", "rmssd", "MinutesAsleep"]]

    # Let's Z-score each column across all the instances in our dataset to deal with outliers better
    results_filtered_clustered = pd.DataFrame(StandardScaler().fit_transform(results_filtered_clustered),
                                              columns=results_filtered_clustered.columns)

    # We will use a hierarchical clustering algorithm (hierarchical DBSCAN)
    clusterer = hdbscan.HDBSCAN(**best_params)

    cluster_labels = clusterer.fit_predict(results_filtered_clustered)

    results_filtered_clustered["cluster"] = cluster_labels.tolist()

    # Compute the validity index
    vi = hdbscan.validity_index(results_filtered_clustered.values, cluster_labels, metric=best_params['metric'])

    print("Relative Validity Index: ", clusterer.relative_validity_)
    print("Validity Index:", vi)
    print("Cluster Counts:")
    print(results_filtered_clustered["cluster"].value_counts())

    return results, results_filtered, results_filtered_clustered, pt_pain_filtered_original, adjustment

def visualize_clusters(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction, n_jobs):
    results, results_filtered, results_filtered_clustered, pt_pain_filtered_original, adjustment = clustering(pt_id, washout_period, activity_bin_before_pain_survey, activity_bin_after_pain_survey, sleep_binning_direction, n_jobs)

    # let's visualize the clustering results in a 3D scatter plot with more concrete axes
    results_filtered_clustered_scatter = results_filtered_clustered.iloc[:, [0, 1, 2]].to_numpy()

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Creating plot
    sc = ax.scatter3D(results_filtered_clustered_scatter[:, 0],
                      results_filtered_clustered_scatter[:, 1],
                      results_filtered_clustered_scatter[:, 2],
                      c=results_filtered_clustered["cluster"],
                      cmap='viridis')  # Use a colormap for distinct colors

    # Customizing legend
    legend_elements = []
    for cluster in results_filtered_clustered["cluster"].unique():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster}',
                                          markerfacecolor=sc.cmap(sc.norm(cluster)), markersize=8))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Clusters")

    ax.set_xlabel(results_filtered_clustered.columns[0], fontweight='bold')
    ax.set_ylabel(results_filtered_clustered.columns[1], fontweight='bold')
    ax.set_zlabel(results_filtered_clustered.columns[2], fontweight='bold')

    save_fig(pt_id, (pt_id + "_clusters_raw_plot"))

    # let's calculate cluster fingerprints to see
    # if the clusters actually represent high and low sleep based on objective metrics
    maxClusterIndex = results_filtered_clustered["cluster"].max()

    clusterFingerPrintsMeanZScore = pd.DataFrame()
    clusterFingerPrintsStdZScore = pd.DataFrame()

    for i in range(-1, maxClusterIndex + 1):
        rowsOfInterest = results_filtered_clustered[results_filtered_clustered["cluster"] == i]

        clusterFingerPrintsMeanZScore["C" + str(i)] = rowsOfInterest.loc[:, rowsOfInterest.columns != "cluster"].mean(
            axis=0)
        clusterFingerPrintsStdZScore["C" + str(i)] = rowsOfInterest.loc[:, rowsOfInterest.columns != "cluster"].std(
            axis=0)

    print("cluster finger prints mean z score: ")
    print(clusterFingerPrintsMeanZScore)

    plt.figure()
    sns.heatmap(clusterFingerPrintsMeanZScore, vmin=-3, vmax=3, cmap="RdBu_r").set(title="Mean Z Scores")
    save_fig(pt_id, (pt_id + "_cluster_fingerprints_mean_z_score"))

    print("cluster finger prints std z score: ")
    print(clusterFingerPrintsStdZScore)

    plt.figure()
    sns.heatmap(clusterFingerPrintsStdZScore, vmin=0, vmax=1.5, cmap="gray_r").set(title="Standard Deviation Z Scores")
    save_fig(pt_id, (pt_id + "_cluster_fingerprints_std_z_score"))

    results_filtered_clustered_plot = results_filtered.copy()
    results_filtered_clustered_plot["cluster"] = results_filtered_clustered["cluster"].copy()
    results_filtered_clustered_plot = results_filtered_clustered_plot[results_filtered_clustered["cluster"] != -1]

    # Define a custom color palette with distinct colors for each cluster
    cluster_palette = sns.color_palette("Set3", n_colors=results_filtered_clustered_plot["cluster"].nunique())

    # Set the style of the plots
    sns.set(style="ticks")

    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.5)

    # Iterate over each feature and plot the histogram colored by cluster
    for i, column in enumerate(results_filtered_clustered_plot.columns):
        if column != 'cluster':  # Skip the 'cluster' column
            row = i // 3  # Determine the row index
            col = i % 3  # Determine the column index

            ax = axes[row, col]  # Select the current subplot
            sns.histplot(data=results_filtered_clustered_plot, x=column, hue='cluster', ax=ax, palette=cluster_palette,
                         common_norm=False, stat="probability")
            ax.set_title(column)  # Set the title for the subplot
            ax.set_ylim([0, 1])  # Set the y-axis limits to [0, 1]

    # Move the legend outside the plot and place it at the bottom
    handles, labels = ax.get_legend_handles_labels()

    if labels:
        fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=len(labels))

    # Adjust the layout to accommodate the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_fig(pt_id, (pt_id + "_clusters_histograms_plot"))

    # Merge the two DataFrames based on the 'time' column
    results['Used'] = 1

    pt_pain_filtered_original = pd.merge(pt_pain_filtered_original, results[['time', 'Used']], on='time', how='left')

    pt_pain_filtered_original['Used'].fillna(0, inplace=True)

    pt_sleep_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_sleepStageLogInfo_20190101_20230611.csv")
    pt_wear_path = Path() / "data" / (pt_id + "-all-files") / (pt_id + "_fitbitWearTimeViaHR_20190101_20230611.csv")

    pt_sleep_original = pd.read_csv(pt_sleep_path)

    pt_wear_original = pd.read_csv(pt_wear_path)

    # Pain data processing
    pt_pain_filtered_original['day'] = pd.to_datetime(pt_pain_filtered_original['day'])
    pain_daily_avg = pt_pain_filtered_original.groupby('day')['Used'].mean()
    pain_moving_avg = pain_daily_avg.rolling(window=7).mean()

    # Get the first month with pain data available
    start_date = pt_pain_filtered_original['day'].min().replace(day=1)

    # Wear time data processing
    pt_wear_original['Day'] = pd.to_datetime(pt_wear_original['Day'])
    pt_wear_original.sort_values('Day', inplace=True)  # Sort the DataFrame by 'Day' column

    # Filter the wear time data based on the start date
    filtered_wear_time = pt_wear_original[pt_wear_original['Day'] >= start_date]
    wear_moving_avg = filtered_wear_time['PercentageWearTime'].rolling(window=7).mean()

    # Sleep data processing
    pt_sleep_original['StartTime'] = pd.to_datetime(pt_sleep_original['StartTime'])
    pt_sleep_original['Day'] = pt_sleep_original['StartTime'].dt.date
    sleep_records = pt_sleep_original.groupby('Day').apply(lambda x: 1 if x.shape[0] > 0 else 0)
    sleep_dates = sleep_records.index

    # Set the seaborn style
    sns.set(style='whitegrid')

    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)

    # Plotting the pain records
    ax1.plot(pain_daily_avg.index, pain_moving_avg, color='red')
    ax1.set_ylabel('7-Day Moving Average of Used Pain Surveys')
    ax1.set_title('Pain Records')

    # Plotting the wear time
    ax2.plot(filtered_wear_time['Day'], wear_moving_avg, color='blue')
    ax2.set_ylabel('7-Day Moving Average of Percentage Wear Time')
    ax2.set_title('Wear Time')

    # Plotting the sleep raster plot
    ax3.eventplot(sleep_dates, lineoffsets=0.5, linelengths=0.5, linewidths=1, color='gray')
    ax3.set_xlabel('Time (Day)')
    ax3.set_ylabel('Raster Plot')
    ax3.set_title('Sleep Records')

    # Set the locator and formatter for the x-axis to display labels for every month
    months_locator = mdates.MonthLocator()  # Locator for months
    months_formatter = mdates.DateFormatter('%b %Y')  # Formatter for month labels
    ax3.xaxis.set_major_locator(months_locator)
    ax3.xaxis.set_major_formatter(months_formatter)

    # Rotate the x-axis tick labels for all subplots
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90)

    # Adjust the layout
    plt.subplots_adjust(hspace=0.4, bottom=0.2)

    save_fig(pt_id, (pt_id + "_data_filtered_before_clustering_plot"))

    results_filtered_raw_clustered = results_filtered.copy()
    results_filtered_raw_clustered["cluster"] = results_filtered_clustered["cluster"].copy()

    for cluster_index, count in results_filtered_raw_clustered["cluster"].value_counts().items():
        # Loop over each unique cluster label and its frequency in the Series
        if cluster_index != -1:
            # Check if the cluster_index is not equal to -1 (i.e., skip cluster with label -1)
            cluster_data = results_filtered_raw_clustered[results_filtered_raw_clustered["cluster"] == cluster_index]
            plotcorr(cluster_data.loc[:, cluster_data.columns != "cluster"])
            plt.tight_layout()
            save_fig(pt_id, f"{pt_id}_cluster{cluster_index}_corr_plot")

    results["cluster"] = results_filtered_clustered["cluster"].copy()

    results['time'] -= pd.to_timedelta(adjustment, unit='h')

    results.to_excel(Path() / "results-GPU-minutes-HRV" / (pt_id + "-all-files") / (pt_id + "_results.xlsx"), index=False)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Adding optional arguments
    parser.add_argument("--pt_ids", nargs='+', help="Patient IDs")
    parser.add_argument("--washout_period", type=int, help="Washout Period")
    parser.add_argument("--activity_bin_before_pain_survey", type=int, help="Activity bin before pain survey")
    parser.add_argument("--activity_bin_after_pain_survey", type=int, help="Activity bin after pain survey")
    parser.add_argument("--sleep_binning_direction", type=str, help="Sleep binning direction")
    parser.add_argument("--n_jobs", type=int, help="Number of jobs")

    # Read arguments from command line
    args = parser.parse_args()

    # Check if sleep_binning_direction is either 'forward' or 'backward'
    if args.sleep_binning_direction not in ['forward', 'backward']:
        print(f"Error: Invalid value for sleep_binning_direction: {args.sleep_binning_direction}")
        sys.exit(1)

    # Print the arguments
    print(f"Patient IDs: {args.pt_ids}")
    print(f"Washout Period: {args.washout_period}")
    print(f"Activity bin before pain survey: {args.activity_bin_before_pain_survey}")
    print(f"Activity bin after pain survey: {args.activity_bin_after_pain_survey}")
    print(f"Sleep binning direction: {args.sleep_binning_direction}")
    print(f"Number of jobs: {args.n_jobs}")

    for pt_id in args.pt_ids:
        visualize_clusters(pt_id,
                           args.washout_period,
                           args.activity_bin_before_pain_survey,
                           args.activity_bin_after_pain_survey,
                           args.sleep_binning_direction,
                           args.n_jobs)

if __name__ == "__main__":
    main()