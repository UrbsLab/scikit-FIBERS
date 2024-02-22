import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paretoset import paretoset
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from scipy.stats import linregress


def plot_pareto(bin_pop,show=True,save=False,output_folder=None,data_name=None):
    # Initialize lists to store Pareto-optimal solutions
    pareto_pre_fitness = []
    pareto_bin_size = []

    for bin in bin_pop:
        pareto_pre_fitness.append(bin.pre_fitness)
        pareto_bin_size.append(bin.bin_size)
    pareto_df = pd.DataFrame({'Pre-Fitness': pareto_pre_fitness, 'Bin Size': pareto_bin_size})

    mask = paretoset(pareto_df,sense=["max","min"])
    paretoset_fibers = pareto_df[mask]

    plt.figure(figsize=(5, 5))

    plt.scatter(pareto_df["Pre-Fitness"], pareto_df["Bin Size"], zorder=10, label="All Bins", s=50, alpha=0.8)

    plt.scatter(
        paretoset_fibers["Pre-Fitness"],
        paretoset_fibers["Bin Size"],
        zorder=5,
        label="Non-Dominated",
        s=150,
        alpha=1,
    )
    plt.legend()
    plt.xlabel("Pre-Fitness")
    plt.ylabel("Bin Size")
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.tight_layout()
    if save:
        plt.savefig(output_folder+'/'+'Pop_Pareto_'+data_name+'.png')
    if show:
        plt.show()


def plot_feature_tracking(feature_names,feature_tracking,max_features=40,show=True,save=False,output_folder=None,data_name=None): 
    # Sort the names and scores based on scores
    sorted_pairs = sorted(zip(feature_tracking, feature_names), reverse=True)

    # Filter the top scoring features for visualization
    if max_features < len(feature_names):
        sorted_pairs = sorted_pairs[:max_features]

    # Unzip the top features
    top_scores, top_names = zip(*sorted_pairs)

    # Create a bar plot
    plt.figure(figsize=(16, 7))
    plt.bar(top_names, top_scores, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Feature Tracking Score')
    plt.xticks(rotation=90)
    if save:
        plt.savefig(output_folder+'/'+'Feature_Tracking_'+data_name+'.png')
    if show:
        plt.show()


def plot_kaplan_meir(low_outcome,low_censor,high_outcome, high_censor,show=True,save=False,output_folder=None,data_name=None):
    kmf1 = KaplanMeierFitter()

    # fit the model for 1st cohort
    kmf1.fit(low_outcome, low_censor, label='At/Below Bin Threshold')
    a1 = kmf1.plot_survival_function()
    a1.set_ylabel('Survival Probability')

    # fit the model for 2nd cohort
    kmf1.fit(high_outcome, high_censor, label='Above Bin Threshold')
    kmf1.plot_survival_function(ax=a1)
    a1.set_xlabel('Time After Event')
    if save:
        plt.savefig(output_folder+'/'+'KM_'+data_name+'.png')
    if show:
        plt.show()


def plot_fitness_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df = perform_track_df[['Fitness']]

    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['blue']  # Manually set colors
    for i, column in enumerate(df.columns):
        plt.plot(time, df[column], label=column, color=colors[i])

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Top Bin)')

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+'Perform_Track_'+data_name+'.png')
    if show:
        plt.show()


def plot_perform_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df = perform_track_df[['Pre-Fitness','Metric',]]

    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['red', 'blue']   # Manually set colors
    for i, column in enumerate(df.columns):
        plt.plot(time, df[column], label=column, color=colors[i])

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (Top Bin)')
    plt.legend()  # Show legend

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+'Perform_Track_'+data_name+'.png')
    if show:
        plt.show()


def plot_misc_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df = perform_track_df[['Birth Iteration','Bin Size','Group Ratio','Threshold']]
    df = (df - df.min()) / (df.max() - df.min())
    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['red', 'blue', 'green', 'orange']   # Manually set colors
    for i, column in enumerate(df.columns):
        plt.plot(time, df[column], label=column, color=colors[i])

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Values (0-1) ')
    plt.legend()  # Show legend

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+'Misc_Track_'+data_name+'.png')
    if show:
        plt.show()


def plot_residuals_histogram(residuals,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        # Create a histogram
        plt.hist(residuals['deviance'], bins=50, color='skyblue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Residual Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Cox PH Model Residuals')
        if save:
            plt.savefig(output_folder+'/'+'Residuals_Histogram_'+data_name+'.png')
        if show:
            plt.show()
    else:
        print('Error: No residuals available to plot')


def plot_log_rank_residuals(residuals,bin_pop,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        metric_list = []
        residuals_score_list = []
        for bin in bin_pop:
            metric_list.append(bin.metric)
            residuals_score_list.append(bin.residuals_score)

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(metric_list, residuals_score_list)

        # Create scatter plot with trend line
        plt.scatter(metric_list, residuals_score_list, label='Data')
        plt.plot(metric_list, slope*np.array(metric_list) + intercept, color='red', label='Trend Line')
        plt.xlabel('Log-Rank Score')
        plt.ylabel('Residuals Score')
        plt.title('Bin Population: Log-Rank Score vs. Residuals Score')
        plt.legend()

        # Add correlation coefficient to the plot
        plt.text(0.63, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
        if save:
            plt.savefig(output_folder+'/'+'Log_Rank_Residuals_'+data_name+'.png')
        if show:
            plt.show()
        # Calculate and print correlation
    else:
        print('Error: No residuals available to plot')

def plot_adj_HR_residuals(residuals,bin_pop,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        residuals_score_list = []
        adj_HR_list = []
        for bin in bin_pop:
            residuals_score_list.append(bin.residuals_score)
            adj_HR_list.append(bin.adj_HR)

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(adj_HR_list,residuals_score_list)

        # Create scatter plot with trend line
        plt.scatter(adj_HR_list, residuals_score_list, label='Data')
        plt.plot(adj_HR_list, slope*np.array(adj_HR_list) + intercept, color='red', label='Trend Line')
        plt.xlabel('Adjusted HR')
        plt.ylabel('Residuals Score')
        plt.title('Bin Population: Adjusted HR vs. Residuals Score')
        plt.legend()

        # Add correlation coefficient to the plot
        plt.text(0.63, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
        if save:
            plt.savefig(output_folder+'/'+'Adj_HR_Residuals_'+data_name+'.png')
        if show:
            plt.show()
    else:
        print('Error: No residuals available to plot')


def cox_prop_hazard(bin_df, outcome_label, censor_label): #make bin variable beetween 0 and 1
    cph = CoxPHFitter()
    cph.fit(bin_df,outcome_label,event_col=censor_label, show_progress=True)
    return cph.summary
