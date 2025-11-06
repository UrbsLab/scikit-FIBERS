import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paretoset import paretoset
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from scipy.stats import linregress
import seaborn as sns
import matplotlib.patches as mpatches
import collections
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from .bin import BIN


def transform_value(n,cycle_length):
    remainder = n % (2 * cycle_length)
    if remainder > cycle_length:
        return 2 * cycle_length - remainder
    return remainder


def plot_pareto(bin_pop, show=True, save=False, output_folder=None, data_name=None):

    pareto_pre_fitness = []
    pareto_bin_size = []
    group_strata_prop = []
    group_threshold = []
    shapes = []
    sizes = []

    for bin in bin_pop:
        pareto_pre_fitness.append(bin.pre_fitness)
        pareto_bin_size.append(bin.bin_size)
        group_strata_prop.append(bin.group_strata_prop)

        if len(bin.group_threshold_list) == 1:  
            avg_threshold = bin.group_threshold_list[0]
            shapes.append('o')  
        else:  
            avg_threshold = sum(bin.group_threshold_list) / len(bin.group_threshold_list)
            shapes.append('s')  

        sizes.append((avg_threshold + 1) * 5)  
        group_threshold.append(avg_threshold)

    pareto_df = pd.DataFrame({
        'Pre-Fitness': pareto_pre_fitness,
        'Bin Size': pareto_bin_size,
        'Shape': shapes,
        'Size': sizes,
        'Group Strata Prop': group_strata_prop
    })

    mask = paretoset(pareto_df[['Pre-Fitness', 'Bin Size']], sense=["max", "min"])
    paretoset_fibers = pareto_df[mask]

    plt.figure(figsize=(6, 6))

    for shape in pareto_df['Shape'].unique():
        df_shape = pareto_df[pareto_df['Shape'] == shape]
        plt.scatter(df_shape['Pre-Fitness'], df_shape['Bin Size'], label=f'All Bins ({shape})', 
                    alpha=0.8, c=df_shape['Group Strata Prop'], cmap='viridis', 
                    s=df_shape['Size'], marker=shape)

    for shape in paretoset_fibers['Shape'].unique():
        df_shape = paretoset_fibers[paretoset_fibers['Shape'] == shape]
        plt.scatter(df_shape['Pre-Fitness'], df_shape['Bin Size'], label=f'Non-Dominated ({shape})',
                    s=df_shape['Size'], marker=shape, edgecolor='orange', linewidth=1.5, facecolor='none')

    plt.xlabel("Pre-Fitness")
    plt.ylabel("Bin Size")
    plt.colorbar(label='Group Strata Prop.')
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.tight_layout()

    if save:
        plt.savefig(output_folder + '/' + data_name + '_pop_pareto.png', bbox_inches="tight")
    if show:
        plt.show()

def plot_feature_tracking(feature_names,feature_tracking,max_features=40,show=True,save=False,output_folder=None,data_name=None): 

    sorted_pairs = sorted(zip(feature_tracking, feature_names), reverse=True)

    if max_features < len(feature_names):
        sorted_pairs = sorted_pairs[:max_features]

    top_scores, top_names = zip(*sorted_pairs)

    plt.figure(figsize=(16, 7))
    plt.bar(top_names, top_scores, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('Feature Tracking Score')
    plt.xticks(rotation=90)
    if save:
        plt.savefig(output_folder+'/'+data_name+'_feature_tracking.png', bbox_inches="tight")
    if show:
        plt.show()


def plot_kaplan_meir(low_outcome,low_censor,mid_outcome, mid_censor,high_outcome, high_censor,show=True,save=False,output_folder=None,data_name=None):
    kmf1 = KaplanMeierFitter()

    if mid_outcome is not None: # bin has 3 groups
        # fit the model for 1st cohort
        kmf1.fit(low_outcome, low_censor, label='At/Below Bin Low Threshold')
        a1 = kmf1.plot_survival_function()

        # fit the model for 2nd cohort
        kmf1.fit(mid_outcome, mid_censor, label = 'Between Bin Thresholds')
        kmf1.plot_survival_function()

        # fit the model for 3rd cohort
        kmf1.fit(high_outcome, high_censor, label='Above Bin High Threshold')
        kmf1.plot_survival_function(ax=a1)
    else: # bin has 2 groups
        # fit the model for 1st cohort
        kmf1.fit(low_outcome, low_censor, label='At/Below Bin Threshold')
        a1 = kmf1.plot_survival_function()

        # fit the model for 2nd cohort
        kmf1.fit(high_outcome, high_censor, label='Above Bin Threshold')
        kmf1.plot_survival_function(ax=a1)
    
    a1.set_ylabel('Survival Probability')
    a1.set_xlabel('Time After Event')

    if save:
        plt.savefig(output_folder+'/'+data_name+'_km.png', bbox_inches="tight")
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
        plt.savefig(output_folder+'/'+data_name+'_fitness_track.png', bbox_inches="tight")
    if show:
        plt.show()

def plot_threshold_progress(perform_track_df, show=True, save=False, output_folder=None, data_name=None):
    """
    Plot the thresholds progress over time, handling both single and multiple thresholds.
    When threshold values are None, they are replaced with -1 to ensure they appear on the graph.
    """
    # Extract columns for plotting
    time = perform_track_df['Iteration'].tolist()
    thresholds = perform_track_df['Threshold(s)']
    

    df_l = []
    df_h = []
    
    # use -1 to indicate no threshold
    for th in thresholds:
        if th is None or (isinstance(th, list) and len(th) == 0):
            df_l.append(-1)
            df_h.append(-1)
            continue
            
        if isinstance(th, list):
            if len(th) > 0:
                df_l.append(th[0] if th[0] is not None else -1)
                if len(th) > 1:
                    df_h.append(th[1] if th[1] is not None else -1)
                else:
                    df_h.append(-1)
            else:
                df_l.append(-1)
                df_h.append(-1)
        else:
            df_l.append(th if th is not None else -1)
            df_h.append(-1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(time, df_l, label='Lower Threshold', color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot higher threshold where it's not -1 (single threshold indicators)
    valid_h_indices = [i for i, h in enumerate(df_h) if h != -1]
    if valid_h_indices:
        valid_time = [time[i] for i in valid_h_indices]
        valid_h = [df_h[i] for i in valid_h_indices]
        plt.plot(valid_time, valid_h, label='Upper Threshold', color='red', linewidth=2, marker='o', markersize=4)
    
    # Shade regions for periods with just one threshold
    single_threshold_periods = []
    start_idx = None
    
    for i in range(len(df_h)):
        if df_h[i] == -1:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            single_threshold_periods.append((time[start_idx], time[i-1]))
            start_idx = None
    
    # Add the last period if it ends with a single threshold
    if start_idx is not None and len(time) > 0:
        single_threshold_periods.append((time[start_idx], time[len(time)-1]))
    
    # Shade single threshold periods
    for start, end in single_threshold_periods:
        plt.axvspan(start, end, alpha=0.2, color='darkgray')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Threshold Values', fontsize=12)
    plt.title('Threshold Progress Over Time', fontsize=14)
    
    legend_elements = [
        plt.Line2D([0], [0], color='blue', marker='o', markersize=4, linewidth=2, label='Lower Threshold'),
        plt.Line2D([0], [0], color='red', marker='o', markersize=4, linewidth=2, label='Upper Threshold'),
        Patch(facecolor='darkgray', alpha=0.2, label='Single Threshold Period')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.ylim(bottom=-1.5)
    
    plt.tight_layout()
    
    if save and output_folder and data_name:
        plt.savefig(f"{output_folder}/{data_name}_threshold_track.png", bbox_inches="tight", dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_perform_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df = perform_track_df[['Pre-Fitness']]

    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['blue']   # Manually set colors
    for i, column in enumerate(df.columns):
        plt.plot(time, df[column], label=column, color=colors[i])

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Pre-Fitness (Top Bin)')

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+data_name+'_pre-fitness_track.png', bbox_inches="tight")
    if show:
        plt.show()

def plot_misc_progress(perform_track_df, show=True, save=False, output_folder=None, data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    
    df = perform_track_df[['Birth Iteration', 'Bin Size', 'Group Ratio']]
    
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = {
        'Birth Iteration': 'darkred',
        'Bin Size': 'navy',
        'Group Ratio': 'forestgreen'
    }
    
    # Plot 1: Normalized values for comparison (all three metrics)
    for column in df_normalized.columns:
        ax1.plot(time, df_normalized[column], label=column, color=colors[column], 
                linewidth=2, marker='.', markersize=3)
    
    ax1.set_ylabel('Normalized Values (0-1)', fontsize=10)
    ax1.set_title('Normalized Metrics Comparison', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best')
    
    # Plot 2: Original values
    axes = [ax2, ax2.twinx(), ax2.twinx()]
    
    if len(df.columns) > 2:
        axes[2].spines['right'].set_position(('outward', 60))
    
    for i, column in enumerate(df.columns):
        line = axes[i].plot(time, df[column], label=column, color=colors[column], 
                           linewidth=2, marker='.', markersize=3)
        axes[i].set_ylabel(column, color=colors[column], fontsize=10)
        axes[i].tick_params(axis='y', colors=colors[column])
    
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_title('Actual Metric Values', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    lines = []
    labels = []
    for ax, column in zip(axes, df.columns):
        line, = ax.get_lines()
        lines.append(line)
        labels.append(column)
    
    ax2.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    descriptions = {
        'Birth Iteration': 'The iteration when the top bin was created',
        'Bin Size': 'Number of features in the top bin',
        'Group Ratio': 'Proportion of instances in the smaller group'
    }
    
    fig.text(0.1, 0.01, '\n'.join([f"{k}: {v}" for k, v in descriptions.items()]), 
             fontsize=9, ha='left', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save and output_folder and data_name:
        plt.savefig(f"{output_folder}/{data_name}_misc_track.png", bbox_inches="tight", dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_residuals_histogram(residuals,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        # Create a histogram
        plt.hist(residuals['deviance'], bins=50, color='skyblue', edgecolor='black')

        # Add labels and title
        plt.xlabel('Residual Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Cox PH Model Residuals')
        if save:
            plt.savefig(output_folder+'/'+data_name+'_residuals_histogram.png', bbox_inches="tight")
        if show:
            plt.show()
    else:
        print('Error: No residuals available to plot')


def plot_log_rank_residuals(residuals,bin_pop,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        log_rank_list = []
        residuals_score_list = []
        group_strata_prop = []
        group_threshold = []
        for bin in bin_pop:
            log_rank_list.append(bin.log_rank_score)
            residuals_score_list.append(bin.residuals_score)
            group_strata_prop.append(bin.group_strata_prop)
            group_threshold.append(bin.group_threshold)
        group_threshold = [(x+1)*5 for x in group_threshold]

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_rank_list, residuals_score_list)

        # Create scatter plot with trend line
        plt.scatter(log_rank_list, residuals_score_list, c=group_strata_prop, cmap='viridis', label='Bin',s=group_threshold)
        plt.plot(log_rank_list, slope*np.array(log_rank_list) + intercept, color='red', label='Trend Line')
        plt.xlabel('Log-Rank Score')
        plt.ylabel('Residuals Score')
        plt.title('Bin Population: Log-Rank Score vs. Residuals Score')
        plt.colorbar(label='Group Strata Prop.')  # Add colorbar to show the intensity scale
        plt.legend()

        # Add correlation coefficient to the plot
        plt.text(0.53, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
        if save:
            plt.savefig(output_folder+'/'+data_name+'_log_rank_residuals.png', bbox_inches="tight")
        if show:
            plt.show()
        # Calculate and print correlation
    else:
        print('Error: No residuals available to plot')


def plot_adj_HR_residuals(residuals,bin_pop,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        residuals_score_list = []
        adj_HR_list = []
        group_strata_prop = []
        group_threshold = []
        for bin in bin_pop:
            residuals_score_list.append(bin.residuals_score)
            adj_HR_list.append(bin.adj_HR)
            group_strata_prop.append(bin.group_strata_prop)
            group_threshold.append(bin.group_threshold)
        group_threshold = [(x+1)*5 for x in group_threshold]

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(adj_HR_list,residuals_score_list)

        # Create scatter plot with trend line
        plt.scatter(adj_HR_list, residuals_score_list, c=group_strata_prop, cmap='viridis', label='Bin',s=group_threshold)
        plt.plot(adj_HR_list, slope*np.array(adj_HR_list) + intercept, color='red', label='Trend Line')
        plt.xlabel('Adjusted HR')
        plt.ylabel('Residuals Score')
        plt.title('Bin Population: Adjusted HR vs. Residuals Score')
        plt.colorbar(label='Group Strata Prop.')  # Add colorbar to show the intensity scale
        plt.legend()

        # Add correlation coefficient to the plot
        plt.text(0.53, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
        if save:
            plt.savefig(output_folder+'/'+data_name+'_adj_hr_residuals.png', bbox_inches="tight")
        if show:
            plt.show()
    else:
        print('Error: No residuals available to plot')


def plot_log_rank_adj_HR(bin_pop,show=True,save=False,output_folder=None,data_name=None):
    log_rank_list = []
    adj_HR_list = []
    group_strata_prop = []
    group_threshold = []
    for bin in bin_pop:
        log_rank_list.append(bin.log_rank_score)
        adj_HR_list.append(bin.adj_HR)
        group_strata_prop.append(bin.group_strata_prop)
        group_threshold.append(bin.group_threshold)
    group_threshold = [(x+1)*5 for x in group_threshold]

    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_rank_list, adj_HR_list)

    # Create scatter plot with trend line
    plt.scatter(log_rank_list, adj_HR_list, c=group_strata_prop, cmap='viridis', label='Bin',s=group_threshold)
    plt.plot(log_rank_list, slope*np.array(log_rank_list) + intercept, color='red', label='Trend Line')
    plt.xlabel('Log-Rank Score')
    plt.ylabel('Adjusted HR')
    plt.title('Bin Population: Log-Rank Score vs. Adjusted HR')
    plt.colorbar(label='Group Strata Prop.')  # Add colorbar to show the intensity scale
    plt.legend()

    # Add correlation coefficient to the plot
    plt.text(0.53, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
    if save:
        plt.savefig(output_folder+'/'+data_name+'_log_rank_adj_hr.png', bbox_inches="tight")
    if show:
        plt.show()


def plot_adj_HR_metric_product(residuals,bin_pop,show=True,save=False,output_folder=None,data_name=None):
    if isinstance(residuals, pd.DataFrame):
        log_rank_residuals_list = []
        adj_HR_list = []
        group_strata_prop = []
        group_threshold = []
        for bin in bin_pop:
            log_rank_residuals_list.append(bin.log_rank_score*bin.residuals_score)
            adj_HR_list.append(bin.adj_HR)
            group_strata_prop.append(bin.group_strata_prop)
            group_threshold.append(bin.group_threshold)
        group_threshold = [(x+1)*5 for x in group_threshold]

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = linregress(log_rank_residuals_list, adj_HR_list)

        # Create scatter plot with trend line
        plt.scatter(log_rank_residuals_list, adj_HR_list, c=group_strata_prop, cmap='viridis', label='Bin',s=group_threshold)
        plt.plot(log_rank_residuals_list, slope*np.array(log_rank_residuals_list) + intercept, color='red', label='Trend Line')
        plt.xlabel('Log-Rank*Residuals Score')
        plt.ylabel('Adjusted HR')
        plt.title('Bin Population: Log-Rank*Residuals Score vs. Adjusted HR')
        plt.colorbar(label='Group Strata Prop.')  # Add colorbar to show the intensity scale
        plt.legend()

        # Add correlation coefficient to the plot
        plt.text(0.53, 0.02, f'Correlation coeff. = {r_value:.2f}', transform=plt.gca().transAxes)
        if save:
            plt.savefig(output_folder+'/'+data_name+'_metric_product_adj_hr.png', bbox_inches="tight")
        if show:
            plt.show()


def cox_prop_hazard(bin_df, outcome_label, censor_label,show_progress=False): #make bin variable beetween 0 and 1
    cph = CoxPHFitter()
    cph.fit(bin_df,outcome_label,event_col=censor_label, show_progress=show_progress)
    return cph.summary


def match_prefix(feature, locust_names):
    """
    :param feature: the feature
    :param locust_names: the list of locust names, must be exhaustive
    """
    for locust_label in locust_names:
        if feature.startswith(locust_label):
            return locust_label

    return "None"


def plot_bin_population_heatmap(population, feature_names,filtering=None,show=True,save=False,output_folder=None,data_name=None):
    """
    :param population: a list where each element is a list of specified features
    :param feature_list: an alphabetically sorted list containing each of the possible feature
    """
    fontsize = 20
    feature_count = len(feature_names)
    bin_names = []
    for i in range(len(population)):
        bin_names.append("Bin " + str(i + 1))

    feature_index_map = {}
    for i in range(feature_count):
        feature_index_map[feature_names[i]] = i #create feature to index mapping

    graph_df = []
    for bin in population:
        temp_arr = [0] * feature_count
        for feature in bin:
            temp_arr[feature_index_map[feature]] = 1
        graph_df.append(temp_arr)

    graph_df = pd.DataFrame(graph_df, bin_names, feature_names)

    if filtering != None:
        tdf = graph_df
        tdf = pd.DataFrame(tdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False)
        tdf = tdf[tdf['Count'] >= filtering]
        graph_df = graph_df[list(tdf.index)]
        feature_count = len(graph_df.columns)
        print(feature_count)

    num_bins = len(population) 
    max_bins = 100
    max_features = 100
    # iterate through df columns and adjust values as necessary
    if num_bins > max_bins:  #
        if feature_count > max_features: #over max bins and max features - fixed plot with no labels
            fig_size = (max_features // 2, max_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, xticklabels=False, yticklabels=False, vmax=1, vmin=0,
                        square=True, cmap="Blues", cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else: #Over max bins, but under max features
            fig_size = (feature_count// 2, max_bins  // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, yticklabels=False, vmax=1, vmin=0,
                        square=True, cmap="Blues", cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    else:
        if feature_count > max_features: #under max bins but over max features 
            fig_size = (max_features // 2, num_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, xticklabels=False, vmax=1, vmin=0, square=True, cmap="Blues",
                        cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            fig_size = (feature_count// 2 , num_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, vmax=1, vmin=0, square=True, cmap="Blues",
                        cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    legend_elements = [mpatches.Patch(color='aliceblue', label='Not in Bin'),
                        mpatches.Patch(color='darkblue', label='Included in Bin')]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontsize)
    plt.xlabel('Features',fontsize=fontsize)
    plt.ylabel('Bin Population',fontsize=fontsize)

    if save:
        plt.savefig(output_folder+'/'+data_name+'_basic_pop_heatmap.png', bbox_inches="tight")
    if show:
        plt.show()

    return graph_df


def match_prefix(feature, group_names):
    """
    :param feature: the feature
    :param group_names: the list of group names, must be exhaustive
    """
    for group_label in group_names:
        if feature.startswith(group_label):
            return group_label

    return "None"


def plot_custom_bin_population_heatmap(population,feature_names,group_names,legend_group_info,colors,max_bins,max_features,show=True,save=False,output_folder=None,data_name=None):
    """
    :param population: a list where each element is a list of specified features
    :param feature_list: an alphabetically sorted list containing each of the possible feature
    :param group_names: identifies unique text that identifies unique groups of features to group together in the heatmap separated by vertical lines
    :param legend_group_info: text for the different heatmap colors in the legend
    :param color_features: list of lists, where each sublists identifies all feature names in the data to be given a unique color in the heatmap other than default binary coloring
    :param colors: list of tuple objects identifying additional colors to use in the heatmap beyond the two default colors e.g. (0,0,1) for blue
    :param default_colors: list of tuple objects identifying the two default colors used in the heatmap for features unspecified and specified in bins e.g. (0,0,1) for blue
    :param max_bins: maximum number of bins in a population before the heatmap no longer prints these bin name lables on the y-axis
    :param max_features: maximum number of features in the dataset befor the heatmap no longer prints these feature name lables on the x-axis
    """
    fontsize = 20
    #Prepare bin population dataset
    feature_index_map = {}
    for i in range(len(feature_names)):
        feature_index_map[feature_names[i]] = i #create feature to featuer position index mapping

    graph_df = [] #create dataset of bin values
    for bin in population:
        temp_arr = [0] * len(feature_names)
        for feature in bin:
            temp_arr[feature_index_map[feature]] = 1
        graph_df.append(temp_arr)

    # Define bin names for plot
    bin_names = []
    for i in range(len(population)):
        bin_names.append("Bin " + str(i + 1))

    graph_df = pd.DataFrame(graph_df, bin_names, feature_names) #data, index, columns

    #Re order dataframe based on specified group names
    prefix_columns = {prefix: [col for col in graph_df.columns if col.startswith(prefix)] for prefix in group_names} # Get the columns starting with each prefix
    ordered_columns = sum(prefix_columns.values(), []) # Concatenate the columns lists in the desired order
    graph_df = graph_df[ordered_columns] # Reorder the DataFrame columns

    #Prepare for group lines in the figure
    group_size_counter =  group_size_counter = collections.defaultdict(int)

    group_list = [[] for _ in range(len(group_names))] #list of feature lists by group
    for feature in feature_names:
        p = match_prefix(feature, group_names)
        group_size_counter[p] += 1
        index = group_names.index(p)
        group_list[index].append(feature) 

    group_counter_sorted = []
    for name in group_names:
        group_counter_sorted.append((name,group_size_counter[name]))

    #Define color lists
    index_dict = {}
    count = 1
    for group in group_list:
        for feature in group:
            index_dict[feature] = count
        count += 1

    for feature in graph_df.columns: #for each feature
        if feature in index_dict:
            for i in range(len(graph_df[feature])):
                if graph_df[feature][i] == 1:
                    graph_df[feature][i] = index_dict[feature]
    num_bins = len(population) #tmp

    #Identify if one group is not represented (to readjust colors used in colormap)
    code = 1 #starts with specified features
    remove_colors = []
    for group in group_names:
        count = (graph_df == code).sum().sum()
        if count == 0:
            remove_colors.append(colors[code])
        code += 1

    applied_colors = [x for x in colors if x not in remove_colors]

    #Redo dataframe encoding
    code = 1
    if applied_colors != colors: #redo value encoding
        for i in range(0,len(group_names)):
            count = (graph_df == code).sum().sum()
            if count == 0:
                graph_df = graph_df.applymap(lambda x: x - 1 if x > code else x)
            else:
                code +=1

    #Prepare color mapping
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', applied_colors, N=len(applied_colors))
    #custom_cmap = ListedColormap.from_list('custom_cmap', colors, N=256)

    # iterate through df columns and adjust values as necessary
    if num_bins > max_bins:  #
        if len(feature_names) > max_features: #over max bins and max features - fixed plot with no labels
            fig_size = (max_features // 2, max_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, xticklabels=False, yticklabels=False,
                        square=True, cmap=custom_cmap, cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else: #Over max bins, but under max features
            fig_size = (len(feature_names)// 2, max_bins  // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, yticklabels=False,
                        square=True, cmap=custom_cmap, cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    else:
        if len(feature_names) > max_features: #under max bins but over max features 
            fig_size = (max_features // 2, num_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, xticklabels=False, square=True, cmap=custom_cmap,
                        cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        else:
            fig_size = (len(feature_names)// 2, num_bins // 2)
            # Create a heatmap using Seaborn
            plt.subplots(figsize=fig_size)
            ax=sns.heatmap(graph_df, square=True, cmap=custom_cmap,
                        cbar_kws={"shrink": .75}, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    legend_elements = []
    index = 0
    for color in colors:
        legend_elements.append(mpatches.Patch(color=color,label=legend_group_info[index]))
        index += 1

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),fontsize=fontsize)

    running_count = 0
    for name, count in group_counter_sorted:
        running_count += count
        ax.vlines(running_count, colors="Black", *ax.get_ylim())

    plt.xlabel('Features',fontsize=fontsize)
    plt.ylabel('Bin Population',fontsize=fontsize)

    if save:
        plt.savefig(output_folder+'/'+data_name+'_custom_pop_heatmap.png', bbox_inches="tight")
    if show:
        plt.show()

def plot_optimal_bins_km_curves(data, fibers, save_path=None):
    """
    Create and display Kaplan-Meier curves for both optimal bins
    """
    # Create optimal bins
    optimal_bin_3group = BIN(fibers.set.pareto)
    optimal_bin_3group.feature_list = ["P_" + str(i+1) for i in range(10)]
    optimal_bin_3group.group_threshold_list = [1, 3]
    optimal_bin_3group.birth_iteration = 0
    optimal_bin_3group.bin_size = 10
    
    optimal_bin_2group = BIN(fibers.set.pareto)
    optimal_bin_2group.feature_list = ["P_" + str(i+1) for i in range(10)]
    optimal_bin_2group.group_threshold_list = [4]
    optimal_bin_2group.birth_iteration = 0
    optimal_bin_2group.bin_size = 10
    
    # Evaluate both bins
    for bin_obj, thresh_list, name in [(optimal_bin_3group, [1, 3], "3-group"), 
                                       (optimal_bin_2group, [4], "2-group")]:
        bin_obj.evaluate_fixed_bin(data.loc[:,fibers.feature_names], data.loc[:,fibers.outcome_label], data.loc[:,fibers.censor_label], 
                                   fibers.outcome_type, fibers.fitness_metric, fibers.log_rank_weighting, fibers.outcome_label, 
                                   fibers.censor_label, fibers.min_thresh, fibers.max_thresh, fibers.int_thresh, fibers.group_thresh_list, 
                                   False, fibers.multi_thresholding, fibers.iterations, 0, fibers.residuals, 
                                   data.loc[:, fibers.covariates if fibers.covariates else []], fibers.naive_survival_optimization, thresh_list)
        
        bin_obj.calculate_pre_fitness(fibers.group_strata_min,fibers.penalty,fibers.fitness_metric,fibers.feature_names,fibers.naive_survival_optimization)
        
        print(f"\n{name} bin evaluation:")
        print(f"  Thresholds: {bin_obj.group_threshold_list}")
        print(f"  Log-rank score: {bin_obj.log_rank_score:.2f}")
        print(f"  Group counts: {bin_obj.count_bt}, {getattr(bin_obj, 'count_mt', 0)}, {bin_obj.count_at}")
        print(f"  Group proportions: {bin_obj.group_prop_list}")
        if hasattr(bin_obj, 'pairwise_scores') and bin_obj.pairwise_scores:
            print(f"  Pairwise scores: {bin_obj.pairwise_scores}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 3-group bin
    plot_bin_km_curve(data, optimal_bin_3group, fibers, axes[0], "3-Group Bin (Thresholds: 1, 3)")
    
    # Plot 2-group bin  
    plot_bin_km_curve(data, optimal_bin_2group, fibers, axes[1], "2-Group Bin (Threshold: 4)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    print_comparison_summary(optimal_bin_3group, optimal_bin_2group)
    
    return optimal_bin_3group, optimal_bin_2group

def plot_bin_km_curve(data, bin_obj, fibers, ax, title):
    """
    Plot Kaplan-Meier curve for a single bin
    """
    feature_sums = data[bin_obj.feature_list].sum(axis=1)
    
    if len(bin_obj.group_threshold_list) == 2:
        # 3-group bin
        low_thresh, high_thresh = bin_obj.group_threshold_list
        
        low_mask = feature_sums <= low_thresh
        mid_mask = (feature_sums > low_thresh) & (feature_sums <= high_thresh)
        high_mask = feature_sums > high_thresh
        
        groups = [
            (low_mask, f'Low Risk (≤{low_thresh})', 'blue'),
            (mid_mask, f'Medium Risk ({low_thresh+1}-{high_thresh})', 'orange'),
            (high_mask, f'High Risk (>{high_thresh})', 'red')
        ]
    else:
        # 2-group bin
        thresh = bin_obj.group_threshold_list[0]
        
        low_mask = feature_sums <= thresh
        high_mask = feature_sums > thresh
        
        groups = [
            (low_mask, f'Low Risk (≤{thresh})', 'blue'),
            (high_mask, f'High Risk (>{thresh})', 'red')
        ]
    
    # Plot survival curves
    kmf = KaplanMeierFitter()
    
    for mask, label, color in groups:
        if mask.sum() > 0:
            durations = data.loc[mask, fibers.outcome_label]
            events = data.loc[mask, fibers.censor_label]
            
            kmf.fit(durations, events, label=label)
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time After Event', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    y_pos = 0.15
    for mask, label, color in groups:
        count = mask.sum()
        ax.text(0.02, y_pos, f'{label}: n={count}', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        y_pos -= 0.05

def print_comparison_summary(bin_3group, bin_2group):
    """
    Print comparison between the two bins
    """
    print("Comparison Summary:")
    
    print(f"\n3-Group Bin Performance:")
    print(f"  Features: All 10 predictive features (P_1 to P_10)")
    print(f"  Thresholds: {bin_3group.group_threshold_list}")
    print(f"  Log-rank score: {bin_3group.log_rank_score:.2f}")
    print(f"  Group sizes: {bin_3group.count_bt} / {getattr(bin_3group, 'count_mt', 0)} / {bin_3group.count_at}")
    print(f"  Group proportions: {[f'{p:.3f}' for p in bin_3group.group_prop_list]}")
    if hasattr(bin_3group, 'pairwise_scores') and bin_3group.pairwise_scores:
        print(f"  Pairwise scores: {[f'{s:.1f}' for s in bin_3group.pairwise_scores]}")
        avg_pairwise = sum(bin_3group.pairwise_scores) / len(bin_3group.pairwise_scores)
        print(f"  Average pairwise: {avg_pairwise:.2f}")
    
    print(f"\n2-Group Bin Performance:")
    print(f"  Features: All 10 predictive features (P_1 to P_10)")
    print(f"  Threshold: {bin_2group.group_threshold_list}")
    print(f"  Log-rank score: {bin_2group.log_rank_score:.2f}")
    print(f"  Group sizes: {bin_2group.count_bt} / {bin_2group.count_at}")
    print(f"  Group proportions: {[f'{p:.3f}' for p in bin_2group.group_prop_list if p > 0]}")

def run_km_comparison(data, fibers, output_folder, data_name):
    """
    Main function to run the comparison
    """
    save_path = f"{output_folder}/{data_name}_optimal_bins_km_comparison.png"
    
    bin_3group, bin_2group = plot_optimal_bins_km_curves(data, fibers, save_path)
    
    return bin_3group, bin_2group

def find_optimal_bins_in_population(fibers):
    """
    Check if the optimal bins still exist in the population
    """
    optimal_features = set([f"P_{i+1}" for i in range(10)])
    
    found_bins = []
    
    for i, bin_obj in enumerate(fibers.set.bin_pop):
        bin_features = set(bin_obj.feature_list)
        
        # Check if this bin has all predictive features
        if optimal_features.issubset(bin_features):
            extra_features = bin_features - optimal_features
            found_bins.append({
                'index': i,
                'thresholds': bin_obj.group_threshold_list,
                'log_rank': bin_obj.log_rank_score,
                'fitness': bin_obj.fitness,
                'extra_features': list(extra_features),
                'bin_size': bin_obj.bin_size
            })
    
    return found_bins

def extend_curve_to_zero(times, probs, max_time):
    """
    Extend survival curve to maintain 0 probability until max_time
    """
    # Find where curve reaches 0 or very close to 0
    zero_threshold = 1e-10
    zero_indices = np.where(probs <= zero_threshold)[0]
    
    if len(zero_indices) > 0:
        # Curve reaches zero
        first_zero_idx = zero_indices[0]
        zero_time = times[first_zero_idx]
        
        # If curve reaches zero before max_time, extend with zeroes
        if zero_time < max_time:
            extended_times = np.concatenate([
                times[:first_zero_idx + 1],
                np.array([max_time])
            ])
            extended_probs = np.concatenate([
                probs[:first_zero_idx + 1],
                np.array([0.0])
            ])
        else:
            # Curve doesn't reach zero before max_time
            extended_times = times
            extended_probs = probs
    else:
        # Curve never reaches zero, extend to max_time with last probability
        if times[-1] < max_time:
            extended_times = np.concatenate([times, np.array([max_time])])
            extended_probs = np.concatenate([probs, np.array([probs[-1]])])
        else:
            extended_times = times
            extended_probs = probs
    
    return extended_times, extended_probs

def calculate_area_between_curves(data, bin_obj, fibers):
    """
    Calculate area between survival curves with proper zero handling
    """
    feature_sums = data[bin_obj.feature_list].sum(axis=1)
    
    if len(bin_obj.group_threshold_list) == 2:
        # 3-group bin
        low_thresh, high_thresh = bin_obj.group_threshold_list
        
        low_mask = feature_sums <= low_thresh
        mid_mask = (feature_sums > low_thresh) & (feature_sums <= high_thresh)
        high_mask = feature_sums > high_thresh
        
        groups = [
            (low_mask, 'Low', 'blue'),
            (mid_mask, 'Medium', 'orange'),
            (high_mask, 'High', 'red')
        ]
    else:
        # 2-group bin
        thresh = bin_obj.group_threshold_list[0]
        
        low_mask = feature_sums <= thresh
        high_mask = feature_sums > thresh
        
        groups = [
            (low_mask, 'Low', 'blue'),
            (high_mask, 'High', 'red')
        ]
    
    # Fit KM curves for each group
    kmf = KaplanMeierFitter()
    survival_functions = {}
    max_time = 0
    
    for mask, label, color in groups:
        if mask.sum() > 0:
            durations = data.loc[mask, fibers.outcome_label]
            events = data.loc[mask, fibers.censor_label]
            
            kmf.fit(durations, events, label=label)
            
            times = kmf.survival_function_.index.values
            probs = kmf.survival_function_[label].values
            
            max_time = max(max_time, times[-1])
            
            survival_functions[label] = {
                'times': times,
                'probs': probs,
                'count': mask.sum()
            }
    
    for label in survival_functions:
        extended_times, extended_probs = extend_curve_to_zero(
            survival_functions[label]['times'],
            survival_functions[label]['probs'],
            max_time
        )
        survival_functions[label]['times'] = extended_times
        survival_functions[label]['probs'] = extended_probs
    
    # Calculate areas between curves
    areas = {}
    
    if len(survival_functions) == 2:
        # 2-group
        low_data = survival_functions['Low']
        high_data = survival_functions['High']
        
        area = calculate_area_between_two_curves(low_data, high_data)
        areas['Low_vs_High'] = area
        
    elif len(survival_functions) == 3:
        # 3-group
        low_data = survival_functions['Low']
        mid_data = survival_functions['Medium']
        high_data = survival_functions['High']
        
        areas['Low_vs_Medium'] = calculate_area_between_two_curves(low_data, mid_data)
        areas['Low_vs_High'] = calculate_area_between_two_curves(low_data, high_data)
        areas['Medium_vs_High'] = calculate_area_between_two_curves(mid_data, high_data)
    
    return areas, survival_functions

def calculate_area_between_two_curves(curve1_data, curve2_data):
    """
    Calculate area between two survival curves using trapezoidal rule with proper zero handling
    """
    all_times = np.unique(np.concatenate([curve1_data['times'], curve2_data['times']]))
    all_times = np.sort(all_times)
    
    curve1_interp = np.interp(all_times, curve1_data['times'], curve1_data['probs'])
    curve2_interp = np.interp(all_times, curve2_data['times'], curve2_data['probs'])
    
    diff = np.abs(curve1_interp - curve2_interp)
    area = np.trapz(diff, all_times)
    
    return area

def plot_curves_with_shaded_area(data, bin_obj, fibers, title_suffix=""):
    """
    Plot survival curves with properly colored shaded areas between them
    """
    areas, survival_functions = calculate_area_between_curves(data, bin_obj, fibers)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'Low': 'blue', 'Medium': 'orange', 'High': 'red'}
    
    for label, sf_data in survival_functions.items():
        ax.plot(sf_data['times'], sf_data['probs'], 
                color=colors[label], linewidth=3, label=f"{label} Risk (n={sf_data['count']})")
    
    # Shade area between curves with different colors
    if len(survival_functions) == 2:
        # 2-group
        low_data = survival_functions['Low']
        high_data = survival_functions['High']
        
        # Get common time grid for shading
        all_times = np.unique(np.concatenate([low_data['times'], high_data['times']]))
        all_times = np.sort(all_times)
        
        low_interp = np.interp(all_times, low_data['times'], low_data['probs'])
        high_interp = np.interp(all_times, high_data['times'], high_data['probs'])
        
        ax.fill_between(all_times, low_interp, high_interp, 
                       alpha=0.3, color='gray', label=f'Area = {areas["Low_vs_High"]:.3f}')
        
    elif len(survival_functions) == 3:
        # 3-group
        low_data = survival_functions['Low']
        mid_data = survival_functions['Medium']
        high_data = survival_functions['High']
        
        all_times = np.unique(np.concatenate([
            low_data['times'], mid_data['times'], high_data['times']
        ]))
        all_times = np.sort(all_times)
        
        low_interp = np.interp(all_times, low_data['times'], low_data['probs'])
        mid_interp = np.interp(all_times, mid_data['times'], mid_data['probs'])
        high_interp = np.interp(all_times, high_data['times'], high_data['probs'])
        
        # Shade Low vs Medium area
        ax.fill_between(all_times, low_interp, mid_interp, 
                       alpha=0.4, color='lightblue', 
                       label=f'Low vs Medium Area = {areas["Low_vs_Medium"]:.3f}')
        
        # Shade Medium vs High area
        ax.fill_between(all_times, mid_interp, high_interp, 
                       alpha=0.4, color='lightcoral',
                       label=f'Medium vs High Area = {areas["Medium_vs_High"]:.3f}')
        
        # Add Low vs High area as text
        textstr = f'Low vs High Area = {areas["Low_vs_High"]:.3f}'
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax.text(0.02, 0.85, textstr, transform=ax.transAxes, fontsize=11,
                bbox=props, fontweight='bold')
    
    ax.set_xlabel('Time After Event', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves with Area Between Curves{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return areas, fig

def calculate_area_under_curve(times, probs):
    """
    Calculate area under a single survival curve using trapezoidal rule
    """
    return np.trapz(probs, times)
