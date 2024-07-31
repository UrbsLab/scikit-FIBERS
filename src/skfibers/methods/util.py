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

def transform_value(n,cycle_length):
    remainder = n % (2 * cycle_length)
    if remainder > cycle_length:
        return 2 * cycle_length - remainder
    return remainder


def plot_pareto(bin_pop,show=True,save=False,output_folder=None,data_name=None):
    # Initialize lists to store Pareto-optimal solutions
    pareto_pre_fitness = []
    pareto_bin_size = []
    group_strata_prop = []
    group_threshold = []

    for bin in bin_pop:
        pareto_pre_fitness.append(bin.pre_fitness)
        pareto_bin_size.append(bin.bin_size)
        group_strata_prop.append(bin.group_strata_prop)
        group_threshold.append(bin.group_threshold)
    group_threshold = [(x+1)*5 for x in group_threshold]
    pareto_df = pd.DataFrame({'Pre-Fitness': pareto_pre_fitness, 'Bin Size': pareto_bin_size})

    mask = paretoset(pareto_df,sense=["max","min"])
    paretoset_fibers = pareto_df[mask]

    plt.figure(figsize=(5, 5))
    plt.scatter(pareto_df["Pre-Fitness"], pareto_df["Bin Size"], zorder=10, label="All Bins", alpha=0.8,c=group_strata_prop, cmap='viridis', s=group_threshold)
    plt.legend()
    plt.xlabel("Pre-Fitness")
    plt.ylabel("Bin Size")
    plt.colorbar(label='Group Strata Prop.')  # Add colorbar to show the intensity scale
    plt.scatter(
        paretoset_fibers["Pre-Fitness"],
        paretoset_fibers["Bin Size"],
        zorder=5,
        c='orange',
        label="Non-Dominated",
        s=150,
        alpha=1,
    )
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.tight_layout()
    if save:
        plt.savefig(output_folder+'/'+data_name+'_pop_pareto.png', bbox_inches="tight")
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
        plt.savefig(output_folder+'/'+data_name+'_feature_tracking.png', bbox_inches="tight")
    if show:
        plt.show()


def plot_kaplan_meir(low_outcome,low_censor,mid_outcome, mid_censor,high_outcome, high_censor,show=True,save=False,output_folder=None,data_name=None):
    kmf1 = KaplanMeierFitter()

    # fit the model for 1st cohort
    kmf1.fit(low_outcome, low_censor, label='At/Below Bin Low Threshold')
    a1 = kmf1.plot_survival_function()
    a1.set_ylabel('Survival Probability')

    # fit the model for 2nd cohort
    kmf1.fit(mid_outcome, mid_censor, label = 'Between Bin Thresholds')
    kmf1.plot_survival_function()

    # fit the model for 3rd cohort
    kmf1.fit(high_outcome, high_censor, label='Above Bin High Threshold')
    kmf1.plot_survival_function(ax=a1)
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


def plot_threshold_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df_l = perform_track_df[['Low Threshold']]
    df_h = perform_track_df[['High Threshold']]

    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['blue','red']  # Manually set colors
    for i, column in enumerate(df_l.columns):
        plt.plot(time, df_l[column], label=column, color=colors[i])
    for i, column in enumerate(df_h.columns):
        plt.plot(time, df_h[column], label=column, color=colors[i+1])
    

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Thresholds (Top Bin)')
    plt.title('Threshold Progress Over Time')
    plt.legend()

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+data_name+'_threshold_track.png', bbox_inches="tight")
    if show:
        plt.show()

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


def plot_misc_progress(perform_track_df,show=True,save=False,output_folder=None,data_name=None):
    # Extract columns for plotting
    time = perform_track_df['Iteration']
    df = perform_track_df[['Birth Iteration','Bin Size','Group Ratio']]
    df = (df - df.min()) / (df.max() - df.min())
    # Plot the data
    plt.figure(figsize=(5, 3))
    colors = ['red', 'blue', 'green']   # Manually set colors
    for i, column in enumerate(df.columns):
        plt.plot(time, df[column], label=column, color=colors[i])

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Values (0-1) ')
    plt.legend()  # Show legend

    # Show the plot
    plt.grid(True)
    if save:
        plt.savefig(output_folder+'/'+data_name+'_misc_track.png', bbox_inches="tight")
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


def cox_prop_hazard(bin_df, outcome_label, censor_label): #make bin variable beetween 0 and 1
    cph = CoxPHFitter()
    cph.fit(bin_df,outcome_label,event_col=censor_label, show_progress=False)
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
