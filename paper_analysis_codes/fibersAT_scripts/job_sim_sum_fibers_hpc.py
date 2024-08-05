import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import collections
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src.skfibersAT.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--d', dest='datapath', help='name of data path (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--o', dest='outputpath', help='', type=str, default = 'myOutputPath') #full path/filename
    parser.add_argument('--r', dest='random_seeds', help='random seeds in experiment', type=int, default='None')

    #parser.add_argument('--f', dest='figures_only', help='random seeds in experiment', type=str, default='False')

    options=parser.parse_args(argv[1:])

    datapath = options.datapath
    outputpath = options.outputpath
    random_seeds = options.random_seeds

    #Get algorithm name
    outputfolder  = outputpath.split('/')[-1]
    algorithm = outputfolder.split('_')[0]

    #Get experiment name
    data_file = datapath.split('/')[-1]
    data_name = data_file.rstrip('.csv')
    experiment = data_name.split('_')[0]
    ideal_count = int(data_name.split('_')[6])
    ideal_threshold = int(data_name.split('_')[8])
    target_folder = outputpath+'/'+data_name #target output subfolder

    #Make local summary output folder
    if not os.path.exists(target_folder+'/'+'summary'):
        os.mkdir(target_folder+'/'+'summary')  

    #Load/Process Dataset
    data = pd.read_csv(datapath)

    true_risk_group = data[['TrueRiskGroup']]
    data = data.drop('TrueRiskGroup', axis=1)

    #Define columns for replicate results summary:
    columns = ["Bin Features", "Threshold", "Fitness", "Pre-Fitness", "Log-Rank Score","Log-Rank p-value",
               "Bin Size", "Group Ratio", "Count At/Below Threshold", "Count Above Threshold", "Birth Iteration", 
               "Deletion Probability", "Cluster", "Residual", "Residual p-value", "Unadjusted HR", "Unadjusted HR CI",
               "Unadjusted HR p-value", "Adjusted HR", "Adjusted HR CI", "Adjusted HR p-value", "Number of P", 
               "Number of R", "Ideal Iteration", "Accuracy", "Runtime", "Dataset Filename"]
    df = pd.DataFrame(columns=columns)

    #Make intial lists to store metrics across replications
    accuracy = []
    num_P = []
    num_R = []
    ideal = 0
    ideal_iter = []
    ideal_thresh = 0
    threshold = []
    log_rank = []
    residuals = []
    unadj_HR = []
    adj_HR = []
    group_balance = []
    runtime = []
    tc = 0
    bin_size = []
    birth_iteration = []

    top_bin_pop = []
    feature_names = None

    #Create top bin summary across replicates
    for random_seed in range(0, random_seeds):  #for each replicate
        #Unpickle FIBERS Object
        with open(target_folder+'/'+data_name+'_'+str(random_seed)+'_fibers.pickle', 'rb') as f:
            fibers = pickle.load(f)
        #Get top bin object for current fibers population
        bin_index = 0 #top bin
        bin = fibers.set.bin_pop[bin_index]
        feature_names = fibers.feature_names
        top_bin_pop.append(bin)


        results_list = [bin.feature_list, bin.group_threshold, bin.fitness, bin.pre_fitness, bin.log_rank_score,
                        bin.log_rank_p_value, bin.bin_size, bin.group_strata_prop, bin.count_bt, bin.count_at, 
                        bin.birth_iteration, bin.deletion_prop, bin.cluster, bin.residuals_score, bin.residuals_p_value,
                        bin.HR, bin.HR_CI, bin.HR_p_value, bin.adj_HR, bin.adj_HR_CI, bin.adj_HR_p_value, 
                        str(bin.feature_list).count('P'), str(bin.feature_list).count('R'), 
                        ideal_iteration(ideal_count, bin.feature_list, bin.birth_iteration),
                        accuracy_score(fibers.predict(data,bin_number=bin_index),true_risk_group) if true_risk_group is not None else None,
                        fibers.elapsed_time, data_name] 
        df.loc[len(df)] = results_list

        #Update metric lists
        accuracy.append(accuracy_score(fibers.predict(data,bin_number=bin_index),true_risk_group) if true_risk_group is not None else None)
        num_P.append(str(bin.feature_list).count('P'))
        num_R.append(str(bin.feature_list).count('R'))
        if ideal_iteration(ideal_count, bin.feature_list, bin.birth_iteration) != None:
            ideal += 1
            ideal_iter.append(bin.birth_iteration)
        if bin.group_threshold == ideal_threshold:
            ideal_thresh += 1
        threshold.append(bin.group_threshold)
        if bin.log_rank_score != None:
            log_rank.append(bin.log_rank_score)
        if bin.residuals_score != None:
            residuals.append(bin.residuals_score)
        if bin.HR != None:
            unadj_HR.append(bin.HR)
        if bin.adj_HR != None:
            adj_HR.append(bin.adj_HR)
        group_balance.append(bin.group_strata_prop)
        runtime.append(fibers.elapsed_time)
        if str(bin.feature_list).count('T') > 0:
            tc += 1
        bin_size.append(bin.bin_size)
        birth_iteration.append(bin.birth_iteration)

        #Generate Figures:
        #Kaplan Meir Plot
        fibers.get_kaplan_meir(data,bin_index,save=True,show=False, output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

        #Bin Population Heatmap
        group_names=["P", "R"]
        legend_group_info = ['Not in Bin','Predictive Feature in Bin','Non-Predictive Feature in Bin'] #2 default colors first followed by additional color descriptions in legend
        colors = [(.95, .95, 1),(0, 0, 1),(0.1, 0.1, 0.1)] #very light blue, blue, ---Alternatively red (1, 0, 0)  orange (1, 0.5, 0)
        max_bins = 100
        max_features = 100

        fibers.get_custom_bin_population_heatmap_plot(group_names,legend_group_info,colors,max_bins,max_features,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

        # Feature Importance Estimates
        fibers.get_feature_tracking_plot(max_features=50,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

    #Save replicate results as csv

    df.to_csv(target_folder+'/'+'summary'+'/'+data_name+'_summary'+'.csv', index=False)

    #Generate experiment summary 'master list'
    master_columns = ["Algorithm","Experiment", "Dataset", 
                    "Accuracy", "Accuracy (SD)", 
                    "Number of P", "Number of P (SD)",
                    "Number of R", "Number of R (SD)", "Ideal Bin", 
                    "Iteration of Ideal Bin", "Iteration of Ideal Bin (SD)", "Ideal Threshold", 
                    "Threshold", "Threshold (SD)",
                    "Log-Rank Score", "Log-Rank Score (SD)", 
                    "Residual", "Residual (SD)", 
                    "Unadjusted HR", "Unadjusted HR (SD)", 
                    "Adjusted HR", "Adjusted HR (SD)", 
                    "Group Ratio", "Group Ratio (SD)",
                    "Runtime", "Runtime (SD)", "TC1 Present", 
                    "Bin Size", "Bin Size (SD)", 
                    "Birth Iteration", "Birth Iteration (SD)"]
    
    df_master = pd.DataFrame(columns=master_columns)
    master_results_list = [algorithm,experiment,data_name,
                        np.mean(accuracy),np.std(accuracy),
                        np.mean(num_P),np.std(num_P),
                        np.mean(num_R),np.std(num_R), ideal, 
                        np.mean(ideal_iter),np.std(ideal_iter), ideal_thresh,
                        np.mean(threshold),np.std(threshold), 
                        None if len(log_rank) == 0 else np.mean(log_rank), None if len(log_rank) == 0 else np.std(log_rank) ,
                        None if len(residuals) == 0 else np.mean(residuals), None if len(residuals) == 0 else np.std(residuals), 
                        None if len(unadj_HR) == 0 else np.mean(unadj_HR), None if len(unadj_HR) == 0 else np.std(unadj_HR),
                        None if len(adj_HR) == 0 else np.mean(adj_HR), None if len(adj_HR) == 0 else np.std(adj_HR), 
                        np.mean(group_balance),np.std(group_balance),
                        np.mean(runtime),np.std(runtime), tc,
                        np.mean(bin_size),np.std(bin_size), 
                        np.mean(birth_iteration),np.std(birth_iteration)]
    
    df_master.loc[len(df_master)] = master_results_list
    #Save master results as csv
    df_master.to_csv(target_folder+'/'+'summary'+'/'+data_name+'_master_summary'+'.csv', index=False)

    #Generate Top-bin Custom Heatmap across replicates
    group_names=["P", "R"]
    legend_group_info = ['Not in Bin','Predictive Feature in Bin','Non-Predictive Feature in Bin'] #2 default colors first followed by additional color descriptions in legend
    colors = [(.95, .95, 1),(0, 0, 1),(0.1, 0.1, 0.1)] #very light blue, blue, ---Alternatively red (1, 0, 0)  orange (1, 0.5, 0)
    max_bins = 100
    max_features = 100
    filtering = 1

    #Generate Top-bin Custom Heatmap (filtering out zeros) across replicates
    population = pd.DataFrame([vars(instance) for instance in top_bin_pop])
    population = population['feature_list']
    plot_custom_top_bin_population_heatmap(population, feature_names, group_names,legend_group_info,colors,max_bins,max_features,filtering=filtering,save=True,show=False,output_folder=target_folder+'/'+'summary',data_name=data_name)

    #Generate Top-bin Basic Heatmap (filtering out zeros) across replicates
    gdf = plot_bin_population_heatmap(population, feature_names, filtering=filtering, show=False,save=True,output_folder=target_folder+'/'+'summary',data_name=data_name)

    #Generate feature frequency barplot
    pd.DataFrame(gdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False).plot.bar(figsize=(12, 4),
                     ylabel='Count Across Top Bins', xlabel='Feature')
    plt.savefig(target_folder+'/'+'summary'+'/'+data_name+'_feature_frequency_barplot.png', bbox_inches="tight")



def ideal_iteration(ideal_count, feature_list, birth_iteration):
    if str(feature_list).count('P') == ideal_count and str(feature_list).count('R') == 0:
        return birth_iteration
    else:
        return None

def match_prefix(feature, group_names):
    """
    :param feature: the feature
    :param group_names: the list of group names, must be exhaustive
    """
    for group_label in group_names:
        if feature.startswith(group_label):
            return group_label

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
        bin_names.append("Seed " + str(i + 1))

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
    plt.ylabel('Top Bins',fontsize=fontsize)

    if save:
        plt.savefig(output_folder+'/'+data_name+'_top_bins_basic_pop_heatmap.png', bbox_inches="tight")
    if show:
        plt.show()

    return graph_df

def plot_custom_top_bin_population_heatmap(population,feature_names,group_names,legend_group_info,colors,max_bins,max_features,filtering=None,show=True,save=False,output_folder=None,data_name=None):
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
        bin_names.append("Seed " + str(i))

    graph_df = pd.DataFrame(graph_df, bin_names, feature_names) #data, index, columns

    if filtering != None:
        tdf = graph_df
        tdf = pd.DataFrame(tdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False)
        tdf = tdf[tdf['Count'] >= filtering]
        graph_df = graph_df[list(tdf.index)]
        feature_names = graph_df.columns.tolist()

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
    print(remove_colors)
    print(colors)
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
    #custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
    custom_cmap = ListedColormap(applied_colors, 'custom_cmap', N=len(applied_colors))

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
    plt.ylabel('Top Bins',fontsize=fontsize)

    if save:
        plt.savefig(output_folder+'/'+data_name+'_top_bins_custom_pop_heatmap.png', bbox_inches="tight")
    if show:
        plt.show()

if __name__=="__main__":
    sys.exit(main(sys.argv))
