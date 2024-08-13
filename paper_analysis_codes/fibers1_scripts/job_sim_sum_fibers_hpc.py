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
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src_archive.skfibersv1.fibers import FIBERS #SOURCE CODE RUN
from src.skfibersv2.methods.util import plot_feature_tracking
from src.skfibersv2.methods.util import plot_kaplan_meir
from src.skfibersv2.methods.util import cox_prop_hazard
# from src.skfibersv2.methods.util import plot_bin_population_heatmap
from src.skfibersv2.methods.util import plot_custom_bin_population_heatmap

#from skfibers.fibers import FIBERS #PIP INSTALL RUN

covariates = None #Manually included in script

def prepare_data(df, duration_name, label_name, covariates):
    # Make list of feature names (i.e. columns that are not outcome, censor, or covariates)
    feature_names = list(df.columns)
    if covariates != None:
        exclude = covariates + [duration_name,label_name]
    else:
        exclude = [duration_name,label_name]
    feature_names = [item for item in feature_names if item not in exclude]

    # Remove invariant feature columns (data cleaning)
    cols_to_drop = []
    for col in feature_names:
        if len(df[col].unique()) == 1:
            cols_to_drop.append(col)
    df.drop(columns=cols_to_drop, inplace=True)
    feature_names = [item for item in feature_names if item not in cols_to_drop]
    print("Dropped "+str(len(cols_to_drop))+" invariant feature columns.")

    return df, feature_names

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

                # Ordering the bin scores from best to worst
        durations_no, durations_mm, event_observed_no, event_observed_mm, top_bin = fibers.get_duration_event(bin_index)
        results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                               event_observed_B=event_observed_mm)
        
        bin = fibers.bins[top_bin]
        bin_feature_list = fibers.bins[top_bin]
        top_bin_pop.append(bin)
        data_preped = fibers.check_x_y(data, None)
        data_preped, feature_names = prepare_data(data_preped, fibers.duration_name, fibers.label_name, covariates)
        population = feature_names

        group_threshold = 0
        log_rank_score = results.test_statistic
        log_rank_score_p_value = results.p_value
        bin_bin_size = len(fibers.bins[top_bin])
        count_bt = len(durations_no)
        count_at = len(durations_mm)
        group_strata_prop = min(count_bt/(count_bt+count_at),count_at/(count_bt+count_at))
        bin_birth_iteration = np.nan
        ideal_iteration_calc = ideal_iteration(ideal_count, bin_feature_list, bin_birth_iteration)
        
        summary, bin_HR, bin_HR_CI, bin_HR_p_value = get_cox_prop_hazard_unadjust(fibers, data)
        summary, bin_adj_HR, bin_adj_HR_CI, bin_adj_HR_p_value = get_cox_prop_hazard_adjusted(fibers, data)



        residuals_score = None

        results_list = [bin_feature_list, group_threshold, 
                        np.nan, np.nan, log_rank_score,
                        log_rank_score_p_value, bin_bin_size, group_strata_prop, count_bt, count_at, 
                        bin_birth_iteration, np.nan, np.nan, residuals_score, np.nan,
                        # residuals not implemted in script
                        # bin.HR, bin.HR_CI, bin.HR_p_value, bin.adj_HR, bin.adj_HR_CI, bin.adj_HR_p_value, 
                        bin_HR, bin_HR_CI, bin_HR_p_value, bin_adj_HR, bin_adj_HR_CI, bin_adj_HR_p_value,
                        str(bin_feature_list).count('P'), str(bin_feature_list).count('R'), 
                        np.nan,
                        accuracy_score(fibers.predict(data),true_risk_group) if true_risk_group is not None else None,
                        np.nan, data_name] 
        df.loc[len(df)] = results_list

        #Update metric lists
        accuracy.append(accuracy_score(fibers.predict(data),true_risk_group) if true_risk_group is not None else None)
        num_P.append(str(bin_feature_list).count('P'))
        num_R.append(str(bin_feature_list).count('R'))
        if ideal_iteration(ideal_count, bin_feature_list, bin_birth_iteration) != None:
            ideal += 1
            ideal_iter.append(bin_birth_iteration)
        if group_threshold == ideal_threshold:
            ideal_thresh += 1
        threshold.append(group_threshold)
        if log_rank_score != None:
            log_rank.append(log_rank_score)
        if residuals_score != None:
            residuals.append(residuals_score)
        if bin_HR != None:
            unadj_HR.append(bin_HR)
        if bin_adj_HR != None:
            adj_HR.append(bin_adj_HR)
        group_balance.append(group_strata_prop)
        runtime.append(fibers.elapsed_time)
        if str(bin_feature_list).count('T') > 0:
            tc += 1
        bin_size.append(bin_bin_size)
        birth_iteration.append(bin_birth_iteration)

        #Generate Figures:
        #Kaplan Meir Plot
        # fibers.get_kaplan_meir(data,bin_index,save=True,show=False, output_folder=target_folder,data_name=data_name+'_'+str(random_seed))
        try:
            plot_kaplan_meir(durations_no, event_observed_no, durations_mm, event_observed_mm,
                            show=False,save=True,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))
        except Exception as e:
            print("Exception in KM Plot", e, "Dataset", data_name)
            print(durations_no, event_observed_no, durations_mm, event_observed_mm, sep='\n')

        #Bin Population Heatmap
        group_names=["P", "R"]
        legend_group_info = ['Not in Bin','Predictive Feature in Bin','Non-Predictive Feature in Bin'] #2 default colors first followed by additional color descriptions in legend
        colors = [(.95, .95, 1),(0, 0, 1),(0.1, 0.1, 0.1)] #very light blue, blue, ---Alternatively red (1, 0, 0)  orange (1, 0.5, 0)
        max_bins = 100
        max_features = 100
        
        sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
        sorted_bin_list = list(sorted_bin_scores.keys())
        population = [fibers.bins[i] for i in sorted_bin_list]

        # fibers.get_custom_bin_population_heatmap_plot(group_names,legend_group_info,colors,max_bins,max_features,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))
        plot_custom_bin_population_heatmap(population, feature_names, group_names, legend_group_info, colors, max_bins, 
                                           max_features, show=False, save=True,
                                           output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

        # Feature Importance Estimates - No Feature Tracking Estimates in FIBERS-AT
        # fibers.get_feature_tracking_plot(max_features=50,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

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
    # population = pd.DataFrame([vars(instance) for instance in top_bin_pop])
    population = top_bin_pop
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

def get_cox_prop_hazard_unadjust(fibers,x, y=None, bin_index=0, use_bin_sums=False):
    if not fibers.hasTrained:
        raise Exception("FIBERS must be fit first")
    
    # PREPARE DATA ---------------------------------------
    df = fibers.check_x_y(x, y)
    df, feature_names = prepare_data(df, fibers.duration_name, fibers.label_name, covariates)

    # Sum instance values across features specified in the bin
    sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    feature_sums = df.loc[:,feature_names][fibers.bins[sorted_bin_list[bin_index]]].sum(axis=1)
    bin_df = pd.DataFrame({'Bin_'+str(bin_index):feature_sums})

    if not use_bin_sums:
        # Transform bin feature values according to respective bin threshold
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= 0 else 1)

    bin_df = pd.concat([bin_df,df.loc[:,fibers.duration_name],df.loc[:,fibers.label_name]],axis=1)
    summary = None
    HR, HR_CI, HR_p_value = None, None, None
    try:
        summary = cox_prop_hazard(bin_df,fibers.duration_name,fibers.label_name)
        HR = summary['exp(coef)'].iloc[0]
        HR_CI = str(summary['exp(coef) lower 95%'].iloc[0])+'-'+str(summary['exp(coef) upper 95%'].iloc[0])
        HR_p_value = summary['p'].iloc[0]
    except:
        HR = 0
        HR_CI = None
        HR_p_value = None
        pass

    df = None
    return summary, HR, HR_CI, HR_p_value

def get_cox_prop_hazard_adjusted(fibers,x, y=None, bin_index=0, use_bin_sums=False):
    if not fibers.hasTrained:
        raise Exception("FIBERS must be fit first")

    # PREPARE DATA ---------------------------------------
    df = fibers.check_x_y(x, y)
    df, feature_names = prepare_data(df, fibers.duration_name, fibers.label_name, covariates)

    # Sum instance values across features specified in the bin
    
    sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())
    feature_sums = df.loc[:,feature_names][fibers.bins[sorted_bin_list[bin_index]]].sum(axis=1)
    bin_df = pd.DataFrame({'Bin_'+str(bin_index):feature_sums})

    if not use_bin_sums:
        # Transform bin feature values according to respective bin threshold
        bin_df['Bin_'+str(bin_index)] = bin_df['Bin_'+str(bin_index)].apply(lambda x: 0 if x <= 0 else 1)

    bin_df = pd.concat([bin_df,df.loc[:,fibers.duration_name],df.loc[:,fibers.label_name]],axis=1)
    summary = None
    adj_HR, adj_HR_CI, adj_HR_p_value = None, None, None

    try:
        bin_df = pd.concat([bin_df,df.loc[:,covariates]],axis=1)
        summary = cox_prop_hazard(bin_df,fibers.duration_name,fibers.label_name)
        adj_HR = summary['exp(coef)'].iloc[0]
        adj_HR_CI = str(summary['exp(coef) lower 95%'].iloc[0])+'-'+str(summary['exp(coef) upper 95%'].iloc[0])
        adj_HR_p_value = summary['p'].iloc[0]
    except:
        adj_HR = 0
        adj_HR_CI = None
        adj_HR_p_value = None
        pass

    df = None
    return summary, adj_HR, adj_HR_CI, adj_HR_p_value

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
