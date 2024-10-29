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
sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS/')
from src_archive.skfibersv1.fibers import FIBERS #SOURCE CODE RUN
from src.skfibers.methods.util import plot_feature_tracking
from src.skfibers.methods.util import plot_kaplan_meir
from src.skfibers.methods.util import cox_prop_hazard
# from src.skfibers.methods.util import plot_bin_population_heatmap
from src.skfibers.methods.util import plot_custom_bin_population_heatmap

#from skfibers.fibers import FIBERS #PIP INSTALL RUN


covariates = [
              'shared', 'DCD', 'DON_AGE', 'donage_slope_ge18', 'dcadcodanox', 'dcadcodcva', 'dcadcodcnst', 'dcadcodoth', 'don_cmv_negative',
              'don_htn_0c', 'ln_don_wgt_kg_0c', 'ln_don_wgt_kg_0c_s55', 'don_ecd', 'age_ecd', 'yearslice', 'REC_AGE_AT_TX',
              'rec_age_spline_35', 'rec_age_spline_50', 'rec_age_spline_65', 'diab_noted', 'age_diab', 'dm_can_age_spline_50',
              'can_dgn_htn_ndm', 'can_dgn_pk_ndm', 'can_dgn_gd_ndm', 'rec_prev_ki_tx', 'rec_prev_ki_tx_dm', 'rbmi_0c', 'rbmi_miss',
              'rbmi_gt_20', 'rbmi_DM', 'rbmi_gt_20_DM', 'ln_c_hd_m', 'ln_c_hd_0c', 'ln_c_hd_m_ptx', 'PKPRA_MS', 'PKPRA_1080',
              'PKPRA_GE80', 'hispanic', 'CAN_RACE_BLACK', 'CAN_RACE_asian', 'CAN_RACE_WHITE', 'Agmm0']


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
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'A,B,C,DRB1,DQA1,DQB1')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0)

    #parser.add_argument('--f', dest='figures_only', help='random seeds in experiment', type=str, default='False')

    options=parser.parse_args(argv[1:])

    datapath = options.datapath
    outputpath = options.outputpath
    random_seeds = options.random_seeds
    loci_list = options.loci_list.split(',')
    if options.cov_list == 'None':
        cov_list = None
    else:
        cov_list = options.cov_list.split(',')
    rare_filter = options.rare_filter
    print(loci_list)

    #Hard Coded Covariate Information
    #If there is a colinearity issue with calculating residuals, Keith indicated that we can remove dcadcodoth and/or PKPRA_MS from covariate list
    covariates = [
              'shared', 'DCD', 'DON_AGE', 'donage_slope_ge18', 'dcadcodanox', 'dcadcodcva', 'dcadcodcnst', 'dcadcodoth', 'don_cmv_negative',
              'don_htn_0c', 'ln_don_wgt_kg_0c', 'ln_don_wgt_kg_0c_s55', 'don_ecd', 'age_ecd', 'yearslice', 'REC_AGE_AT_TX',
              'rec_age_spline_35', 'rec_age_spline_50', 'rec_age_spline_65', 'diab_noted', 'age_diab', 'dm_can_age_spline_50',
              'can_dgn_htn_ndm', 'can_dgn_pk_ndm', 'can_dgn_gd_ndm', 'rec_prev_ki_tx', 'rec_prev_ki_tx_dm', 'rbmi_0c', 'rbmi_miss',
              'rbmi_gt_20', 'rbmi_DM', 'rbmi_gt_20_DM', 'ln_c_hd_m', 'ln_c_hd_0c', 'ln_c_hd_m_ptx', 'PKPRA_MS', 'PKPRA_1080',
              'PKPRA_GE80', 'hispanic', 'CAN_RACE_BLACK', 'CAN_RACE_asian', 'CAN_RACE_WHITE', 'Agmm0']
    
    #DRB345 has redundant covariates with DRB1, so both should not be specified together
    #Also, Malek indicated we will not corret for DPA1 or DPB1 for the time being.
    cov_typ_dict = {
        'A': ['AgmmA0', 'AgmmA1'],
        'B': ['AgmmB0', 'AgmmB1'],
        'C': ['Agmmc0', 'Agmmc1'],
        'DRB1':  ['AgmmDR0', 'AgmmDR1'],
        'DRB345':  ['AgmmDR0', 'AgmmDR1'],
        'DQA1':  ['Agmmdqa10', 'Agmmdqa11'],
        'DQB1':  ['Agmmdqb10', 'Agmmdqb11'],
        'DPA1':  ['Agmmdpa10', 'Agmmdpa11'],
        'DPB1':  ['Agmmdpb10', 'Agmmdpb11']} 
    
    #hard coded specific AA-MM positions to include in these analyses
    locus_range_dict = {
        'A': [1,182],
        'B': [1,182],
        'C': [1,182],
        'DRB1': [6,94],
        'DRB345': [6,94],
        'DQA1':  [6,94],
        'DQB1':  [6,95],
        'DPA1':  [6,94],
        'DPB1':  [6,94]}
    
    #Create Final Covariate List
    if cov_list != None:
        for covariate in cov_list:
            cov_sub_list = cov_typ_dict[covariate]
            for each in cov_sub_list:
                covariates.append(each) #add selected Ag covariate to primary covariates
    print(covariates) #temporary

    #Get algorithm name
    outputfolder  = outputpath.split('/')[-1]
    algorithm = outputfolder.split('_')[0]
    experiment = outputfolder.replace(algorithm+'_',"") #Unique experiment is identified by the output folder name

    #Get experiment name
    data_file = datapath.split('/')[-1]
    data_name = data_file.rstrip('.csv')

    target_folder = outputpath+'/'+data_name #target output subfolder

    #Make local summary output folder
    if not os.path.exists(target_folder+'/'+'summary'):
        os.mkdir(target_folder+'/'+'summary')  

    #Load/Process Dataset
    data = pd.read_csv(datapath)

    #Identify MM features to include as independent features
    MM_feature_list = []
    for locus in loci_list: #each specified locus to evaluate as independent features
        for j in range(locus_range_dict[locus][0],locus_range_dict[locus][1]+1):
            MM_feature_list.append('MM_'+str(locus)+'_'+str(j))

    #Missing data values check
    # missing_sum = data.isna().sum().sum()
    # if missing_sum > 0:
    #     print("Sum of data missing values:", missing_sum)

    #Data Cleaning
    if rare_filter > 0.0: #filter out rare features and invariant features
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        columns_to_remove = percentages[percentages < rare_filter].index.tolist()
        data = data.drop(columns=columns_to_remove)
        MM_feature_list = [x for x in MM_feature_list if x not in columns_to_remove]
    else: #filter out invariant features only
        # Calculate the percentage of occurrences greater than 0 for each column
        percentages = data.loc[:,MM_feature_list].apply(lambda x: (x > 0).mean())
        columns_to_remove = percentages[percentages == 0.0].index.tolist()
        data = data.drop(columns=columns_to_remove)
        MM_feature_list = [x for x in MM_feature_list if x not in columns_to_remove]


    #Define columns for replicate results summary:
    columns = ["Dataset Filename","Random Seed","Bin Features", "Threshold", "Fitness", "Pre-Fitness", "Log-Rank Score","Log-Rank p-value",
               "Bin Size", "Group Ratio", "Count At/Below Threshold", "Count Above Threshold", "Birth Iteration", 
               "Deletion Probability", "Cluster", "Residual", "Residual p-value", "Unadjusted HR", "Unadjusted HR CI",
               "Unadjusted HR p-value", "Adjusted HR", "Adjusted HR CI", "Adjusted HR p-value", "Runtime"]
    df = pd.DataFrame(columns=columns)

    #Make intial lists to store metrics across replications
    threshold = []
    log_rank = []
    residuals = []
    unadj_HR = []
    adj_HR = []
    group_balance = []
    runtime = []
    bin_size = []
    birth_iteration = []
    top_bin_pop = []
    all_feature_names = set()

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

        # feature_names = MM_feature_list + covariates + [fibers.duration_name] + [fibers.label_name]
        # print("Feature Names", feature_names) #temporary
        # for i in feature_names:
        #     if i not in data.columns:
        #         print(i)
        # data = data[feature_names]

        # # Remove invariant feature columns (data cleaning)
        # cols_to_drop = []
        # for col in feature_names:
        #     if len(data[col].unique()) == 1:
        #         cols_to_drop.append(col)
        # print("Cols to drop", cols_to_drop)
        # data.drop(columns=cols_to_drop, inplace=True)
        # feature_names = [item for item in feature_names if item not in cols_to_drop]
        # MM_feature_list = [item for item in MM_feature_list if item not in cols_to_drop]
        # covariates = [item for item in covariates if item not in cols_to_drop]

        # print("Dropped "+str(len(cols_to_drop))+" invariant feature columns.")
        # 
        data_preped = fibers.check_x_y(data, None)
        data_preped, feature_names = prepare_data(data_preped, fibers.duration_name, fibers.label_name, covariates)
        all_feature_names.update(set(feature_names))

        group_threshold = 0
        log_rank_score = results.test_statistic
        log_rank_score_p_value = results.p_value
        bin_bin_size = len(fibers.bins[top_bin])
        count_bt = len(durations_no)
        count_at = len(durations_mm)
        group_strata_prop = min(count_bt/(count_bt+count_at),count_at/(count_bt+count_at))
        bin_birth_iteration = np.nan
        
        summary, bin_HR, bin_HR_CI, bin_HR_p_value = get_cox_prop_hazard_unadjust(fibers, data)
        summary, bin_adj_HR, bin_adj_HR_CI, bin_adj_HR_p_value = get_cox_prop_hazard_adjusted(fibers, data)

        residuals_score = None

        # results_list = [data_name,random_seed, bin.feature_list, bin.group_threshold, bin.fitness, bin.pre_fitness, bin.log_rank_score,
        #                 bin.log_rank_p_value, bin.bin_size, bin.group_strata_prop, bin.count_bt, bin.count_at, 
        #                 bin.birth_iteration, bin.deletion_prop, bin.cluster, bin.residuals_score, bin.residuals_p_value,
        #                 bin.HR, bin.HR_CI, bin.HR_p_value, bin.adj_HR, bin.adj_HR_CI, bin.adj_HR_p_value, 
        #                 fibers.elapsed_time] 
        # df.loc[len(df)] = results_list

        results_list = [data_name,random_seed, bin_feature_list, group_threshold, 
                        np.nan, np.nan, log_rank_score,
                        log_rank_score_p_value, bin_bin_size, group_strata_prop, count_bt, count_at, 
                        bin_birth_iteration, np.nan, np.nan, residuals_score, np.nan,
                        # residuals not implemted in script
                        # bin.HR, bin.HR_CI, bin.HR_p_value, bin.adj_HR, bin.adj_HR_CI, bin.adj_HR_p_value, 
                        bin_HR, bin_HR_CI, bin_HR_p_value, bin_adj_HR, bin_adj_HR_CI, bin_adj_HR_p_value,
                        fibers.elapsed_time] 
        df.loc[len(df)] = results_list

        #Update metric lists
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
        
        sorted_bin_scores = dict(sorted(fibers.bin_scores.items(), key=lambda item: item[1], reverse=True))
        sorted_bin_list = list(sorted_bin_scores.keys())
        population = [fibers.bins[i] for i in sorted_bin_list]

        #Generate Top-bin Custom Heatmap across replicates
        # COLORS:    very light blue, blue, red, green, purple, pink, orange, yellow, light blue, grey
        all_colors = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.5, 0, 1),(1, 0, 1),(1, 0.5, 0),(1, 1, 0),(0, 1, 1),(0.5, 0.5, 0.5)] 
        max_bins = 100
        max_features = 100
        filtering = 1
        group_names = []
        legend_group_info = ['Not in Bin']
        colors = [(.95, .95, 1)]
        i = 0
        for locus in loci_list:
            group_names.append('MM_'+str(locus))
            legend_group_info.append(locus)
            colors.append(all_colors[i])
            i += 1

        # fibers.get_custom_bin_population_heatmap_plot(group_names,legend_group_info,colors,max_bins,max_features,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))
        # plot_custom_bin_population_heatmap(population, feature_names, group_names, legend_group_info, colors, max_bins, 
        #                                    max_features, show=False, save=True,
        #                                    output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

        # Feature Importance Estimates - No Feature Tracking Estimates in FIBERS-AT
        # fibers.get_feature_tracking_plot(max_features=50,save=True,show=False,output_folder=target_folder,data_name=data_name+'_'+str(random_seed))

    #Save replicate results as csv

    df.to_csv(target_folder+'/'+'summary'+'/'+data_name+'_summary'+'.csv', index=False)

    #Generate experiment summary 'master list'
    master_columns = ["Algorithm","Experiment", "Dataset", 
                    "Threshold", "Threshold (SD)",
                    "Log-Rank Score", "Log-Rank Score (SD)", 
                    "Residual", "Residual (SD)", 
                    "Unadjusted HR", "Unadjusted HR (SD)", 
                    "Adjusted HR", "Adjusted HR (SD)", 
                    "Group Ratio", "Group Ratio (SD)",
                    "Runtime", "Runtime (SD)", 
                    "Bin Size", "Bin Size (SD)", 
                    "Birth Iteration", "Birth Iteration (SD)"]
    
    df_master = pd.DataFrame(columns=master_columns)
    master_results_list = [algorithm,experiment,data_name,
                        np.mean(threshold),np.std(threshold), 
                        None if len(log_rank) == 0 else np.mean(log_rank), None if len(log_rank) == 0 else np.std(log_rank) ,
                        None if len(residuals) == 0 else np.mean(residuals), None if len(residuals) == 0 else np.std(residuals), 
                        None if len(unadj_HR) == 0 else np.mean(unadj_HR), None if len(unadj_HR) == 0 else np.std(unadj_HR),
                        None if len(adj_HR) == 0 else np.mean(adj_HR), None if len(adj_HR) == 0 else np.std(adj_HR), 
                        np.mean(group_balance),np.std(group_balance),
                        np.mean(runtime),np.std(runtime), 
                        np.mean(bin_size),np.std(bin_size), 
                        np.mean(birth_iteration),np.std(birth_iteration)]
    
    df_master.loc[len(df_master)] = master_results_list
    #Save master results as csv
    df_master.to_csv(target_folder+'/'+'summary'+'/'+data_name+'_master_summary'+'.csv', index=False)

    #Generate Top-bin Custom Heatmap across replicates
    # COLORS:    very light blue, blue, red, green, purple, pink, orange, yellow, light blue, grey
    all_colors = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.5, 0, 1),(1, 0, 1),(1, 0.5, 0),(1, 1, 0),(0, 1, 1),(0.5, 0.5, 0.5)] 
    max_bins = 100
    max_features = 100
    filtering = 1
    group_names = []
    legend_group_info = ['Not in Bin']
    colors = [(.95, .95, 1)]
    i = 0
    for locus in loci_list:
        group_names.append('MM_'+str(locus))
        legend_group_info.append(locus)
        colors.append(all_colors[i])
        i += 1

    #Generate Top-bin Custom Heatmap (filtering out zeros) across replicates
    # population = pd.DataFrame([vars(instance) for instance in top_bin_pop])
    population = top_bin_pop
    plot_custom_top_bin_population_heatmap(population, list(all_feature_names), group_names,legend_group_info,colors,max_bins,max_features,filtering=filtering,save=True,show=False,output_folder=target_folder+'/'+'summary',data_name=data_name)

    #Generate Top-bin Basic Heatmap (filtering out zeros) across replicates
    gdf = plot_bin_population_heatmap(population, list(all_feature_names), filtering=filtering, show=False,save=True,output_folder=target_folder+'/'+'summary',data_name=data_name)

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
