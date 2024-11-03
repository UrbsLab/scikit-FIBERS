import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import collections
from sklearn.metrics import accuracy_score
sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
from scipy.stats import wilcoxon


def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputpath', help='', type=str, default = None) #full path/filename
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    parser.add_argument('--re', dest='replicates', help='number of data replicates', type=int, default= 10)

    options=parser.parse_args(argv[1:])

    writepath = options.writepath +'output/'
    outputpath = options.outputpath
    random_seeds = options.random_seeds
    replicates = options.replicates

    p_val = 0.05

    #Get names of all experiment folders
    experiment_folder_names = [name for name in os.listdir(writepath) if os.path.isdir(os.path.join(writepath, name))]
    print(experiment_folder_names)
    #Get names of all dataset folders (used within each experiment)
    dataset_folder_names = [name for name in os.listdir(writepath+experiment_folder_names[0]) if os.path.isdir(os.path.join(writepath+experiment_folder_names[0], name))]
    dataset_folder_names.remove('imp_summary') #extra folder added to real world analysis that we don't want to include.
    print(dataset_folder_names)

    significance_metrics = ['Adjusted HR','Unadjusted HR','Log-Rank Score','Residual','Threshold','Group Ratio','Count At/Below Threshold','Count Above Threshold','Bin Size','Birth Iteration','Runtime']
    #count_metrics = ['Ideal Bin','Ideal Threshold','TC1 Present']

    #FIBERS 1.0 DOES NOT HAVE - fitness or pre-fitness, birth iteration, deletion prob, cluster or residuals #no comparison

    #Global comparisons
        #set up global comparsisons across all 100 top bins comparing all key metrics
    #Impuation comparisons
        # Do comparsisons between imputated datasets within the same 

    # CORE EXPERIMENTS ***************************************
    #Table - FIBERS2 Fit compare DP 3
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_product_DP3']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare DP 0
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_0'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP0','Fibers2.0_hla_7locus_residuals_DP0','Fibers2.0_hla_7locus_product_DP0']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP0'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 Fit compare Basic
    table_name = 'Real_7Locus_FIBERS2_Fit_Basic'
    variable_element = ['Fibers2.0_hla_7locus_logrank_Basic','Fibers2.0_hla_7locus_residuals_Basic','Fibers2.0_hla_7locus_product_Basic']
    baseline = 'Fibers2.0_hla_7locus_logrank_Basic'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS1v2 compare Basic
    table_name = 'Real_7Locus_FIBERS1v2_Fit_Basic'
    variable_element = ['Fibers1.0_hla_7locus_baseline','Fibers2.0_hla_7locus_logrank_Basic']
    baseline = 'Fibers1.0_hla_7locus_baseline'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - 7Locus LogRank Compare
    table_name = 'Real_7Locus_LogRank'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_7locus_logrank_DP0','Fibers2.0_hla_7locus_logrank_Basic','Fibers1.0_hla_7locus_baseline']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - 7Locus Residuals Compare
    table_name = 'Real_7Locus_Residuals'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_7locus_residuals_DP0','Fibers2.0_hla_7locus_residuals_Basic']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - 7Locus Product Compare
    table_name = 'Real_7Locus_Product'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_7locus_product_DP0','Fibers2.0_hla_7locus_product_Basic']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    # LOCUS COUNT EXPERIMENTS ***************************************
    #Table - FIBERS2 4 Locus Fit
    table_name = 'Real_4Locus_FIBERS2_Fit_DP_3'
    variable_element = ['Fibers2.0_hla_4locus_logrank_DP3','Fibers2.0_hla_4locus_residuals_DP3','Fibers2.0_hla_4locus_product_DP3']
    baseline = 'Fibers2.0_hla_4locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 2 Locus Fit
    table_name = 'Real_2Locus_FIBERS2_Fit_DP_3'
    variable_element = ['Fibers2.0_hla_2locus_logrank_DP3','Fibers2.0_hla_2locus_residuals_DP3','Fibers2.0_hla_2locus_product_DP3']
    baseline = 'Fibers2.0_hla_2locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - LocusCount LogRank Compare
    table_name = 'Real_LogRank_LocusCount'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3','Fibers2.0_hla_4locus_logrank_DP3','Fibers2.0_hla_2locus_logrank_DP3']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - LocusCount Residuals Compare
    table_name = 'Real_Residuals_LocusCount'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3','Fibers2.0_hla_4locus_residuals_DP3','Fibers2.0_hla_2locus_residuals_DP3']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - LocusCount Product Compare
    table_name = 'Real_Product_LocusCount'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3','Fibers2.0_hla_4locus_product_DP3','Fibers2.0_hla_2locus_product_DP3']
    baseline = 'Fibers2.0_hla_7locus_product_DP3'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    # FILTER AND THRESH EXPERIMENTS ***************************************
    #Table - FIBERS2 7 Locus Fit Unfiltered
    table_name = 'Real_7Locus_Unfiltered_FIBERS2_Fit_DP_3'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3_nofilter']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_nofilter'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 7 Locus Fit Unfiltered Thresh 0-10
    table_name = 'Real_7Locus_Unfiltered_FIBERS2_Fit_DP_3_T_10'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_nofilter_T_10','Fibers2.0_hla_7locus_residuals_DP3_nofilter_T_10','Fibers2.0_hla_7locus_product_DP3_nofilter_T_10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_nofilter_T_10'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - FIBERS2 7 Locus Fit Thresh 0-10
    table_name = 'Real_7Locus_FIBERS2_Fit_DP_3_T_10'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_T_10','Fibers2.0_hla_7locus_residuals_DP3_T_10','Fibers2.0_hla_7locus_product_DP3_T_10']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_T_10'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)


    #Table - Filtering and THRESH LogRank Compare
    table_name = 'Real_LogRank_Other'
    variable_element = ['Fibers2.0_hla_7locus_logrank_DP3_nofilter_T_10','Fibers2.0_hla_7locus_logrank_DP3_T_10','Fibers2.0_hla_7locus_logrank_DP3_nofilter','Fibers2.0_hla_7locus_logrank_DP3']
    baseline = 'Fibers2.0_hla_7locus_logrank_DP3_nofilter_T_10'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - Filtering and THRESH Residuals Compare
    table_name = 'Real_Residuals_Other'
    variable_element = ['Fibers2.0_hla_7locus_residuals_DP3_nofilter_T_10','Fibers2.0_hla_7locus_residuals_DP3_T_10','Fibers2.0_hla_7locus_residuals_DP3_nofilter','Fibers2.0_hla_7locus_residuals_DP3']
    baseline = 'Fibers2.0_hla_7locus_residuals_DP3_nofilter_T_10'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)

    #Table - Filtering and THRESH Product Compare
    table_name = 'Real_Product_Other'
    variable_element = ['Fibers2.0_hla_7locus_product_DP3_nofilter_T_10','Fibers2.0_hla_7locus_product_DP3_T_10','Fibers2.0_hla_7locus_product_DP3_nofilter','Fibers2.0_hla_7locus_product_DP3']
    baseline = 'Fibers2.0_hla_7locus_product_DP3_nofilter_T_10'
    stat_list_columns = ['Variable Element','Adjusted HR','Unadjusted HR','Log-Rank Score','Residuals','Threshold','Group Ratio','Low Risk','High Risk','Bin Size','Birth Iteration','Runtime']
    run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val)




def run_analysis(writepath,outputpath,significance_metrics,table_name,variable_element,baseline,stat_list_columns,p_val):
    dataframe_stat_list = []
    raw_dataframes = []
    baseline_index = variable_element.index(baseline)

    for var in variable_element:
        #master_summary = writepath+var+'/'+var+'_master_summary.csv'
        summary = writepath+var+'/'+var+'_summary.csv'

        # Load the stats summary CSV file into a pandas DataFrame
        df_sum = pd.read_csv(summary)
        raw_dataframes.append(df_sum)
        formated_df = format_data(df_sum,stat_list_columns, var)
        dataframe_stat_list.append(formated_df) # Add a row of results for each experiment

    #now have the basic stats collected for each experiment and dataset
    # Determine statistical significance differences for significance_metrics 
    # add '* next to any non-baseline results where a significant difference observed on contrast with baseline
    for metric in significance_metrics:
        #Get baseline data for metric
        base_col = raw_dataframes[baseline_index][metric]
        for i in range(len(variable_element)):
            #Get comparison data for metric
            if i != baseline_index: #don't include baseline data
                compare_col = raw_dataframes[i][metric]
                #Apply Wilcoxon Significance comparison
                try:
                    is_sig = wilcoxon_sig(base_col,compare_col,p_val)
                    if is_sig: # indicate significance within dataframe stat_list
                        # Find appropriate metric
                        #cell_value = str(dataframe_stat_list[i][metric])
                        #dataframe_stat_list[i][metric] = str(dataframe_stat_list[i][metric])+'*'
                        dataframe_stat_list[i][metric] += '*'
                except ValueError as e:
                    print(f"Error while performing the wilcoxon test: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

    # combine experiment results into a single dataframe
    combined_df = pd.concat(dataframe_stat_list, ignore_index=False)
    #transpose for easy copying
    combined_df_T = combined_df.T
    combined_df_T.to_csv(outputpath+'/'+str(table_name)+'_Table.csv', index=False)

def wilcoxon_sig(col1,col2,p_val):
    statistic, p_value = wilcoxon(col1, col2)
    print('Comparison: '+str(statistic)+ ' '+ str(p_value))
    if p_value <= p_val:
        return True

def format_data(df,stat_list_columns, var):
    #significance_metrics = ['Adjusted HR','Unadjusted HR','Log-Rank Score','Residual','Threshold','Group Ratio','Count At/Below Threshold','Count Above Threshold','Bin Size','Birth Iteration','Runtime']
    experiment = []
    experiment.append(var)
    experiment.append(str(round(df['Adjusted HR'].mean(),3))+' ('+str(round(df['Adjusted HR'].std(),3))+')') #adj HR
    experiment.append(str(round(df['Unadjusted HR'].mean(),3))+' ('+str(round(df['Unadjusted HR'].std(),3))+')') #unadj HR                           
    experiment.append(str(round(df['Log-Rank Score'].mean(),1))+' ('+str(round(df['Log-Rank Score'].std(),1))+')') #log rank 
    experiment.append(str(round(df['Residual'].mean(),2))+' ('+str(round(df['Residual'].std(),2))+')') #Residual
    experiment.append(str(round(df['Threshold'].mean(),1))+' ('+str(round(df['Threshold'].std(),1))+')') #Threshold
    experiment.append(str(round(df['Group Ratio'].mean(),2))+' ('+str(round(df['Group Ratio'].std(),2))+')') #Group Ratio
    experiment.append(str(round(df['Count At/Below Threshold'].mean(),0))+' ('+str(round(df['Count At/Below Threshold'].std(),0))+')') #Count At/Below Threshold
    experiment.append(str(round(df['Count Above Threshold'].mean(),0))+' ('+str(round(df['Count Above Threshold'].std(),0))+')') #Count Above Threshold
    experiment.append(str(round(df['Bin Size'].mean(),1))+' ('+str(round(df['Bin Size'].std(),1))+')') #Bin Size
    experiment.append(str(round(df['Birth Iteration'].mean(),1))+' ('+str(round(df['Birth Iteration'].std(),1))+')') #Birth Iteration
    experiment.append(str(round(df['Runtime'].mean()/60,1))+' ('+str(round(df['Runtime'].std()/60,1))+')') #Runtime

    new_df = pd.DataFrame([experiment], columns=stat_list_columns)
    return new_df


if __name__=="__main__":
    sys.exit(main(sys.argv))
