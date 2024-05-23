import os
import argparse
import sys
import pandas as pd
import pickle
import collections
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
#from sklearn.metrics import accuracy_score
sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS/')
from src.skfibers.fibers import FIBERS #SOURCE CODE RUN
#from skfibers.fibers import FIBERS #PIP INSTALL RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 10)
    parser.add_argument('--re', dest='replicates', help='number of data replicates', type=int, default= 10)
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')

    options=parser.parse_args(argv[1:])

    datafolder = options.datafolder
    writepath = options.writepath+'/output'
    outputfolder = options.outputfolder
    targetfolder = writepath +'/'+outputfolder
    replicates = options.replicates
    random_seeds = options.random_seeds
    loci_list = options.loci_list

    datanames = []
    for dataname in os.listdir(datafolder):
        datanames.append(dataname)
    base_name = datanames[0].split('.')[0] #remove file extension
    base_name = base_name.split('_')[0] #remove replicate number

    imp_sum_path = targetfolder+'/imp_summary'
    if not os.path.exists(imp_sum_path):
        os.mkdir(imp_sum_path) 

    #Create Combined Master List - results for each imputation analysis averaged over random seed runs
    master_files = []
    for replicate in range(1,replicates+1): #one indexed datasets
        data_name = base_name+'_'+str(replicate)
        subfolder = targetfolder+'/'+data_name
        master_files.append(subfolder+'/summary/'+data_name+'_master_summary.csv')
    #Read the header from the first CSV file
    header = pd.read_csv(master_files[0], nrows=0).columns.tolist()
    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame(columns=header)
    # Iterate over each CSV file (excluding the first one) and add its first row to the combined DataFrame
    for master_file in master_files:
        data = pd.read_csv(master_file, nrows=1, header=0, names=header)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    # Save the combined data to a new CSV file
    combined_data.to_csv(targetfolder+'/'+outputfolder+'_master_summary.csv', index=False)

    #Create Summary List of Individual results
    all_summary_files = []
    for replicate in range(1,replicates+1): #one indexed datasets
        data_name = base_name+'_'+str(replicate)
        subfolder = targetfolder+'/'+data_name
        all_summary_files.append(subfolder+'/summary/'+data_name+'_summary.csv')
    #Read the header from the first CSV file
    header = pd.read_csv(all_summary_files[0], nrows=0).columns.tolist()
    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame(columns=header)
    # Iterate over each CSV file (excluding the first one) and add its first row to the combined DataFrame
    for summary_file in all_summary_files:
        data = pd.read_csv(summary_file, nrows=random_seeds, header=0, names=header)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    # Save the combined data to a new CSV file
    combined_data.to_csv(targetfolder+'/'+outputfolder+'_summary.csv', index=False)

    #Custom heatmaps that combines the top bins across data imputations for each random seed (10 figures)
    all_top_bins = []
    all_bin_labels = []
    for random_seed in range(0, random_seeds):  #for each replicate
        top_bin_pop = []
        for replicate in range(1,replicates+1): #one indexed datasets
            data_name = base_name+'_'+str(replicate)

            #Unpickle FIBERS Object
            with open(targetfolder+'/'+data_name+'/'+data_name+'_'+str(random_seed)+'_fibers.pickle', 'rb') as f:
                fibers = pickle.load(f)

            #Get top bin object for current fibers population
            bin_index = 0 #top bin
            bin = fibers.set.bin_pop[bin_index]
            feature_names = fibers.feature_names
            top_bin_pop.append(bin)
            all_top_bins.append(bin)
            all_bin_labels.append("Seed_" + str(random_seed + 1)+" Imp_"+str(replicate))

        #Generate Top-bin Custom Heatmap across replicates
        # COLORS:    very light blue, blue, red, green, purple, pink, orange, yellow, light blue, grey
        all_colors = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.5, 0, 1),(1, 0, 1),(1, 0.5, 0),(1, 1, 0),(0, 1, 1),(0.5, 0.5, 0.5)] 
        max_bins = 100
        max_features = 100
        group_names = []
        legend_group_info = ['Not in Bin']
        colors = [(.95, .95, 1)]
        filtering = 1
        i = 0
        for locus in loci_list:
            group_names.append('MM_'+str(locus))
            legend_group_info.append(locus)
            colors.append(all_colors[i])
            i += 1
            print('next')
            print(i)
            print(locus)

        #Generate Top-bin Custom Heatmap (filtering out zeros) across replicates
        population = pd.DataFrame([vars(instance) for instance in top_bin_pop])
        population = population['feature_list']
        plot_custom_top_bin_population_heatmap(population, feature_names, group_names,legend_group_info,colors,max_bins,max_features,filtering=filtering, all_bin_labels=None,save=True,show=False,output_folder=imp_sum_path,data_name=base_name+'_rs_'+str(random_seed))

        #Generate Top-bin Basic Heatmap (filtering out zeros) across replicates
        gdf = plot_bin_population_heatmap(population, feature_names, filtering=filtering, all_bin_labels=None,show=False,save=True,output_folder=imp_sum_path,data_name=base_name+'_rs_'+str(random_seed))

        #Generate feature frequency barplot
        pd.DataFrame(gdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False).plot.bar(figsize=(12, 4),
                        ylabel='Count Across Top Bins', xlabel='Dataset Feature')
        plt.savefig(imp_sum_path+'/'+base_name+'_rs_'+str(random_seed)+'_top_bins_feature_frequency_barplot.png', bbox_inches="tight")

    #Generate Top-bin Custom Heatmap across replicates and random seeds
    # COLORS:    very light blue, blue, red, green, purple, pink, orange, yellow, light blue, grey
    all_colors = [(0, 0, 1),(1, 0, 0),(0, 1, 0),(0.5, 0, 1),(1, 0, 1),(1, 0.5, 0),(1, 1, 0),(0, 1, 1),(0.5, 0.5, 0.5)] 
    max_bins = 100
    max_features = 100
    group_names = []
    legend_group_info = ['Not in Bin']
    colors = [(.95, .95, 1)]
    filtering = 1
    i = 0
    for locus in loci_list:
        group_names.append('MM_'+str(locus))
        legend_group_info.append(locus)
        colors.append(all_colors[i])
        i += 1

    #Generate Top-bin Custom Heatmap (filtering out zeros) across replicates and random seeds
    population = pd.DataFrame([vars(instance) for instance in all_top_bins])
    population = population['feature_list']
    plot_custom_top_bin_population_heatmap(population, feature_names, group_names,legend_group_info,colors,max_bins,max_features,filtering=filtering, all_bin_labels=all_bin_labels,save=True,show=False,output_folder=imp_sum_path,data_name=base_name)

    #Generate Top-bin Basic Heatmap (filtering out zeros) across replicates
    gdf = plot_bin_population_heatmap(population, feature_names, filtering=filtering, all_bin_labels=all_bin_labels,show=False,save=True,output_folder=imp_sum_path,data_name=base_name)

    #Generate feature frequency barplot
    pd.DataFrame(gdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False).plot.bar(figsize=(12, 4),
                    ylabel='Count Across Top Bins', xlabel='Dataset Feature')
    plt.savefig(imp_sum_path+'/'+base_name+'_top_bins_feature_frequency_barplot.png', bbox_inches="tight")


def match_prefix(feature, group_names):
    """
    :param feature: the feature
    :param group_names: the list of group names, must be exhaustive
    """
    for group_label in group_names:
        if feature.startswith(group_label):
            return group_label

    return "None"

def plot_bin_population_heatmap(population, feature_names,filtering=None,all_bin_labels=None,show=True,save=False,output_folder=None,data_name=None):
    """
    :param population: a list where each element is a list of specified features
    :param feature_list: an alphabetically sorted list containing each of the possible feature
    """
    fontsize = 20
    feature_count = len(feature_names)
    if all_bin_labels == None:
        bin_names = []
        for i in range(len(population)):
            bin_names.append("Seed " + str(i + 1))
    else:
        bin_names = all_bin_labels

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
        plt.savefig(output_folder+'/'+data_name+'_basic_pop_heatmap.png', bbox_inches="tight")

    if show:
        plt.show()

    return graph_df

def plot_custom_top_bin_population_heatmap(population,feature_names,group_names,legend_group_info,colors,max_bins,max_features,filtering=None,all_bin_labels=None,show=True,save=False,output_folder=None,data_name=None):
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
    if all_bin_labels == None:
        bin_names = []
        for i in range(len(population)):
            bin_names.append("Seed " + str(i + 1))
    else:
        bin_names = all_bin_labels

    graph_df = pd.DataFrame(graph_df, bin_names, feature_names) #data, index, columns

    if filtering != None:
        tdf = graph_df
        tdf = pd.DataFrame(tdf.sum(axis=0), columns=['Count']).sort_values('Count', ascending=False)
        tdf = tdf[tdf['Count'] >= filtering]
        graph_df = graph_df[list(tdf.index)]
        feature_count = len(graph_df.columns)
        print(feature_count)

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
