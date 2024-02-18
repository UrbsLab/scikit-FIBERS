import numpy as np
import pandas as pd
import copy
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import ranksums

class BIN:
    def __init__(self):
        self.feature_list = [] # List of feature names (across which instance values are summed)
        self.group_threshold = None # Threshold after which an instance is place in the 'above threshold' group - determines group strata of instances
        self.fitness = None # Bin fitness (higher fitness is better) - proportional to parent selection probability, and inversely proportional to deletion probability
        self.pre_fitness = None
        self.metric = None  # Metric score of applied evaluation metric
        self.p_value = None # p-value of applied evaluation metric (if available)
        self.bin_size = None # Number of features included in bin
        self.group_strata_prop = None # Proportion of instances in the smallest group (e.g. 0.5 --> equal number of instances in each group)
        self.count_bt = None # Instance count at/below threshold
        self.count_at = None # Instance count above threshold
        self.birth_iteration = None # Iteration where bin was introduced to population
        self.residuals_score = None
        self.residuals_p_value = None


    def initialize_random(self,feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,iteration,random):
        self.birth_iteration = iteration
        # Initialize features in bin
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)
        if group_thresh != None: # Defined group threshold
            self.group_threshold = group_thresh
        else: # Adaptive group threshold
            self.group_threshold = random.randint(min_thresh,max_thresh)
    

    def evaluate(self,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,covariate_df):
        # Sum instance values across features specified in the bin
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})

        # Create evaluation dataframe including bin sum feature with 
        bin_df = pd.concat([bin_df,outcome_df,censor_df],axis=1)

        if (group_thresh == None and not threshold_evolving) or (group_thresh == None and iteration == iterations-1): #Adaptive thresholding activated (always applied on last iteration)
            # Select best threshold by evaluating all considered
            best_score = 0
            for threshold in range(min_thresh, max_thresh + 1):
                score, p_value,residuals_score,residuals_p_value = self.evaluate_for_threshold(bin_df,outcome_label,censor_label,outcome_type,fitness_metric,
                        log_rank_weighting,threshold,residuals,covariate_df)
                if score > best_score:
                    self.metric = score
                    self.group_threshold = threshold
                    best_score = score
        else: #Use the given group threshold to evaluate the bin
            score, p_value,residuals_score,residuals_p_value = self.evaluate_for_threshold(bin_df,outcome_label,censor_label,outcome_type,fitness_metric,
                        log_rank_weighting,self.group_threshold,residuals,covariate_df)
        self.metric = score
        self.p_value = p_value
        self.residuals_score = residuals_score
        self.residuals_p_value = residuals_p_value
        self.bin_size = len(self.feature_list)


    def evaluate_for_threshold(self,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,log_rank_weighting,group_threshold,residuals,covariate_df):
        # Apply selected evaluation strategy/metric
        if outcome_type == 'survival':
            residuals_score = None
            residuals_p_value = None

            if fitness_metric == 'log_rank':
                #Create dataframes including instances from either strata-groups
                low_df = bin_df[bin_df['feature_sum'] <= group_threshold]
                high_df = bin_df[bin_df['feature_sum'] > group_threshold]
                low_outcome = low_df[outcome_label].to_list()
                high_outcome = high_df[outcome_label].to_list()
                low_censor = low_df[censor_label].to_list()
                high_censor =high_df[censor_label].to_list()
                self.count_bt = len(low_outcome)
                self.count_at = len(high_outcome)
                try:
                    results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                    score = results.test_statistic #test all thresholds by default in initial pop.
                    p_value = results.p_value
                except:
                    score = 0
                    p_value = None

                if fitness_metric == 'residuals': # In addition to log_rank, calculate residuals differences between groups
                    bin_residuals = residuals.loc[bin_df['feature_sum'] <= group_threshold] #Does the threshold work the same way since these are residual? Transformed?
                    high_residuals_df = residuals.loc[bin_df['feature_sum'] > group_threshold] # or is the residuals data the same and only the duration changed?

                    low_residuals_df = low_residuals_df["deviance"]
                    high_residuals_df = high_residuals_df["deviance"]

                    results = abs(ranksums(low_residuals_df, high_residuals_df))
                    residuals_score = results.statistic #test all thresholds by default in initial pop.
                    residuals_p_value = results.pvalue

            elif fitness_metric == 'aic':
                 #Create dataframes including instances from either strata-groups
                low_df = bin_df[bin_df['feature_sum'] <= group_threshold]
                high_df = bin_df[bin_df['feature_sum'] > group_threshold]

                low_outcome = low_df[outcome_label].to_list()
                high_outcome = high_df[outcome_label].to_list()
                low_censor = low_df[censor_label].to_list()
                high_censor =high_df[censor_label].to_list()
                self.count_bt = len(low_outcome)
                self.count_at = len(high_outcome)

                #Original code
                #def aic_score(self, covariate_matrix, bin_feature_matrix, label_name, duration_name,informative_cutoff, threshold):
                column_values = bin_df['feature_sum'].to_list()
                for r in range(0, len(column_values)):
                    if column_values[r] > 0: # Ryan - Is this still correct?
                        column_values[r] = 1
                data = covariate_df.copy()
                data['Bin'] = column_values
                data = data.loc[:, (data != data.iloc[0]).any()]
                cph = CoxPHFitter()
                cph.fit(data, outcome_label, event_col=censor_label)

                score = 0 - cph.AIC_partial_  #Ryan- can we just use - cph (no zero start?)
            else:
                print("Warning: fitness_metric not found.")

        elif outcome_type == 'class':
            print("Classification not yet implemented")
        else:
            print("Specified outcome_type not supported")

        return score,p_value,residuals_score,residuals_p_value
    
    
    def copy_parent(self,parent,iteration):
        #Attributes cloned from parent
        self.feature_list = copy.deepcopy(parent.feature_list) #sorting is for feature list comparison
        self.group_threshold = copy.deepcopy(parent.group_threshold)
        self.birth_iteration = iteration


    def uniform_crossover(self,other_offspring,crossover_prob,threshold_evolving,random):
        # Create list of feature names unique to one list or another
        set1 = set(self.feature_list)
        set2 = set(other_offspring.feature_list)
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1
        unique_features = list(sorted(unique_to_list1.union(unique_to_list2)))

        for feature in unique_features:
            if random.random() < crossover_prob:
                if feature in self.feature_list:
                    self.feature_list.remove(feature)
                    other_offspring.feature_list.append(feature)
                else:
                    other_offspring.feature_list.remove(feature)
                    self.feature_list.append(feature)

        # Apply crossover to thresholding if threshold_evolving
        if threshold_evolving:
            if random.random() < crossover_prob:
                temp = self.group_threshold
                self.group_threshold = other_offspring.group_threshold
                other_offspring.group_threshold = temp


    def mutation(self,mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random):
        self.feature_list = sorted(self.feature_list)

        if len(self.feature_list) == 0: #Initialize new bin if empty after crossover
            feature_count = random.randint(min_bin_size,max_bin_init_size)
            self.feature_list = random.sample(feature_names,feature_count)
            

        elif len(self.feature_list) == 1: # Addition and Swap Only (to avoid empy bins)
            for feature in self.feature_list:
                if random.random() < mutation_prob:
                    other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                    random_feature = random.choice(other_features)
                    if random.random() < 0.5: # Swap
                        self.feature_list.remove(feature)
                        self.feature_list.append(random_feature)
                    else: # Addition
                        self.feature_list.append(random_feature)
            # Enforce minimum bin size
            while len(self.feature_list) < min_bin_size: 
                other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                self.feature_list.append(random.choice(other_features))

        else: # Addition, Deletion, or Swap 
            mutate_options = ['A','D','S'] #Add, delete, swap
            for feature in self.feature_list:
                if random.random() < mutation_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'D' or len(feature_names) == len(self.feature_list): # Deletion - also if bin (i.e. feature_list) is at the maximum possible size
                        self.feature_list.remove(feature)
                    else:
                        other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                        random_feature = random.choice(other_features)
                        if mutate_type == 'S': # Swap
                            self.feature_list.remove(feature)
                            self.feature_list.append(random_feature)
                        elif mutate_type == 'A': # Addition
                            self.feature_list.append(random_feature)
            # Enforce minimum bin size
            while len(self.feature_list) < min_bin_size: 
                other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                self.feature_list.append(random.choice(other_features))
                        
        # Apply mutation to thresholding if threshold_evolving
        if threshold_evolving:
            if random.random() < mutation_prob:
                if min_thresh == max_thresh:
                    pass
                else:
                    thresh_list = [i for i in range(min_thresh,max_thresh+1)] #random.randint(min_thresh,max_thresh)
                    thresh_list.pop(thresh_list.index(self.group_threshold)) #[value for value in thresh_count if value != self.group_threshold] #pick a feature not already in the bin
                    random_thresh = random.choice(thresh_list)
                    self.group_threshold = random_thresh


    def calculate_pre_fitness(self,pareto_fitness,group_strata_min,penalty,fitness_metric,feature_names):

        if pareto_fitness: #Apply pareto-front-based multi-objective fitness
            print("Pareto-fitness has not yet been implemented")
            pass
        else:
            # Penalize fitness if group counts are beyond the minimum group strata parameter (Ryan Check below)
            self.group_strata_prop = min(self.count_bt/(self.count_bt+self.count_at),self.count_at/(self.count_bt+self.count_at))
            if self.group_strata_prop < group_strata_min: 
                self.pre_fitness = penalty * self.metric
            else:
                self.pre_fitness = self.metric
            # Residuals 
            if fitness_metric == 'residuals':
                self.pre_fitness = self.pre_fitness*self.residuals_score


    def random_bin(self,feature_names,min_bin_size,max_bin_init_size,random):
        """Takes an previously generated offspring bin (that already existed in the pop) and generates an new feature_list """
        # Initialize features in bin
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)


    def is_equivalent(self,other_bin):
        # Bin equivalence is based on 'feature_list' and 'group_threshold'
        equivalent = False
        if self.group_threshold == other_bin.group_threshold: #Same group threshold
            if sorted(self.feature_list) == sorted(other_bin.feature_list):
                equivalent = True
        return equivalent
    

    def bin_report(self):
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Metric Score:', 'p-value:' ,'Bin Size:', 'Group Ratio:', 
                    'Count At/Below Threshold:', 'Count Above Threshold:','Birth Iteration:','Residuals Score:','Residuals p-value']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold, self.fitness,self.pre_fitness,self.metric, self.p_value,
                                   self.bin_size, self.group_strata_prop, self.count_bt, self.count_at, self.birth_iteration,self.residuals_score,self.residuals_p_value]],columns=columns,index=None)
        return report_df
    

    def bin_short_report(self):
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Bin Size:', 'Group Ratio:','Birth Iteration:']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold, self.fitness,self.pre_fitness, self.bin_size, self.group_strata_prop,self.birth_iteration]],columns=columns,index=None).T
        return report_df
