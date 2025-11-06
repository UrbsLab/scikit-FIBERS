import numpy as np
import pandas as pd
import copy
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import ranksums
from scipy.integrate import simps  # for numerical integration
import matplotlib.pyplot as plt    # testing purposes
import sys                         # testing purposes

class BIN:
    def __init__(self, pareto):
        self.feature_list = [] # List of feature names (across which instance values are summed)
        self.group_threshold_list = [] # Thresholds for each group
        self.group_threshold = None # Threshold after which an instance is place in the 'above threshold' group - determines group strata of instances
        self.fitness = None # Bin fitness (higher fitness is better) - proportional to parent selection probability, and inversely proportional to deletion probability
        self.pre_fitness = None
        self.log_rank_score = None  # Log-rank Score
        self.log_rank_p_value = None # p-value of log rank test 
        self.bin_size = None # Number of features included in bin
        self.group_strata_prop = None # Proportion of instances in the smallest group (e.g. 0.5 --> equal number of instances in each group)
        self.count_bt = None # Instance count at/below threshold
        self.count_mt = None
        self.count_at = None # Instance count above threshold
        self.low_risk_area = None
        self.birth_iteration = None # Iteration where bin was introduced to population
        self.deletion_prop = None
        self.cluster = None
        self.residuals_score = None
        self.residuals_p_value = None
        self.HR = None
        self.HR_CI = None
        self.HR_p_value = None
        self.adj_HR = None
        self.adj_HR_CI = None
        self.is_merge = False
        self.adj_HR_p_value = None
        # added features
        self.pareto = pareto
        self.pairwise_scores = [] # pairwise logrank scores
        self.group_prop_list = [] # count / total for each risk group


    def update_deletion_prop(self,deletion_prop, cluster):
        self.deletion_prop = deletion_prop
        self.cluster = cluster


    def initialize_random(self,feature_names,min_bin_size,max_bin_init_size,group_thresh_list,multi_thresholding,min_thresh,max_thresh,iteration,random):
        self.birth_iteration = iteration
        # Initialize features in bin
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)
        if group_thresh_list is not None: # Defined group threshold
            self.group_threshold_list = group_thresh_list
        else: # Adaptive group threshold
            self.group_threshold_list = [random.randint(min_thresh, max_thresh), random.randint(min_thresh, max_thresh)]
            
            if self.group_threshold_list[0] == self.group_threshold_list[1] or not multi_thresholding:
                self.group_threshold_list = self.group_threshold_list[:-1]
                
            else:
                self.group_threshold_list.sort()
    

    def initialize_manual(self,feature_names,loaded_bin,loaded_thresh_list,group_thresh,min_thresh,max_thresh,birth_iteration):
        if birth_iteration is None:
            self.birth_iteration = 0
        else:
            self.birth_iteration = birth_iteration
        
        for feature in loaded_bin:
            #Initialize manual feature lists
            if feature in feature_names:
                self.feature_list.append(feature)
            else:
                print("Warning: feature ("+str(feature)+") not found in dataset for manual bin initialization")
        
        if not isinstance(loaded_thresh_list, list):
            loaded_thresh_list = [loaded_thresh_list]

        if group_thresh is not None and loaded_thresh_list != group_thresh:
            print("Warning: thresholds ("+str(loaded_thresh_list)+") is not equal to the specified group_thresh")
        elif min(loaded_thresh_list) < min_thresh or max(loaded_thresh_list) > max_thresh:
            print("Warning: thresholds ("+str(loaded_thresh_list)+") are outside of min and max thresh")
        else:
            self.group_threshold_list = loaded_thresh_list
        
        self.bin_size = len(self.feature_list)


    def evaluate(self,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,multi_thresholding,iterations,iteration,residuals,covariate_df, naive_survival_optimization):
        # Sum instance values across features specified in the bin
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})

        # Create evaluation dataframe including bin sum feature with 
        bin_df = pd.concat([bin_df,outcome_df,censor_df],axis=1)

        if (group_thresh == None and not threshold_evolving) or (group_thresh == None and iteration == iterations-1): #Adaptive thresholding activated (always applied on last iteration)
            # Select best threshold by evaluating all considered
            best_score = None
            thresh_score = 0
            if multi_thresholding:
                for low_thresh in range(min_thresh, max_thresh):
                    for high_thresh in range(low_thresh + 1, max_thresh + 1):
                        log_rank_score, p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at, low_risk_area = self.evaluate_for_thresholds(multi_thresholding,[low_thresh, high_thresh],bin_df,outcome_label,censor_label,
                                                                                                                                outcome_type,fitness_metric, log_rank_weighting,residuals,covariate_df, naive_survival_optimization)
                        if fitness_metric == 'log_rank' or fitness_metric == 'pareto':
                            thresh_score = log_rank_score

                        elif fitness_metric == 'residuals': 
                            thresh_score = residuals_score

                        elif fitness_metric == 'log_rank_residuals':
                            thresh_score = log_rank_score * residuals_score

                        if best_score is None or thresh_score > best_score:
                            self.log_rank_score = log_rank_score
                            self.log_rank_p_value = p_value
                            self.residuals_score = residuals_score
                            self.residuals_p_value = residuals_p_value
                            self.group_threshold_list = [low_thresh, high_thresh]
                            self.count_bt= count_bt
                            self.count_mt = count_mt
                            self.count_at = count_at
                            best_score = thresh_score
            
            for low_thresh in range(min_thresh, max_thresh + 1):
                log_rank_score, p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at,low_risk_area = self.evaluate_for_thresholds(multi_thresholding, [low_thresh],bin_df,outcome_label,censor_label,
                    outcome_type,fitness_metric, log_rank_weighting,residuals,covariate_df, naive_survival_optimization)
                if fitness_metric == 'log_rank' or fitness_metric == 'pareto':
                    thresh_score = log_rank_score

                elif fitness_metric == 'residuals': 
                    thresh_score = residuals_score

                elif fitness_metric == 'log_rank_residuals':
                    thresh_score = log_rank_score * residuals_score

                if naive_survival_optimization == True:
                    thresh_score *= low_risk_area

                if best_score == None or thresh_score > best_score:
                    self.log_rank_score = log_rank_score
                    self.log_rank_p_value = p_value
                    self.residuals_score = residuals_score
                    self.residuals_p_value = residuals_p_value
                    self.group_threshold_list = [low_thresh]
                    self.count_bt = count_bt
                    self.count_at = count_at
                    self.low_risk_area = low_risk_area
                    best_score = thresh_score

        else: #Use the given group threshold to evaluate the bin
            log_rank_score,p_value,residuals_score,residuals_p_value,count_bt, count_mt, count_at, low_risk_area = self.evaluate_for_thresholds(multi_thresholding,self.group_threshold_list,bin_df,outcome_label,censor_label,
                    outcome_type,fitness_metric, log_rank_weighting,residuals,covariate_df, naive_survival_optimization)
            self.log_rank_score = log_rank_score
            self.log_rank_p_value = p_value
            self.residuals_score = residuals_score
            self.residuals_p_value = residuals_p_value
            self.count_bt = count_bt
            self.count_mt = count_mt
            self.count_at = count_at
            self.low_risk_area = low_risk_area
        self.bin_size = len(self.feature_list)

    # New method added to evaluate a fixed bin - helpful for evaluation after bin injection
    def evaluate_fixed_bin(self,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,multi_thresholding,iterations,
                iteration,residuals,covariate_df, naive_survival_optimization, thresh_list):
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})
        bin_df = pd.concat([bin_df,outcome_df,censor_df],axis=1)

        num_thresh = len(thresh_list)
        if num_thresh == 2:
            # For 3-group bins
            log_rank_score, p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at, low_risk_area = self.evaluate_for_thresholds(
                multi_thresholding, [thresh_list[0],thresh_list[1]], bin_df, outcome_label, censor_label, 
                outcome_type, fitness_metric, log_rank_weighting, residuals, covariate_df, naive_survival_optimization)
            
            self.count_mt = count_mt
            
        else:
            # For 2-group bins
            log_rank_score, p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at,low_risk_area = self.evaluate_for_thresholds(
                multi_thresholding, [thresh_list[0]], bin_df, outcome_label, censor_label,
                outcome_type, fitness_metric, log_rank_weighting, residuals, covariate_df, naive_survival_optimization)
            
            self.count_mt = 0
            self.low_risk_area = low_risk_area
        
        # Assign values to bin
        self.log_rank_score = log_rank_score
        self.log_rank_p_value = p_value
        self.residuals_score = residuals_score
        self.residuals_p_value = residuals_p_value
        self.count_bt = count_bt
        self.count_at = count_at
        self.bin_size = len(self.feature_list)
        
        self.group_threshold_list = thresh_list.copy()


    def evaluate_for_thresholds(self,multi_thresholding,group_thresh_list,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,log_rank_weighting,residuals,covariate_df, naive_survival_optimization):
        low_thresh = None
        high_thresh = None
        num_thresh = len(group_thresh_list)
        if num_thresh == 2:
            low_thresh, high_thresh = group_thresh_list
        else:
            low_thresh = group_thresh_list[0]

        if multi_thresholding:
            if outcome_type == 'survival':
                residuals_score = None
                residuals_p_value = None
                log_rank_score = None
                p_value = None
                count_bt = None
                count_mt = None
                count_at = None

                if fitness_metric == 'log_rank' or fitness_metric == 'log_rank_residuals':
                    #Create dataframes including instances from either strata-groups
                    low_df = bin_df[bin_df['feature_sum'] <= low_thresh]
                    if num_thresh == 2:
                        mid_df = bin_df[(bin_df['feature_sum'] > low_thresh) & (bin_df['feature_sum'] <= high_thresh)]
                        mid_outcome = mid_df[outcome_label].to_list()
                        high_df = bin_df[bin_df['feature_sum'] > high_thresh]
                        mid_censor = mid_df[censor_label].to_list()
                        count_mt = len(mid_outcome)
                    else:
                        high_df = bin_df[bin_df['feature_sum'] > low_thresh]
                        count_mt = 0
                    
                    low_outcome = low_df[outcome_label].to_list()
                    high_outcome = high_df[outcome_label].to_list()
                    
                    low_censor = low_df[censor_label].to_list()
                    high_censor = high_df[censor_label].to_list()
                    
                    count_bt = len(low_outcome)
                    count_at = len(high_outcome)
                    
                    # FINDING AREA UNDER CURVE
                    kmf1 = KaplanMeierFitter()

                    if (low_df.size == 0 or high_df.size == 0):
                        log_rank_score = 0
                        p_value = None
                        residuals_score = 0
                        residuals_p_value = None
                        low_risk_area = 0
                        return log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at,low_risk_area
                    
                    # fit the model for 1st cohort
                    kmf1.fit(low_outcome, low_censor, label='At/Below Bin Threshold')

                    # Extracting the fitted survival function
                    scale_factor = 10       # scaling and rounding to prevent inaccuracy with np.trapz with high accuracy, small data
                    round_num = 5
                    survival_times = kmf1.survival_function_.index.values
                    survival_times = np.around(survival_times, round_num)
                    survival_probabilities = kmf1.survival_function_['At/Below Bin Threshold'].values
                    survival_probabilities_scaled = np.around(survival_probabilities, round_num) * scale_factor
                    
                    # Calculating the area under the fitted Kaplan-Meier curve using the trapezoidal rule
                    low_risk_area = np.trapz(survival_probabilities_scaled, survival_times)
                    low_risk_area = low_risk_area / scale_factor

                if fitness_metric == 'log_rank' or fitness_metric == 'log_rank_residuals' or fitness_metric == 'pareto':
                    #Create dataframes including instances from either strata-groups
                    try:
                        if num_thresh == 2:
                            # For 3-group bins, use pairwise log-rank tests
                            mid_outcome = mid_df[outcome_label].to_list()
                            mid_censor = mid_df[censor_label].to_list()
                            
                            # Check if any group is empty
                            if len(low_outcome) == 0 or len(mid_outcome) == 0 or len(high_outcome) == 0:
                                log_rank_score = 0
                                p_value = None
                                self.pairwise_scores = [0, 0, 0]
                            else:
                                # Calculate three pairwise log-rank tests: Low-High, Low-Medium, Medium-High
                                lh_results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                                lm_results = logrank_test(low_outcome, mid_outcome, event_observed_A=low_censor,event_observed_B=mid_censor,weightings=log_rank_weighting)
                                mh_results = logrank_test(mid_outcome, high_outcome, event_observed_A=mid_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                                
                                lh_stat = lh_results.test_statistic if lh_results is not None else 0
                                lm_stat = lm_results.test_statistic if lm_results is not None else 0
                                mh_stat = mh_results.test_statistic if mh_results is not None else 0

                                lh_pval = lh_results.p_value if lh_results is not None else None
                                lm_pval = lm_results.p_value if lm_results is not None else None
                                mh_pval = mh_results.p_value if mh_results is not None else None

                                # Use average of pairwise scores as overall log-rank score
                                avg_stat = (lh_stat + lm_stat + mh_stat) / 3
                                self.pairwise_scores = [round(stat, 3) for stat in [lh_stat, lm_stat, mh_stat]]
                                
                                log_rank_score = avg_stat
                                p_value = min(lh_pval, lm_pval, mh_pval) if all(p is not None for p in [lh_pval, lm_pval, mh_pval]) else None
                            
                            total = count_bt + count_mt + count_at
                            if total > 0:
                                self.group_prop_list = [count_bt/total, count_mt/total, count_at/total]
                            else:
                                self.group_prop_list = [0, 0, 0]
                        else:
                            # For 2-group bins, use standard log-rank test
                            results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                            log_rank_score = results.test_statistic
                            p_value = results.p_value
                            
                            total = count_bt + count_at
                            if total > 0:
                                self.group_prop_list = [count_bt/total, 0, count_at/total]
                            else:
                                self.group_prop_list = [0, 0, 0]
                            
                            self.pairwise_scores = []  # No pairwise scores for 2-group bins
                            
                        # Multivariate log-rank test approach
                        # This code can be uncommented and used instead of the pairwise approach above
                        # if num_thresh == 2:
                        #     mid_outcome = mid_df[outcome_label].to_list()
                        #     mid_censor = mid_df[censor_label].to_list()
                        #     count_mt = len(mid_outcome)
                        #     
                        #     combined_outcomes = low_outcome + mid_outcome + high_outcome # event_durations for all individuals
                        #     combined_groups = [0] * len(low_outcome) + [1] * len(mid_outcome)  + [2] * len(high_outcome) # Assign group 0 to low_outcome and group 1 to high_outcome (group labels for each individual)
                        #     combined_censors = low_censor + mid_censor + high_censor # event_observed (censoring) for all individuals
                        #     
                        # else:
                        #     count_mt = 0
                        #     combined_outcomes = low_outcome + high_outcome
                        #     combined_groups = [0] * len(low_outcome) + [1] * len(high_outcome)
                        #     combined_censors = low_censor + high_censor
                        #
                        # try:
                        #     # results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                        #     results = multivariate_logrank_test(combined_outcomes, combined_groups, event_observed=combined_censors, weightings=log_rank_weighting)
                        #     log_rank_score = results.test_statistic #test all thresholds by default in initial pop.
                        #     p_value = results.p_value
                        # except:
                        #     log_rank_score = 0
                        #     p_value = None
                            
                    except Exception as e:
                        log_rank_score = 0
                        p_value = None
                        self.pairwise_scores = [0, 0, 0] if num_thresh == 2 else []

                # Handle residuals evaluation if needed
                if fitness_metric == 'residuals' or fitness_metric == 'log_rank_residuals':
                    # Add residuals evaluation code here if needed
                    residuals_score = 0
                    residuals_p_value = None

                return log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,count_mt,count_at,low_risk_area
        else:
            if outcome_type == 'survival':
                residuals_score = None
                residuals_p_value = None
                log_rank_score = None
                p_value = None
                count_bt = None
                count_at = None
                low_risk_area = None

                low_df = bin_df[bin_df['feature_sum'] <= low_thresh]
                high_df = bin_df[bin_df['feature_sum'] > low_thresh]
                low_outcome = low_df[outcome_label].to_list()
                high_outcome = high_df[outcome_label].to_list()
                low_censor = low_df[censor_label].to_list()
                high_censor = high_df[censor_label].to_list()
                count_bt = len(low_outcome)
                count_at = len(high_outcome)

                # FINDING AREA UNDER CURVE
                kmf1 = KaplanMeierFitter()

                if (low_df.size == 0 or high_df.size == 0):
                    log_rank_score = 0
                    p_value = None
                    residuals_score = 0
                    residuals_p_value = None
                    low_risk_area = 0
                    return log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,0,count_at,low_risk_area
                
                # fit the model for 1st cohort
                kmf1.fit(low_outcome, low_censor, label='At/Below Bin Threshold')

                # Extracting the fitted survival function
                scale_factor = 10       # scaling and rounding to prevent inaccuracy with np.trapz with high accuracy, small data
                round_num = 5
                survival_times = kmf1.survival_function_.index.values
                survival_times = np.around(survival_times, round_num)
                survival_probabilities = kmf1.survival_function_['At/Below Bin Threshold'].values
                survival_probabilities_scaled = np.around(survival_probabilities, round_num) * scale_factor
                
                # Calculating the area under the fitted Kaplan-Meier curve using the trapezoidal rule
                low_risk_area = np.trapz(survival_probabilities_scaled, survival_times)
                low_risk_area = low_risk_area / scale_factor

                total = self.count_bt + self.count_at
                self.group_prop_list = [self.count_bt/total,0,self.count_at/total]

                if fitness_metric == 'log_rank' or fitness_metric == 'log_rank_residuals' or fitness_metric == 'pareto':
                    #Create dataframes including instances from either strata-groups
                    try:
                        results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                        log_rank_score = results.test_statistic #test all thresholds by default in initial pop.
                        p_value = results.p_value
                    except:
                        log_rank_score = 0
                        p_value = None

                if fitness_metric == 'residuals' or fitness_metric == 'log_rank_residuals': # In addition to log_rank, calculate residuals differences between groups
                    low_residuals_df = residuals.loc[bin_df['feature_sum'] <= low_thresh] #Does the threshold work the same way since these are residual? Transformed?
                    high_residuals_df = residuals.loc[bin_df['feature_sum'] > low_thresh] # or is the residuals data the same and only the duration changed?
                    low_residuals_df = low_residuals_df["deviance"]
                    high_residuals_df = high_residuals_df["deviance"]
                    count_bt = len(low_residuals_df)
                    count_at = len(high_residuals_df)
                    if len(low_residuals_df) == 0 or len(high_residuals_df) == 0:
                        residuals_score = 0
                        residuals_p_value = None
                    else:
                        try:
                            results = ranksums(low_residuals_df, high_residuals_df, alternative='less')
                            # ideal high risk group: we want positive residuals (death occurred earlier than expected)
                            # ideal low risk group: we want negative residuals (death occurred later than expected)
                            residuals_score = -results.statistic 
                            residuals_p_value = results.pvalue
                        except:
                            residuals_score = 0
                            residuals_p_value = None

            elif outcome_type == 'class':
                print("Classification not yet implemented")
            else:
                print("Specified outcome_type not supported")

            return log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,0, count_at, low_risk_area
    
    
    def copy_parent(self,parent,iteration):
        #Attributes cloned from parent
        self.feature_list = copy.deepcopy(parent.feature_list) #sorting is for feature list comparison
        self.group_threshold_list = copy.deepcopy(parent.group_threshold_list)
        self.birth_iteration = iteration
        self.pairwise_scores = copy.deepcopy(parent.pairwise_scores)


    def uniform_crossover(self,other_offspring,crossover_prob,threshold_evolving,multi_thresholding,max_thresh,random):
        # Create list of feature names unique to one list or another
        set1 = set(self.feature_list)
        set2 = set(other_offspring.feature_list)
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1
        unique_features = list(sorted(unique_to_list1.union(unique_to_list2)))

        swap_probability = 0.5
        for feature in unique_features:
            if random.random() < swap_probability:
                if feature in self.feature_list:
                    self.feature_list.remove(feature)
                    other_offspring.feature_list.append(feature)
                else:
                    other_offspring.feature_list.remove(feature)
                    self.feature_list.append(feature)

        def crossover_threshold(threshold_list1, threshold_list2, swap_probability, max_thresh, random):
            set1_th = set(threshold_list1)
            set2_th = set(threshold_list2)
            unique_to_list1_th = set1_th - set2_th
            unique_to_list2_th = set2_th - set1_th

            # Crossover unique thresholds based on probability
            if random.random() < swap_probability:
                for th in unique_to_list1_th:
                    if len(threshold_list1) > 1:
                        threshold_list1.remove(th)
                        threshold_list2.append(th)
                for th in unique_to_list2_th:
                    if len(threshold_list2) > 1:
                        threshold_list2.remove(th)
                        threshold_list1.append(th)

            # Ensure both threshold lists have at least 1 threshold
            if len(threshold_list1) < 1:
                threshold_list1.append(min(threshold_list2))
                threshold_list2 = list(set(threshold_list2) - set(threshold_list1))
            
            if len(threshold_list2) < 1:
                threshold_list2.append(min(threshold_list1))
                threshold_list1 = list(set(threshold_list1) - set(threshold_list2))

            # Ensure no threshold list has more than 2 thresholds
            threshold_list1 = threshold_list1[:2]
            threshold_list2 = threshold_list2[:2]

            # Ensure thresholds are within bounds
            threshold_list1 = [min(max(0, th), max_thresh) for th in threshold_list1]
            threshold_list2 = [min(max(0, th), max_thresh) for th in threshold_list2]

            return sorted(threshold_list1), sorted(threshold_list2)

        if threshold_evolving:
            if multi_thresholding:
                self.group_threshold_list, other_offspring.group_threshold_list = crossover_threshold(
                    self.group_threshold_list, other_offspring.group_threshold_list, swap_probability, max_thresh, random)
            else:
                if random.random() < swap_probability:
                    temp = self.group_threshold_list[0]
                    self.group_threshold_list[0] = other_offspring.group_threshold_list[0]
                    other_offspring.group_threshold_list[0] = temp


    
    def mutation(self,mutation_prob,feature_names,min_bin_size,max_bin_size,max_bin_init_size,threshold_evolving,multi_thresholding,min_thresh,max_thresh,random):
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
                        if len(self.feature_list) < max_bin_size:
                            self.feature_list.append(random_feature)
            # Enforce minimum bin size
            while len(self.feature_list) < min_bin_size: 
                other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                self.feature_list.append(random.choice(other_features))

        else: # Addition, Deletion, or Swap
            mutate_options = ['A','D','S'] #Add, delete, swap
            original_feature_list = copy.deepcopy(self.feature_list)
            for feature in original_feature_list:
                if random.random() < mutation_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'D' or len(feature_names) == len(self.feature_list): # Deletion - also if bin (i.e. feature_list) is at the maximum possible size
                        self.feature_list.remove(feature)
                    else:
                        other_features = [value for value in feature_names if value not in original_feature_list] #pick a feature not already in the bin
                        random_feature = random.choice(other_features)
                        if mutate_type == 'S': # Swap
                            self.feature_list.remove(feature)
                            self.feature_list.append(random_feature)
                            original_feature_list.append(random_feature)
                        elif mutate_type == 'A': # Addition
                            self.feature_list.append(random_feature)
                            original_feature_list.append(random_feature)
            # Enforce minimum bin size
            while len(self.feature_list) < min_bin_size:
                other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                self.feature_list.append(random.choice(other_features))
            # Enforce maximum bin size
            while len(self.feature_list) > max_bin_size:
                self.feature_list.remove(random.choice(self.feature_list))

        if threshold_evolving:
            th_list = [i for i in range(min_thresh, max_thresh+1)]
            th_list.pop(th_list.index(self.group_threshold_list[0]))    
            if len(self.group_threshold_list) == 1: # Addition and Swap Only (to avoid empy bins)
                mutate_options = ['A', 'S'] # Add, swap
                if random.random() < mutation_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'A' and multi_thresholding: # Add (only if multi_thresholding)
                        self.group_threshold_list.append(random.choice(th_list))
                    else: # Swap
                        self.group_threshold_list[0] = random.choice(th_list)
            elif multi_thresholding: # Delete, swap
                th_list.pop(th_list.index(self.group_threshold_list[1]))
                mutate_options = ['D', 'S'] # Delete, swap
                if random.random() < mutation_prob:
                    mutate_type = random.choice(mutate_options)
                    if mutate_type == 'D': # Delete
                        if random.random() < 0.5:
                            self.group_threshold_list = self.group_threshold_list[:-1]
                        else:
                            self.group_threshold_list = self.group_threshold_list[-1:]
                    else: # Swap
                        self.group_threshold_list[0] = random.choice(th_list)
                        if random.random() < mutation_prob:
                            self.group_threshold_list[1] = random.choice(th_list)
            self.group_threshold_list.sort()


    def merge(self,other_parent,max_bin_size,threshold_evolving,multi_thresholding,max_thresh,random):
        # Merge feature lists of two parents
        # Create list of feature names unique to one list or another
        set1 = set(self.feature_list)
        set2 = set(other_parent.feature_list)
        #unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1
        #unique_features = list(sorted(unique_to_list1.union(unique_to_list2)))
        self.feature_list = self.feature_list + list(unique_to_list2) 
        #self.feature_list = unique_features
        #Enforce maximum bin size
        while len(self.feature_list) > max_bin_size:
            self.feature_list.remove(random.choice(self.feature_list))

        # Merge threshold lists of two parents
        if threshold_evolving:
            if multi_thresholding:
                set1_th = set(self.group_threshold_list)
                set2_th = set(other_parent.group_threshold_list)
                unique_to_list2_th = set2_th - set1_th
                unique_to_list1_th = set1_th - set2_th

                # Combine thresholds
                merged_thresholds = sorted(self.group_threshold_list + list(unique_to_list2_th))
                
                # Determine whether to keep 1 or 2 thresholds
                if len(merged_thresholds) > 1 and random.random() < 0.5:
                    self.group_threshold_list = random.sample(merged_thresholds, 2)
                else:
                    self.group_threshold_list = [random.choice(merged_thresholds)]
                
                self.group_threshold_list.sort()

                # Ensure that only 2 thresholds exist
                while len(self.group_threshold_list) > 2:
                    self.group_threshold_list.remove(random.choice(self.group_threshold_list))

                if unique_to_list2_th is None and unique_to_list1_th is None:
                    self.group_threshold_list = list(np.asarray(self.group_threshold_list) + 1)
                # Enforce maximum threshold
                if len(self.group_threshold_list) > 1 and self.group_threshold_list[1] > max_thresh:
                    self.group_threshold_list[1] = max_thresh
                    if self.group_threshold_list[0] == self.group_threshold_list[1]:
                        self.group_threshold_list[0] = max(0, self.group_threshold_list[1] - 1)
            else:
                if self.group_threshold_list[0] == 0 or other_parent.group_threshold_list[0] == 0:
                    self.group_threshold_list[0] += 1
                self.group_threshold_list[0] += other_parent.group_threshold_list[0]
                #Enforce maximum group threshold
                if self.group_threshold_list[0] > max_thresh:
                    self.group_threshold_list[0] = max_thresh


    def calculate_pre_fitness(self,group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization):
        # adjust minimum group strata parameter based on number of risk curves in bin
        num_curves = len(self.group_threshold_list) + 1
        group_strata_min /= num_curves
        # Penalize fitness if group counts are beyond the minimum group strata parameter (Ryan Check below)
        total = self.count_bt + self.count_at
        if num_curves == 3:
            total += self.count_mt
            self.group_strata_prop = min(self.count_bt/total,self.count_mt/total,self.count_at/total)
        else:
            self.group_strata_prop = min(self.count_bt/total,self.count_at/total)

        if self.group_strata_prop == 0.0:
            self.pre_fitness = 0.0
        else:
            if fitness_metric == 'log_rank':
                if self.group_strata_prop < group_strata_min:
                    self.pre_fitness = (1-penalty) * self.log_rank_score
                else:
                    self.pre_fitness = self.log_rank_score

            if fitness_metric == 'residuals':
                if self.group_strata_prop < group_strata_min: 
                    self.pre_fitness = (1-penalty) * self.residuals_score
                else:
                    self.pre_fitness = self.residuals_score

            if fitness_metric == 'log_rank_residuals':
                if self.group_strata_prop < group_strata_min: 
                    self.pre_fitness = (1-penalty) * self.log_rank_score * self.residuals_score
                else:
                    self.pre_fitness = self.log_rank_score * self.residuals_score
            
            if fitness_metric == 'pareto':
                self.pre_fitness = self.pareto.get_pareto_fitness(self.log_rank_score, self.low_risk_area, None)
            
            if naive_survival_optimization == True:
                self.pre_fitness *= self.low_risk_area


    def random_bin(self,feature_names,min_bin_size,max_bin_init_size,random):
        """Takes an previously generated offspring bin (that already existed in the pop) and generates an new feature_list """
        # Initialize features in bin
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)


    def is_equivalent(self,other_bin):
        # Bin equivalence is based on 'feature_list' and 'group_threshold'
        equivalent = False
        if set(self.group_threshold_list) == set(other_bin.group_threshold_list): #Same group threshold
            if sorted(self.feature_list) == sorted(other_bin.feature_list):
                equivalent = True
        return equivalent
    

    def bin_report(self):
        pd.set_option('display.max_colwidth', None) # prevent truncation of dataframe
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Log-Rank Score:', 'Log-Rank p-value:' ,'Bin Size:', 'Group Ratio:', 
                    'Count At/Below Threshold:', 'Count Above Threshold:','Birth Iteration:','Residuals Score:','Residuals p-value', 'Area', 'Pairwise Scores', 'Strata Prop List']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold_list, self.fitness,self.pre_fitness,self.log_rank_score, self.log_rank_p_value,
                                   self.bin_size, self.group_strata_prop, self.count_bt, self.count_at, self.birth_iteration,self.residuals_score,
                                   self.residuals_p_value,self.low_risk_area,self.pairwise_scores, self.group_prop_list]],columns=columns,index=None)
        return report_df
    
    def get_bin_composition(self, df,feature_names, predictive_features, threshold):
        predictive_feature_list = []
        for x in range(predictive_features):
            name = 'P_' + str(x + 1)
            predictive_feature_list.append(name)

        outcome_label = "Duration"
        feature_df = df.loc[:,feature_names]
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        predictive_feature_sums = feature_df[predictive_feature_list].sum(axis=1)

        sum_df = pd.DataFrame({'feature_sum':feature_sums})
        real_sum_df = pd.DataFrame({'real_feature_sum': predictive_feature_sums})
    
        bin_df = pd.concat([df, sum_df, real_sum_df],axis=1)

        
        low_df = bin_df[bin_df['feature_sum'] <= self.group_threshold]
        real_low_df = bin_df[bin_df['real_feature_sum'] <= threshold]
        high_df = bin_df[bin_df['feature_sum'] > self.group_threshold]
        real_high_df = bin_df[bin_df['real_feature_sum'] > threshold]


        real_low_african_american = real_low_df[real_low_df['AFRICAN-AMERICAN'] == 1]
        real_low_aa_ct = len(real_low_african_american[outcome_label].to_list())
        low_african_american = low_df[low_df['AFRICAN-AMERICAN'] == 1]
        bin_low_aa_ct = len(low_african_american[outcome_label].to_list())
        
        real_high_african_american = real_high_df[real_high_df['AFRICAN-AMERICAN'] == 1]
        real_high_aa_ct = len(real_high_african_american[outcome_label].to_list())
        high_african_american = high_df[high_df['AFRICAN-AMERICAN'] == 1]
        bin_high_aa_ct = len(high_african_american[outcome_label].to_list())

        real_low_white = real_low_df[real_low_df['WHITE'] == 1]
        real_low_white_ct = len(real_low_white[outcome_label].to_list())
        low_white = low_df[low_df['WHITE'] == 1]
        bin_low_white_ct = len(low_white[outcome_label].to_list())

        real_high_white = real_high_df[real_high_df['WHITE'] == 1]
        real_high_white_ct = len(real_high_white[outcome_label].to_list())
        high_white = high_df[high_df['WHITE'] == 1]
        bin_high_white_ct = len(high_white[outcome_label].to_list())

        real_low_hispanic = real_low_df[real_low_df['HISPANIC'] == 1]
        real_low_hispanic_ct = len(real_low_hispanic[outcome_label].to_list())
        low_hispanic = low_df[low_df['HISPANIC'] == 1]
        bin_low_hispanic_ct = len(low_hispanic[outcome_label].to_list())

        real_high_hispanic = real_high_df[real_high_df['HISPANIC'] == 1]
        real_high_hispanic_ct = len(real_high_hispanic[outcome_label].to_list())
        high_hispanic = high_df[high_df['HISPANIC'] == 1]
        bin_high_hispanic_ct = len(high_hispanic[outcome_label].to_list())

        real_low_asian = real_low_df[real_low_df['ASIAN'] == 1]
        real_low_asian_ct = len(real_low_asian[outcome_label].to_list())
        low_asian = low_df[low_df['ASIAN'] == 1]
        bin_low_asian_ct = len(low_asian[outcome_label].to_list())

        real_high_asian = real_high_df[real_high_df['ASIAN'] == 1]
        real_high_asian_ct = len(real_high_asian[outcome_label].to_list())
        high_asian = high_df[high_df['ASIAN'] == 1]
        bin_high_asian_ct = len(high_asian[outcome_label].to_list())

        real_low_other = real_low_df[real_low_df['OTHER'] == 1]
        real_low_other_ct = len(real_low_other[outcome_label].to_list())
        low_other = low_df[low_df['OTHER'] == 1]
        bin_low_other_ct = len(low_other[outcome_label].to_list())

        real_high_other = real_high_df[real_high_df['OTHER'] == 1]
        real_high_other_ct = len(real_high_other[outcome_label].to_list())
        high_other = high_df[high_df['OTHER'] == 1]
        bin_high_other_ct = len(high_other[outcome_label].to_list())

        real_low_mdmr = real_low_df[real_low_df['MDMR'] == 1]
        real_low_mdmr_ct = len(real_low_mdmr[outcome_label].to_list())
        low_mdmr = low_df[low_df['MDMR'] == 1]
        bin_low_mdmr_ct = len(low_mdmr[outcome_label].to_list())

        real_high_mdmr = real_high_df[real_high_df['MDMR'] == 1]
        real_high_mdmr_ct = len(real_high_mdmr[outcome_label].to_list())
        high_mdmr = high_df[high_df['MDMR'] == 1]
        bin_high_mdmr_ct = len(high_mdmr[outcome_label].to_list())

        real_low_fdfr = real_low_df[real_low_df['FDFR'] == 1]
        real_low_fdfr_ct = len(real_low_fdfr[outcome_label].to_list())
        low_fdfr = low_df[low_df['FDFR'] == 1]
        bin_low_fdfr_ct = len(low_fdfr[outcome_label].to_list())

        real_high_fdfr = real_high_df[real_high_df['FDFR'] == 1]
        real_high_fdfr_ct = len(real_high_fdfr[outcome_label].to_list())
        high_fdfr = high_df[high_df['FDFR'] == 1]
        bin_high_fdfr_ct = len(high_fdfr[outcome_label].to_list())

        real_low_fdmr = real_low_df[real_low_df['FDMR'] == 1]
        real_low_fdmr_ct = len(real_low_fdmr[outcome_label].to_list())
        low_fdmr = low_df[low_df['FDMR'] == 1]
        bin_low_fdmr_ct = len(low_fdmr[outcome_label].to_list())

        real_high_fdmr = real_high_df[real_high_df['FDMR'] == 1]
        real_high_fdmr_ct = len(real_high_fdmr[outcome_label].to_list())
        high_fdmr = high_df[high_df['FDMR'] == 1]
        bin_high_fdmr_ct = len(high_fdmr[outcome_label].to_list())

        real_low_mdfr = real_low_df[real_low_df['MDFR'] == 1]
        real_low_mdfr_ct = len(real_low_mdfr[outcome_label].to_list())
        low_mdfr = low_df[low_df['MDFR'] == 1]
        bin_low_mdfr_ct = len(low_mdfr[outcome_label].to_list())

        real_high_mdfr = real_high_df[real_high_df['MDFR'] == 1]
        real_high_mdfr_ct = len(real_high_mdfr[outcome_label].to_list())
        high_mdfr = high_df[high_df['MDFR'] == 1]
        bin_high_mdfr_ct = len(high_mdfr[outcome_label].to_list())

        return real_low_aa_ct, bin_low_aa_ct, real_high_aa_ct, bin_high_aa_ct, \
    real_low_white_ct, bin_low_white_ct, real_high_white_ct, bin_high_white_ct, \
    real_low_hispanic_ct, bin_low_hispanic_ct, real_high_hispanic_ct, bin_high_hispanic_ct, \
    real_low_asian_ct, bin_low_asian_ct, real_high_asian_ct, bin_high_asian_ct, \
    real_low_other_ct, bin_low_other_ct, real_high_other_ct, bin_high_other_ct, \
    real_low_mdmr_ct, bin_low_mdmr_ct, real_high_mdmr_ct, bin_high_mdmr_ct, \
    real_low_fdfr_ct, bin_low_fdfr_ct, real_high_fdfr_ct, bin_high_fdfr_ct, \
    real_low_fdmr_ct, bin_low_fdmr_ct, real_high_fdmr_ct, bin_high_fdmr_ct, \
    real_low_mdfr_ct, bin_low_mdfr_ct, real_high_mdfr_ct, bin_high_mdfr_ct


    def bin_short_report(self):
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Bin Size:', 'Group Ratio:','Birth Iteration:']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold_list, self.fitness,self.pre_fitness, self.bin_size, self.group_strata_prop,self.birth_iteration]],columns=columns,index=None).T
        return report_df
