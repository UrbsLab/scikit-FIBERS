import numpy as np
import pandas as pd
import copy
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import restricted_mean_survival_time as lifelines_restricted_mean_survival_time
from scipy.stats import ranksums

class BIN:
    def __init__(self):
        self.feature_list = [] # List of feature names (across which instance values are summed)
        self.group_threshold = None # Threshold after which an instance is placed in the 'above threshold' or high-risk group - determines group strata of instances
        self.fitness = None # Global bin fitness (higher fitness is better) - proportional to parent selection probability, and inversely proportional to deletion probability
        self.pre_fitness = None # Core fitness metric (either based on log-rank score, residuals score, or the product of both)
        self.log_rank_score = None  # Log-rank Score
        self.log_rank_p_value = None # p-value of log rank test 
        self.bin_size = None # Number of features included in bin
        self.group_strata_prop = None # Proportion of instances in the smallest group (e.g. 0.5 --> equal number of instances in each group)
        self.count_bt = None # Instance count at/below threshold
        self.count_at = None # Instance count above threshold
        self.birth_iteration = None # Iteration where bin was introduced to population
        self.deletion_prop = None #Last assigned probability of deletion
        self.cluster = None #Last assigned bin diversity cluster
        self.residuals_score = None # Wilcoxon Rank Sum Test score comparing deviance residuals of two risk groups
        self.residuals_p_value = None # P-value for the Wilcoxon Rank Sum Test compraring deviance residuals of two risk groups
        self.HR = None # The unadjusted hazard ratio of the bin calculated with CoxPHFitter after algorithm training
        self.HR_CI = None #The associated unadjusted hazard ratio confidence interval calculated with CoxPHFitter after algorithm training
        self.HR_p_value = None # The associated unadjusted hazard ratio p-value calculated with CoxPHFitter after algorithm training
        self.adj_HR = None #The adjusted hazard ratio of the bin calculated with CoxPHFitter after algorithm training
        self.adj_HR_CI = None #The associated adjusted hazard ratio confidence interval calculated with CoxPHFitter after algorithm training
        self.adj_HR_p_value = None # The associated adjusted hazard ratio p-value calculated with CoxPHFitter after algorithm training
        self.used_group_strata_fallback = False


    def update_deletion_prop(self,deletion_prop, cluster):
        self.deletion_prop = deletion_prop
        self.cluster = cluster


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
    

    def initialize_manual(self,feature_names,loaded_bin,loaded_thresh,group_thresh,min_thresh,max_thresh,birth_iteration):
        if birth_iteration == None:
            self.birth_iteration = 0
        else:
            self.birth_iteration = birth_iteration
        for feature in loaded_bin:
            #Initialize manual feature lists
            if feature in feature_names:
                self.feature_list.append(feature)
            else:
                print("Warning: feature ("+str(feature)+") not found in dataset for manual bin initialization")
            if group_thresh != None and loaded_thresh != group_thresh:
                print("Warning: threshold ("+str(loaded_thresh)+") is not equal to the specified group_thresh")
            elif loaded_thresh < min_thresh or loaded_thresh > max_thresh:
                print("Warning: threshold ("+str(loaded_thresh)+") is outside of min and max thresh")
            else:
                self.group_threshold = loaded_thresh
        self.bin_size = len(self.feature_list)


    def evaluate(self,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,covariate_df,
                 desired_bin_effect,group_strata_min):
        self.used_group_strata_fallback = False
        # Sum instance values across features specified in the bin
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})

        # Create evaluation dataframe including bin sum feature with 
        bin_df = pd.concat([bin_df,outcome_df,censor_df],axis=1)

        if (group_thresh == None and not threshold_evolving) or (group_thresh == None and iteration == iterations-1): #Adaptive thresholding activated (always applied on last iteration)
            # Select best threshold by evaluating all considered
            best_score = None
            raw_best_score = None
            fallback_result = None
            fallback_threshold = None
            # Track the best threshold that matches direction, even if it misses group_strata_min.
            directional_fallback_result = None
            directional_fallback_threshold = None
            directional_fallback_score = None
            # Track the best threshold that matches group_strata_min, even if it misses direction.
            strata_valid_fallback_result = None
            strata_valid_fallback_threshold = None
            strata_valid_fallback_score = None
            found_valid_threshold = False
            for threshold in range(min_thresh, max_thresh + 1):
                if desired_bin_effect == "protective" or desired_bin_effect == "high_risk":
                    # Score each threshold twice: once with raw/default behavior for ranking,
                    # and once with directional gating for the actual stored threshold result.
                    raw_result = self.evaluate_for_threshold(
                        threshold,
                        bin_df,
                        outcome_label,
                        censor_label,
                        outcome_type,
                        fitness_metric,
                        log_rank_weighting,
                        residuals,
                        covariate_df,
                        "default",
                    )
                    directional_result = self.evaluate_for_threshold(
                        threshold,
                        bin_df,
                        outcome_label,
                        censor_label,
                        outcome_type,
                        fitness_metric,
                        log_rank_weighting,
                        residuals,
                        covariate_df,
                        desired_bin_effect,
                    )

                    raw_thresh_score = self.get_threshold_score(fitness_metric, raw_result[0], raw_result[2])
                    directional_thresh_score = self.get_threshold_score(fitness_metric, directional_result[0], directional_result[2])
                    threshold_matches_effect = directional_result[6]
                    threshold_matches_group_strata = self.meets_group_strata_min(
                        directional_result[4],
                        directional_result[5],
                        group_strata_min,
                    )

                    if raw_best_score == None or raw_thresh_score > raw_best_score:
                        raw_best_score = raw_thresh_score
                        fallback_result = directional_result
                        fallback_threshold = threshold

                    if threshold_matches_effect and (directional_fallback_score == None or raw_thresh_score > directional_fallback_score):
                        directional_fallback_score = raw_thresh_score
                        directional_fallback_result = directional_result
                        directional_fallback_threshold = threshold

                    if threshold_matches_group_strata and (strata_valid_fallback_score == None or raw_thresh_score > strata_valid_fallback_score):
                        strata_valid_fallback_score = raw_thresh_score
                        strata_valid_fallback_result = directional_result
                        strata_valid_fallback_threshold = threshold

                    # Fully valid thresholds must satisfy both the requested direction and group_strata_min.
                    if threshold_matches_effect and threshold_matches_group_strata and (best_score == None or directional_thresh_score > best_score):
                        self.assign_eval_result(threshold, directional_result)
                        best_score = directional_thresh_score
                        found_valid_threshold = True
                else:
                    result = self.evaluate_for_threshold(threshold,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,
                            log_rank_weighting,residuals,covariate_df,desired_bin_effect)
                    thresh_score = self.get_threshold_score(fitness_metric, result[0], result[2])

                    if best_score == None or thresh_score > best_score:
                        self.assign_eval_result(threshold, result)
                        best_score = thresh_score

            if (desired_bin_effect == "protective" or desired_bin_effect == "high_risk") and not found_valid_threshold and strata_valid_fallback_result is not None:
                # If nothing satisfies both direction and group_strata_min, prefer a direction-correct
                # threshold and let pre-fitness apply an extra penalty for this fallback case.
                if directional_fallback_result is not None:
                    self.assign_eval_result(directional_fallback_threshold, directional_fallback_result)
                    self.used_group_strata_fallback = True
                else:
                    self.assign_eval_result(strata_valid_fallback_threshold, strata_valid_fallback_result)
            elif (desired_bin_effect == "protective" or desired_bin_effect == "high_risk") and not found_valid_threshold and fallback_result is not None:
                # Last resort: keep the best direction-correct threshold if one exists, otherwise keep
                # the raw-best placeholder threshold. Wrong-direction fallbacks already carry zero score.
                if directional_fallback_result is not None:
                    self.assign_eval_result(directional_fallback_threshold, directional_fallback_result)
                    self.used_group_strata_fallback = True
                else:
                    self.assign_eval_result(fallback_threshold, fallback_result)

        else: #Use the given group threshold to evaluate the bin
            log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,count_at,_ = self.evaluate_for_threshold(self.group_threshold,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,
                        log_rank_weighting,residuals,covariate_df,desired_bin_effect)
            self.log_rank_score = log_rank_score
            self.log_rank_p_value = p_value
            self.residuals_score = residuals_score
            self.residuals_p_value = residuals_p_value
            self.count_bt = count_bt
            self.count_at = count_at
        self.bin_size = len(self.feature_list)


    def evaluate_for_threshold(self,threshold,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,log_rank_weighting,residuals,covariate_df,
                               desired_bin_effect):
        # Apply selected evaluation strategy/metric(s)
        if outcome_type == 'survival':
            residuals_score = None
            residuals_p_value = None
            log_rank_score = None
            p_value = None
            count_bt = None
            count_at = None

            low_df = bin_df[bin_df['feature_sum'] <= threshold]
            high_df = bin_df[bin_df['feature_sum'] > threshold]
            low_outcome = low_df[outcome_label].to_list()
            high_outcome = high_df[outcome_label].to_list()
            low_censor = low_df[censor_label].to_list()
            high_censor = high_df[censor_label].to_list()
            count_bt = len(low_outcome)
            count_at = len(high_outcome)

            directionally_valid_bin = True
            if desired_bin_effect == "protective" or desired_bin_effect == "high_risk":
                try:
                    # Directional-only rule using censoring-aware RMST.
                    time_point = min(max(low_outcome), max(high_outcome))
                    low_rmst = self.restricted_mean_survival_time(low_outcome, low_censor, time_point)
                    high_rmst = self.restricted_mean_survival_time(high_outcome, high_censor, time_point)
                    if desired_bin_effect == "protective":
                        if high_rmst <= low_rmst:
                            directionally_valid_bin = False
                    elif desired_bin_effect == "high_risk":
                        if high_rmst >= low_rmst:
                            directionally_valid_bin = False
                except:
                    directionally_valid_bin = False

            if fitness_metric == 'log_rank' or fitness_metric == 'log_rank_residuals':
                if (desired_bin_effect == "protective" or desired_bin_effect == "high_risk") and not directionally_valid_bin:
                    log_rank_score = 0
                    p_value = None
                else:
                    try:
                        results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor,weightings=log_rank_weighting)
                        log_rank_score = results.test_statistic #test all thresholds by default in initial pop.
                        p_value = results.p_value
                    except:
                        log_rank_score = 0
                        p_value = None

            if fitness_metric == 'residuals' or fitness_metric == 'log_rank_residuals': # In addition to log_rank, calculate residuals differences between groups
                # if desired_bin_effect == "permissive":
                #     high_residuals_df = residuals.loc[bin_df['feature_sum'] <= threshold] #Does the threshold work the same way since these are residual? Transformed?
                #     low_residuals_df = residuals.loc[bin_df['feature_sum'] > threshold] # or is the residuals data the same and only the duration changed?
                #     low_residuals_df = low_residuals_df["deviance"]
                #     high_residuals_df = high_residuals_df["deviance"]
                # else:
                #     low_residuals_df = residuals.loc[bin_df['feature_sum'] <= threshold] #Does the threshold work the same way since these are residual? Transformed?
                #     high_residuals_df = residuals.loc[bin_df['feature_sum'] > threshold] # or is the residuals data the same and only the duration changed?
                #     low_residuals_df = low_residuals_df["deviance"]
                #     high_residuals_df = high_residuals_df["deviance"]
                
                mask = bin_df['feature_sum'] <= threshold  # choose ONE convention and stick to it
                if desired_bin_effect == "permissive":
                    high_residuals_df = residuals.loc[mask, "deviance"]
                    low_residuals_df  = residuals.loc[~mask, "deviance"]
                else:
                    low_residuals_df  = residuals.loc[mask, "deviance"]
                    high_residuals_df = residuals.loc[~mask, "deviance"]
                count_bt = len(low_residuals_df)
                count_at = len(high_residuals_df)
                if (desired_bin_effect == "protective" or desired_bin_effect == "high_risk") and not directionally_valid_bin:
                    residuals_score = 0
                    residuals_p_value = None
                elif len(low_residuals_df) == 0 or len(high_residuals_df) == 0:
                    residuals_score = 0
                    residuals_p_value = None
                else:
                    try:
                        results = ranksums(low_residuals_df, high_residuals_df)
                        residuals_score = abs(results.statistic) 
                        residuals_p_value = results.pvalue
                    except:
                        residuals_score = 0
                        residuals_p_value = None

        elif outcome_type == 'class':
            print("Classification not yet implemented")
            raise NotImplementedError
        else:
            print("Specified outcome_type not supported")
            raise Exception("Specified outcome_type not supported")

        return log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,count_at,directionally_valid_bin


    def get_threshold_score(self,fitness_metric,log_rank_score,residuals_score):
        if fitness_metric == 'log_rank':
            return log_rank_score
        if fitness_metric == 'residuals':
            return residuals_score
        if fitness_metric == 'log_rank_residuals':
            return log_rank_score * residuals_score
        return 0


    def get_group_strata_prop(self,count_bt,count_at):
        total_count = count_bt + count_at
        if total_count == 0:
            return 0.0
        return min(count_bt / total_count, count_at / total_count)


    def meets_group_strata_min(self,count_bt,count_at,group_strata_min):
        return self.get_group_strata_prop(count_bt, count_at) >= group_strata_min


    def assign_eval_result(self,threshold,result):
        log_rank_score,p_value,residuals_score,residuals_p_value,count_bt,count_at,_ = result
        self.log_rank_score = log_rank_score
        self.log_rank_p_value = p_value
        self.residuals_score = residuals_score
        self.residuals_p_value = residuals_p_value
        self.group_threshold = threshold
        self.count_bt = count_bt
        self.count_at = count_at

    def km_survival_at_time(self,outcome,censor,time_point):
        kmf = KaplanMeierFitter()
        kmf.fit(outcome,event_observed=censor)
        return float(kmf.survival_function_at_times(time_point).iloc[0])


    def restricted_mean_survival_time(self,outcome,censor,time_point):
        kmf = KaplanMeierFitter()
        kmf.fit(outcome,event_observed=censor)
        return float(lifelines_restricted_mean_survival_time(kmf, t=time_point))
    
    
    def copy_parent(self,parent,iteration):
        #Attributes cloned from parent
        self.feature_list = copy.deepcopy(parent.feature_list) #sorting is for feature list comparison
        self.group_threshold = copy.deepcopy(parent.group_threshold)
        self.birth_iteration = iteration


    def uniform_crossover(self,other_offspring,threshold_evolving,random):
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

        # Apply crossover to thresholding if threshold_evolving
        if threshold_evolving:
            if random.random() < swap_probability:
                temp = self.group_threshold
                self.group_threshold = other_offspring.group_threshold
                other_offspring.group_threshold = temp


    def mutation(self,mutation_prob,feature_names,min_bin_size,max_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random):
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

        # Apply mutation to thresholding if threshold_evolving
        if threshold_evolving:
            if random.random() < mutation_prob:
                if min_thresh == max_thresh:
                    pass
                else:
                    thresh_list = [i for i in range(min_thresh,max_thresh+1)] #random.randint(min_thresh,max_thresh)
                    thresh_list.pop(thresh_list.index(self.group_threshold)) #pick a feature not already in the bin
                    random_thresh = random.choice(thresh_list)
                    self.group_threshold = random_thresh


    def merge(self,other_parent,max_bin_size,threshold_evolving,max_thresh,random):
        # Merge feature lists of two parents
        # Create list of feature names unique to one list or another
        set1 = set(self.feature_list)
        set2 = set(other_parent.feature_list)
        unique_to_list2 = set2 - set1
        self.feature_list = self.feature_list + list(unique_to_list2)   
        #Enforce maximum bin size
        while len(self.feature_list) > max_bin_size: 
            self.feature_list.remove(random.choice(self.feature_list))

        if threshold_evolving:
            if self.group_threshold == 0 or other_parent.group_threshold == 0:
                self.group_threshold += 1
            self.group_threshold += other_parent.group_threshold
            #Enforce maximum group threshold
            if self.group_threshold > max_thresh:
                self.group_threshold = max_thresh


    def calculate_pre_fitness(self,group_strata_min,penalty,fitness_metric,feature_names):
        # Penalize fitness if group counts are beyond the minimum group strata parameter (Ryan Check below)
        self.group_strata_prop = min(self.count_bt/(self.count_bt+self.count_at),self.count_at/(self.count_bt+self.count_at))
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

        if self.used_group_strata_fallback and self.pre_fitness != None:
            self.pre_fitness = (1-penalty) * self.pre_fitness

    def random_bin(self,feature_names,min_bin_size,max_bin_init_size,random):
        """Takes an previously generated offspring bin (that already existed in the pop) and generates an new feature_list """
        # Initialize features in bin
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)


    def is_equivalent(self,other_bin):
        # Bin equivalence is based on 'feature_list' and 'group_threshold'
        equivalent = False
        if int(self.group_threshold) == int(other_bin.group_threshold): #Same group threshold
            if sorted(self.feature_list) == sorted(other_bin.feature_list):
                equivalent = True
        return equivalent
    

    def bin_report(self):
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Log-Rank Score:', 'Log-Rank p-value:' ,'Bin Size:', 'Group Ratio:', 
                    'Count At/Below Threshold:', 'Count Above Threshold:','Birth Iteration:','Residuals Score:','Residuals p-value']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold, self.fitness,self.pre_fitness,self.log_rank_score, self.log_rank_p_value,
                                   self.bin_size, self.group_strata_prop, self.count_bt, self.count_at, self.birth_iteration,self.residuals_score,self.residuals_p_value]],columns=columns,index=None)
        return report_df
    

    def bin_short_report(self):
        columns = ['Features in Bin:', 'Threshold:', 'Fitness','Pre-Fitness:', 'Bin Size:', 'Group Ratio:','Birth Iteration:']
        report_df = pd.DataFrame([[self.feature_list, self.group_threshold, self.fitness,self.pre_fitness, self.bin_size, self.group_strata_prop,self.birth_iteration]],columns=columns,index=None).T
        return report_df
