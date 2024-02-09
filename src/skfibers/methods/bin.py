import numpy as np
import pandas as pd
import random
import copy
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import ranksums

class BIN:
    def __init__(self):
        self.feature_list = []
        self.group_threshold = None
        self.fitness = None
        self.metric = None
        self.bin_size = None
        self.group_strata_prop = None
        self.low_risk_count = None
        self.high_risk_count = None
        self.birth_iteration = None

    def initialize_random(self,feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,random_seed):
        self.birth_iteration = 0
        # Initialize features in bin
        random.seed(random_seed)  
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)
        if group_thresh != None: # Defined group threshold
            self.group_threshold = group_thresh
        else: # Adaptive group threshold
            self.group_threshold = random.randint(min_thresh,max_thresh)
    


    def evaluate(self,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,
                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving):
        # Sum instance values across features specified in the bin
        feature_sums = feature_df[self.feature_list].sum(axis=1)
        bin_df = pd.DataFrame({'feature_sum':feature_sums})
        #print("Bin Data Shape: "+str(bin_df.shape))

        # Create evaluation dataframe including bin sum feature with 
        bin_df = pd.concat([bin_df,outcome_df,censor_df],axis=1) #, ignore_index=True
        #print("Bin Data Shape2: "+str(bin_df.shape))
        #print(bin_df.head())

        if group_thresh == None and not threshold_evolving: #Adaptive thresholding activated
            # Select best threshold by evaluating all considered
            best_score = 0
            for threshold in range(min_thresh, max_thresh + 1):
                score = self.evaluate_for_threshold(bin_df,outcome_label,censor_label,outcome_type,fitness_metric,threshold)
                if score > best_score:
                    self.metric = score
                    self.group_threshold = threshold
                    best_score = score
        else: #Use the given group threshold to evaluate the bin
            score = self.evaluate_for_threshold(bin_df,outcome_label,censor_label,outcome_type,fitness_metric,self.group_threshold)
        self.metric = score


    def evaluate_for_threshold(self,bin_df,outcome_label,censor_label,outcome_type,fitness_metric,group_threshold):
        #Create dataframes including instances from either high or low risk groups
        low_df = bin_df[bin_df['feature_sum'] <= group_threshold]
        high_df = bin_df[bin_df['feature_sum'] > group_threshold]

        low_outcome = low_df[outcome_label].to_list()
        high_outcome = high_df[outcome_label].to_list()
        low_censor = low_df[censor_label].to_list()
        high_censor =high_df[censor_label].to_list()
        self.low_risk_count = len(low_outcome)
        self.high_risk_count = len(high_outcome)
 
        # Apply selected evaluation strategy/metric
        if outcome_type == 'survival':
            if fitness_metric == 'log_rank':
                results = logrank_test(low_outcome, high_outcome, event_observed_A=low_censor,event_observed_B=high_censor)
                score = results.test_statistic #test all thresholds by default in initial pop.
            if fitness_metric == 'residuals':
                pass
            if fitness_metric == 'aic':
                pass
        elif outcome_type == 'class':
            print("Classification not yet implemented")
        else:
            print("Specified outcome_type not supported")
        return score
    

    def copy_parent(self,parent,iteration):
        #Attributes cloned from parent
        self.feature_list = sorted(copy.deepcopy(parent.feature_list)) #sorting is for feature list comparison
        self.group_threshold = copy.deepcopy(parent.group_threshold)
        self.birth_iteration = iteration


    def uniform_crossover(self,other_offspring,crossover_prob,threshold_evolving,random_seed):
        #create list of feature names unique to one list or another
        random.seed(random_seed)  

        # Convert lists to sets for efficient set operations
        set1 = set(self.feature_list)
        set2 = set(other_offspring.feature_list)
        
        # Find elements unique to list1 and list2
        unique_to_list1 = set1 - set2
        unique_to_list2 = set2 - set1

        # Combine unique elements from both lists
        unique_features = list(unique_to_list1.union(unique_to_list2))

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


    def mutation(self,mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random_seed):

        random.seed(random_seed) 

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

        else: # Addition, Deletion, or Swap 
            for feature in self.feature_list:
                if random.random() < mutation_prob:
                    other_features = [value for value in feature_names if value not in self.feature_list] #pick a feature not already in the bin
                    random_feature = random.choice(other_features)
                    random_num = random.random()
                    if random_num < 0.333: # Swap
                        self.feature_list.remove(feature)
                        self.feature_list.append(random_feature)
                    elif random_num >= 0.333 and random_num < 0.666: # Addition
                        self.feature_list.append(random_feature)
                    else: # Deletion
                        self.feature_list.remove(feature)

        # Apply crossover to thresholding if threshold_evolving
        if threshold_evolving:
            if random.random() < mutation_prob:
                thresh_count = random.randint(min_thresh,max_thresh)
                other_thresh = [value for value in thresh_count if value != self.group_threshold] #pick a feature not already in the bin
                random_thresh = random.choice(other_thresh)
                self.group_threshold = random_thresh


    def calculate_fitness(self,pareto_fitness,group_strata_min,penalty):
        if pareto_fitness: #Apply pareto-front-based multi-objective fitness
            print("Pareto-fitness has not yet been implemented")
            pass
        else:
            # Penalize fitness if risk group counts are beyond minimum risk group strata parameter (Ryan Check below)
            self.group_strata_prop = min(self.low_risk_count/(self.low_risk_count+self.high_risk_count),self.high_risk_count/(self.low_risk_count+self.high_risk_count))
            if self.group_strata_prop < group_strata_min: 
            #if self.low_risk_count > (group_strata_min * (self.low_risk_count + self.high_risk_count)) and self.high_risk_count > (group_strata_min * (self.low_risk_count + self.high_risk_count)):
                self.fitness = penalty * self.metric
            else:
                self.fitness = self.metric


    def random_bin(self,feature_names,min_bin_size,max_bin_init_size,random_seed):
        """Takes an previously generated offspring bin (that already existed in the pop) and generates an new feature_list """
        # Initialize features in bin
        random.seed(random_seed)  
        feature_count = random.randint(min_bin_size,max_bin_init_size)
        self.feature_list = random.sample(feature_names,feature_count)
        self.bin_size = len(self.feature_list)


    def is_equivalent(self,other_offspring):
        # Bin equivalence is based on 'feature_list' and 'group_threshold'
        equivalent = False
        if self.group_threshold == other_offspring.group_threshold: #Same group threshold
            if sorted(self.feature_list) == sorted(other_offspring.feature_list):
                equivalent = True
        return equivalent
    



    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.feature_list):
            x = self.feature_list[self.index]
            self.index += 1
            return x
        else:
            raise StopIteration

    def __eq__(self, other):
        if not isinstance(other, BIN):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return sorted(self.feature_list) == sorted(other.feature_list)

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, item):
        return self.feature_list[item]

    def __str__(self):
        return str(self.feature_list)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return other.__lt__(self)

    # Setter methods of the BIN attributes

    def set_name(self, bin_name):
        self.bin_name = bin_name

    def set_score(self, score):
        self.score = score

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list

    def set_seen(self):
        self.seen = True

    def set_not_seen(self):
        self.seen = False

        # Getter methods of the BIN attributes

    def get_feature_list(self):
        return self.feature_list.copy()

    def get_threshold(self):
        return self.threshold

    def get_score(self):
        return self.score

    def get_name(self):
        return self.bin_name

    def was_seen(self):
        return self.seen

    # Log rank test used to score the different thresholds and fitness of the BIN
    def log_rank_test(self, bin_feature_matrix, label_name, duration_name,
                      informative_cutoff, threshold):
        score = 0
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]

        durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        durations_mm = df_1[duration_name].to_list()
        event_observed_mm = df_1[label_name].to_list()

        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            results = logrank_test(durations_no, durations_mm, event_observed_A=event_observed_no,
                                   event_observed_B=event_observed_mm)
            score = results.test_statistic

        return score

    def residuals_score(self, residuals, bin_feature_matrix, label_name, duration_name,
                        informative_cutoff, threshold):
        score = 0
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]  # SPHIA
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]
        durations_no = df_0[duration_name].to_list()
        event_observed_no = df_0[label_name].to_list()
        durations_mm = df_1[duration_name].to_list()
        event_observed_mm = df_1[label_name].to_list()

        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            bin_residuals = residuals.loc[bin_feature_matrix[self.bin_name] <= threshold]
            bin_residuals = bin_residuals["deviance"]

            non_bin_residuals = residuals.loc[bin_feature_matrix[self.bin_name] > threshold]
            non_bin_residuals = non_bin_residuals["deviance"]

            score = abs(ranksums(bin_residuals, non_bin_residuals).statistic)
        return score

    def aic_score(self, covariate_matrix, bin_feature_matrix, label_name, duration_name,
                  informative_cutoff, threshold):
        df_0 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] <= threshold]
        df_1 = bin_feature_matrix.loc[bin_feature_matrix[self.bin_name] > threshold]

        event_observed_no = df_0[label_name].to_list()
        event_observed_mm = df_1[label_name].to_list()
        if len(event_observed_no) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)) and len(
                event_observed_mm) > informative_cutoff * (len(event_observed_no) + len(event_observed_mm)):
            column_values = bin_feature_matrix[self.bin_name].to_list()
            for r in range(0, len(column_values)):
                if column_values[r] > 0:
                    column_values[r] = 1
            data = covariate_matrix.copy()
            data['Bin'] = column_values
            data = data.loc[:, (data != data.iloc[0]).any()]
            cph = CoxPHFitter()
            cph.fit(data, duration_name, event_col=label_name)

            score = 0 - cph.AIC_partial_
        else:
            score = - np.inf
        return score

    # This method will update the threshold for the bin by try all the thresholds from min to max threshold
    # and uses the threshold that will get the highest score
    def try_all_thresholds(self, min_threshold, max_threshold, bin_feature_matrix,
                           label_name, duration_name, informative_cutoff, scoring_method="log_rank",
                           residuals=None, covariate_matrix=None):
        # to avoid unnecessary computations, if they have already tried all the thresholds which is determined by the
        # seen variable, then no need to try all the thresholds
        if not self.seen:
            highest_score = 0
            for threshold in range(min_threshold, max_threshold + 1):
                # Variable that will store the highest score used to determine the threshold
                if scoring_method == "log_rank":
                    score = self.log_rank_test(bin_feature_matrix, label_name,
                                               duration_name, informative_cutoff, threshold)
                elif scoring_method == "AIC":
                    score = self.aic_score(covariate_matrix, bin_feature_matrix, label_name,
                                           duration_name, informative_cutoff, threshold)
                elif scoring_method == "residuals":
                    score = self.residuals_score(residuals, bin_feature_matrix, label_name,
                                                 duration_name, informative_cutoff, threshold)
                if score > highest_score:
                    self.score = score
                    self.threshold = threshold
                    highest_score = score

            self.seen = True
