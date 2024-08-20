import numpy as np
import pandas as pd
import copy
import math
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from .bin import BIN
from .pareto import PARETO
import warnings


class BIN_SET:
    def __init__(self,manual_bin_init,df,feature_names,pop_size,min_bin_size,max_bin_init_size,
                 group_thresh,min_thresh,max_thresh,int_thresh,outcome_type,fitness_metric,log_rank_weighting,group_strata_min,
                 outcome_label,censor_label,threshold_evolving,penalty,iterations,iteration,residuals,covariates, naive_survival_optimization, random):
        #Initialize bin population
        self.bin_pop = []
        self.offspring_pop = []
        self.feature_tracking = [0]*len(feature_names)
        self.pareto = PARETO()

        if isinstance(manual_bin_init, pd.DataFrame): # Load manually curated or previously trained bin population
            for index, row in manual_bin_init.iterrows():
                feature_text = row[0]
                feature_list = eval(feature_text)
                loaded_bin = [item.strip("[]'") for item in feature_list]
                loaded_thresh = row[1]
                birth_iteration = row[10]
                new_bin = BIN(self.pareto)
                new_bin.initialize_manual(feature_names,loaded_bin,loaded_thresh,group_thresh,min_thresh,max_thresh,birth_iteration)
                # Bin metric score evaluation
                new_bin.evaluate(df.loc[:,feature_names],df.loc[:,outcome_label],df.loc[:,censor_label],outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,df.loc[:,covariates], naive_survival_optimization)
                #Add new bin to population
                self.bin_pop.append(new_bin)
                if fitness_metric == 'pareto':
                    if self.pareto.update_front(new_bin.log_rank_score, new_bin.low_risk_area, ['max', 'max']):
                        # If a bin is added to the front, all bin's pre-fitness must be recalculated
                        for bin in self.bin_pop:
                            bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
                new_bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization) 

        #Random bin initialization
        while len(self.bin_pop) < pop_size:
            new_bin = BIN(self.pareto)
            new_bin.initialize_random(feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,iteration,random)
            # Check for duplicate rules based on feature list and threshold
            while self.equivalent_bin_in_pop(new_bin,iteration): # May slow down evolutionary cycles if new bins aren't found right away
                new_bin.random_bin(feature_names,min_bin_size,max_bin_init_size,random)
            # Bin metric score evaluation
            new_bin.evaluate(df.loc[:,feature_names],df.loc[:,outcome_label],df.loc[:,censor_label],outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                                censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,df.loc[:,covariates], naive_survival_optimization)
            #Add new bin to population
            self.bin_pop.append(new_bin)
            if fitness_metric == 'pareto':
                if self.pareto.update_front(new_bin.log_rank_score, new_bin.low_risk_area, ['max', 'max']):
                    # If a bin is added to the front, all bin's pre-fitness must be recalculated
                    for bin in self.bin_pop:
                        bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
            # Fitness metric calculation based on bin metric score
            new_bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization) 

    def update_feature_tracking(self, feature_names):
        for bin in self.bin_pop:
            for feature in bin.feature_list:
                index = feature_names.index(feature)
                self.feature_tracking[index] += bin.pre_fitness
    
    """
    def custom_sort_key(self, obj):
        return (-obj.pre_fitness,obj.group_threshold,obj.bin_size,-obj.group_strata_prop)
    """
    # change to custom sort key since pre_fitness used to be just log rank score
    def custom_sort_key(self, obj):
        if obj.log_rank_score == None:
            return (-obj.pre_fitness,-obj.residuals_score, -obj.low_risk_area, obj.group_threshold,obj.bin_size,-obj.group_strata_prop)
        else:
            return (-obj.pre_fitness,-obj.log_rank_score, -obj.low_risk_area, obj.group_threshold,obj.bin_size,-obj.group_strata_prop)
    
    def global_fitness_update(self,penalty):
        self.bin_pop = sorted(self.bin_pop, key=self.custom_sort_key)
        # Sort bin population first by pre-fitness, then by group_theshold, then by bin_size, then by group_strata_prop (to form a global bin ranking)
        # Sort DataFrame by maximizing column A (descending) and minimizing column B (ascending) for ties
        decay = 0.2
        self.bin_pop = sorted(self.bin_pop, key=self.custom_sort_key)
        previous_objective_list = [None,None,None,None,None,None]
        index = -1

        for bin in self.bin_pop:
            if bin.pre_fitness == 0:
                bin.fitness = 0
                index += 1 
            else:
                objective_list = [bin.pre_fitness, bin.log_rank_score, bin.low_risk_area, bin.group_threshold, bin.bin_size, bin.group_strata_prop]
                if objective_list != previous_objective_list: 
                    index += 1 #Only advance bin ranking if next bin is different across at least one objective
                bin.fitness = np.exp(-index / (len(self.bin_pop)*decay)) 

            previous_objective_list = [bin.pre_fitness, bin.log_rank_score, bin.low_risk_area, bin.group_threshold, bin.bin_size, bin.group_strata_prop]


    def select_parent_pair(self,tournament_prop,random):
        #Tournament Selection
        #parent_list = [None, None]
        tSize = int(len(self.bin_pop) * tournament_prop) #Tournament Size
        #currentCount = 0
        #while currentCount < 2:
        #    random.shuffle(self.bin_pop)
        #    parent_list[currentCount] = max(self.bin_pop[:tSize], key=lambda x: x.fitness)
        #    currentCount += 1
        #return parent_list
        parent_1 = self.tournament_selection(tSize,random)
        parent_2 = self.tournament_selection(tSize,random)

        while parent_1 == parent_2:
            parent_2 = self.tournament_selection(tSize,random)

        return [parent_1,parent_2]
    

    def tournament_selection(self,tSize,random):
        random.shuffle(self.bin_pop)
        new_parent = max(self.bin_pop[:tSize], key=lambda x: x.fitness)
        return new_parent


    def generate_offspring(self,crossover_prob,mutation_prob,merge_prob,iterations,iteration,parent_list,feature_names,threshold_evolving,min_bin_size,max_bin_size,
                           max_bin_init_size,min_thresh,max_thresh,df,outcome_type,fitness_metric,log_rank_weighting,
                           outcome_label,censor_label,int_thresh,group_thresh,group_strata_min,penalty,residuals,covariates, naive_survival_optimization, random):
        #print("Random Seed Check - genoff: "+ str(random.random()))
        # Clone Parents
        offspring_1 = BIN(self.pareto)
        offspring_2 = BIN(self.pareto)
        offspring_1.copy_parent(parent_list[0],iteration)
        offspring_2.copy_parent(parent_list[1],iteration)

        #if iteration == 49:
        #    print('Parent1:'+str(offspring_1.feature_list)+'_'+str(offspring_1.group_threshold))
        #    print('Parent2:'+str(offspring_2.feature_list)+'_'+str(offspring_2.group_threshold))

        if random.random() < merge_prob: #Generate a single novel bin that is the combination of the two parent bins (yielding 3 total bins created during this mating)
            offspring_3 = BIN(self.pareto)
            offspring_3.copy_parent(parent_list[0],iteration)
            offspring_3.merge(parent_list[1],max_bin_size,threshold_evolving,max_thresh,random)

            # Check for duplicate rules based on feature list and threshold
            #if iteration == 49:
            #    print('merge')
            while self.equivalent_bin_in_pop(offspring_3,iteration): # May slow down evolutionary cycles if new bins arent' found right away
                offspring_3.random_bin(feature_names,min_bin_size,max_bin_init_size,random)
                #if iteration == 49:
                #    print(str(offspring_3.feature_list)+'_'+str(offspring_3.group_threshold))
            offspring_3.evaluate(df.loc[:,feature_names],df.loc[:,outcome_label],df.loc[:,censor_label],outcome_type,fitness_metric,log_rank_weighting,outcome_label,censor_label,min_thresh,max_thresh,
                                int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,df.loc[:,covariates], naive_survival_optimization)
            #if iteration == 49:
            #    print(str(offspring_3.feature_list)+'_'+str(offspring_3.group_threshold))
            if not self.equivalent_bin_in_pop(offspring_3,iteration):
                self.offspring_pop.append(offspring_3)
                if fitness_metric == 'pareto':
                    if self.pareto.update_front(offspring_3.log_rank_score, offspring_3.low_risk_area, ["max", "max"]):
                        # If a bin is added to the front, all bin's pre-fitness must be recalculated
                        for bin in self.bin_pop:
                            bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
            # Fitness metric calculation based on bin metric score
            offspring_3.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization) 
        # Crossover
        offspring_1.uniform_crossover(offspring_2,crossover_prob,threshold_evolving,random)

        # Mutation - check for duplicate rules
        offspring_1.mutation(mutation_prob,feature_names,min_bin_size,max_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random)
        offspring_2.mutation(mutation_prob,feature_names,min_bin_size,max_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random)

        #if iteration == 49:
        #    print('Offspring1:'+str(offspring_1.feature_list)+'_'+str(offspring_1.group_threshold))
        #    print('Offspring2:'+str(offspring_2.feature_list)+'_'+str(offspring_2.group_threshold))

        # Check for duplicate bins based on feature list and threshold
        #if iteration == 49:
        #    print('off1')
        while self.equivalent_bin_in_pop(offspring_1,iteration): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_1.random_bin(feature_names,min_bin_size,max_bin_init_size,random)
            #if iteration == 49:
            #    print(str(offspring_1.feature_list)+'_'+str(offspring_1.group_threshold))

        # Offspring 1 Evalution 
        offspring_1.evaluate(df.loc[:,feature_names],df.loc[:,outcome_label],df.loc[:,censor_label],outcome_type,fitness_metric,log_rank_weighting,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,df.loc[:,covariates], naive_survival_optimization)
        #Add New Offspring 1 to the Population
        #if iteration == 49:
        #    print(str(offspring_1.feature_list)+'_'+str(offspring_1.group_threshold))
        if not self.equivalent_bin_in_pop(offspring_1,iteration):
            self.offspring_pop.append(offspring_1)
            if fitness_metric == 'pareto':
                if self.pareto.update_front(offspring_1.log_rank_score, offspring_1.low_risk_area, ["max", "max"]):
                    # If a bin is added to the front, all bin's pre-fitness must be recalculated
                    for bin in self.bin_pop:
                        bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
        # Fitness metric calculation based on bin metric score
        offspring_1.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
        #if iteration == 49:
        #    print('off2')
        # Check for duplicate bins based on feature list and threshold
        while self.equivalent_bin_in_pop(offspring_2,iteration): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_2.random_bin(feature_names,min_bin_size,max_bin_init_size,random)
            #if iteration == 49:
            #    print(str(offspring_2.feature_list)+'_'+str(offspring_2.group_threshold))

        # Offspring 2 Evalution 
        offspring_2.evaluate(df.loc[:,feature_names],df.loc[:,outcome_label],df.loc[:,censor_label],outcome_type,fitness_metric,log_rank_weighting,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration,residuals,df.loc[:,covariates], naive_survival_optimization)
        #Add New Offspring 2 to the Population
        #if iteration == 49:
        #    print(str(offspring_2.feature_list)+'_'+str(offspring_2.group_threshold))
        if not self.equivalent_bin_in_pop(offspring_2,iteration):
            self.offspring_pop.append(offspring_2)
            if fitness_metric == 'pareto':
                if self.pareto.update_front(offspring_2.log_rank_score, offspring_2.low_risk_area, ["max", "max"]):
                    # If a bin is added to the front, all bin's pre-fitness must be recalculated
                    for bin in self.bin_pop:
                        bin.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization)
        # Fitness metric calculation based on bin metric score
        offspring_2.calculate_pre_fitness(group_strata_min,penalty,fitness_metric,feature_names, naive_survival_optimization) 

    def equivalent_bin_in_pop(self,new_bin,iteration):
        for existing_bin in self.offspring_pop:
            if new_bin.is_equivalent(existing_bin):
                #if iteration == 49:
                #    print('duplicate in offpop')
                #    print(str(new_bin.feature_list)+'_'+str(new_bin.group_threshold))
                #    print(str(existing_bin.feature_list)+'_'+str(new_bin.group_threshold))
                return True
            
        for existing_bin in self.bin_pop:
            if new_bin.is_equivalent(existing_bin):
                #if iteration == 49:
                #    print('duplicate in pop')
                #    print(str(new_bin.feature_list)+'_'+str(new_bin.group_threshold))
                #    print(str(existing_bin.feature_list)+'_'+str(new_bin.group_threshold))
                return True

        return False
        

    def similarity_bin_deletion(self,pop_size,diversity_pressure,random):
        # Automatically delete bins with a fitness of 0
        delete_indexes = []
        i = 0
        for bin in self.bin_pop:
            if bin.fitness == 0 and len(delete_indexes)<(len(self.bin_pop)-pop_size):
                delete_indexes.append(i)
            i += 1
        delete_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in delete_indexes:
            del self.bin_pop[index]

        #Prepare for deletion
        list_of_feature_lists = []
        for bin in self.bin_pop:
            bin_composition = copy.deepcopy(bin.feature_list)
            bin_composition.append('Thresh_'+str(bin.group_threshold))
            list_of_feature_lists.append(bin_composition)

        # Vectorize the lists using TF-IDF
        vectorizer = TfidfVectorizer(analyzer=lambda x: x)
        X = vectorizer.fit_transform(list_of_feature_lists)

        # Calculate pairwise cosine similarity
        cos_sim = cosine_similarity(X)

        # Cluster the lists using KMeans
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seed = random.randint(0,100)
            kmeans = KMeans(n_clusters=diversity_pressure, n_init='auto',random_state=seed).fit(cos_sim)
            group_labels = kmeans.labels_

        #For each group find elites (to preserve) i.e. assign 0 deletion probability othewise assign deletion probability
        for cluster in range(0,diversity_pressure):
            # Get bin indexes and respective fitness scores for bins in this cluster
            bin_indexs = [i for i, x in enumerate(group_labels) if x == cluster]
            bin_fitness_list = []
            for bin_index in bin_indexs: # for each bin in this cluster
                bin_fitness_list.append(self.bin_pop[bin_index].fitness) #get the fitness
            # Find the best bin in this cluster
            max_fitness = max(bin_fitness_list)
            max_index = bin_indexs[bin_fitness_list.index(max_fitness)] #index of bin from bin_indexs
            for bin_index in bin_indexs: #original bin indexes limited to bins in cluster
                if bin_index == max_index or self.bin_pop[max_index].fitness == self.bin_pop[bin_index].fitness: # Top bin in cluster and any other bins with same highest fitness
                    self.bin_pop[bin_index].update_deletion_prop(0.0, cluster) #Top bin s have zero chance of deletion
                else:
                    # Get similarity score to top bin in cluster
                    similarity = cos_sim[bin_index][max_index] #compare the current bin to the top bin in this cluster
                    self.bin_pop[bin_index].update_deletion_prop((1/bin.fitness)*similarity+(1/bin.fitness), cluster) # Assign deletion probability as the inverse of fitness * similarity

        # ROULETTE WHEEL SELECTION - deletion selection probability inversely related to bin fitness
        # Delete remaining bins required (from non-elite set) based on bin selection that is inversely proportional to bin fitness
        while len(self.bin_pop) > pop_size:
            #Calculate total fitness across all bins
            total_fitness = sum(bin.deletion_prop for bin in self.bin_pop)
            # Calculate deletion probabilities for each object
            deletion_probabilities = [bin.deletion_prop / total_fitness for bin in self.bin_pop]
            index = random.choices(range(len(self.bin_pop)), weights=deletion_probabilities)[0]
            del self.bin_pop[index]


    def probabilistic_bin_deletion(self,pop_size,elitism,random):
        # Automatically delete bins with a fitness of 0
        delete_indexes = []
        i = 0
        for bin in self.bin_pop:
            if bin.fitness == 0 and len(delete_indexes)<(len(self.bin_pop)-pop_size):
                delete_indexes.append(i)
            i += 1
        delete_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in delete_indexes:
            del self.bin_pop[index]

        # Preseve any proportion of elite bins specified
        x = 0
        while self.bin_pop[x].fitness == 1:
            x+=1 #gets the bin index where fitness begins to drop
        elite_count = int(pop_size*(elitism))
        if (elite_count < len(self.pareto.bin_front)):
            elite_count = len(self.pareto.bin_front)
        if elite_count < x+1: #
            elite_count = x #count is -1 for indexing below

        elite_bins = self.bin_pop[:elite_count]
        remaining_bins = self.bin_pop[elite_count:]

        for bin in elite_bins:
            bin.update_deletion_prop(0, None)

        # ROULETTE WHEEL SELECTION - deletion selection probability inversely related to bin fitness
        # Delete remaining bins required (from non-elite set) based on bin selection that is inversely proportional to bin fitness
        while len(remaining_bins)+len(elite_bins) > pop_size:
            #Calculate total fitness across all bins
            total_fitness = sum(1/bin.fitness for bin in remaining_bins)
            # Calculate deletion probabilities for each object
            deletion_probabilities = [(1/bin.fitness) / total_fitness for bin in remaining_bins]
            remaining_index = 0
            for bin in remaining_bins:
                bin.update_deletion_prop(deletion_probabilities[remaining_index],None)
                remaining_index += 1
            index = random.choices(range(len(remaining_bins)), weights=deletion_probabilities)[0]
            del remaining_bins[index]

        self.bin_pop = elite_bins + remaining_bins


    def deterministic_bin_deletion(self,pop_size):
        # Automatically delete bins with a fitness of 0
        delete_indexes = []
        i = 0
        for bin in self.bin_pop:
            if bin.fitness == 0 and len(delete_indexes)<(len(self.bin_pop)-pop_size):
                delete_indexes.append(i)
            i += 1
        delete_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in delete_indexes:
            del self.bin_pop[index]
        # Delete remaining lowest fitness bins until pop_size reached
        while len(self.bin_pop) > pop_size:
            del self.bin_pop[-1]


    def add_offspring_into_pop(self,iteration):
        #if iteration == 49:
        #    print("---------------------------------------------------------")
        #    for each in self.offspring_pop:
        #        print(str(each.feature_list)+'_'+str(each.group_threshold))
        #    print("---------------------------------------------------------")
        self.bin_pop = self.bin_pop + self.offspring_pop
        self.offspring_pop = []

    # Fitness sharing is a niching technique to encourage bin diversity in both objectives, penalizing bins with similar
    # values in both objectives
    def sharing_penalization(self, sharing_thresh):
        for bin in self.bin_pop:
            penalty = 1
            p1 = (bin.log_rank_score, bin.low_risk_area)
            # penalize bins with similar results only if it is not in the pareto front
            if p1 not in self.pareto.bin_front:
                for other in self.bin_pop:
                    p2 = (other.log_rank_score, other.low_risk_area)
                    dist = math.dist(p1, p2)
                    temp = max(0, (1 - dist / sharing_thresh))
                    penalty += temp ** 0.5
                bin.pre_fitness = float(bin.pre_fitness / penalty)

    def sort_feature_lists(self):
        for bin in self.bin_pop:
            bin.feature_list = sorted(bin.feature_list)


    def report_pop(self):
        self.sort_feature_lists()
        pd.set_option('display.max_colwidth', None) # prevent truncation of dataframe
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pop_df = pd.DataFrame([vars(instance) for instance in self.bin_pop])
        print(pop_df)


    def get_all_top_bins(self):
        top_bin_list = [self.bin_pop[0]]
        highest_fitness = self.bin_pop[0].fitness
        bin_index = 1
        while self.bin_pop[bin_index].fitness == highest_fitness:
            top_bin_list.append(self.bin_pop[bin_index])
            bin_index += 1
        return top_bin_list
    
    def pop_clean_group_thresh(self,group_strata_min):
        temp_pop = []
        for bin in self.bin_pop:
            if bin.group_strata_prop >= group_strata_min:
                temp_pop.append(bin)
            else:
                if bin in self.pareto.bin_front:
                    self.pareto.delete_from_front(bin.log_rank_score, bin.low_risk_area)
        self.bin_pop = temp_pop

    def get_pareto_front(self):
        pareto_front = []
        for bin in self.bin_pop:
            if (bin.log_rank_score, bin.low_risk_area) in self.pareto.bin_front:
                pareto_front.append(bin)
        return pareto_front

    def get_min_area(self):
        min = None
        for bin in self.bin_pop:
            if min == None or bin.low_risk_area < float(min) :
                min = bin.low_risk_area
        return min