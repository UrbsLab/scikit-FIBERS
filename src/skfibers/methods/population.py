#import random
import pandas as pd
from .bin import BIN

class BIN_SET:
    def __init__(self,manual_bin_init,feature_df,outcome_df,censor_df,feature_names,pop_size,min_bin_size,max_bin_init_size,
                 group_thresh,min_thresh,max_thresh,int_thresh,outcome_type,fitness_metric,log_rank_weighting,pareto_fitness,group_strata_min,
                 outcome_label,censor_label,threshold_evolving,penalty,iterations,iteration,random):
        #Initialize bin population
        self.bin_pop = []
        self.offspring_pop = []

        if manual_bin_init != None:
            # Load manually curated or previously trained bin population
            print("manual bin intitialization not yet implemented")
        else:
            #Random bin initialization
            while len(self.bin_pop) < pop_size:
                new_bin = BIN()
                new_bin.initialize_random(feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,iteration,random)
                # Check for duplicate rules based on feature list and threshold
                while self.equivalent_bin_in_pop(new_bin): # May slow down evolutionary cycles if new bins aren't found right away
                    new_bin.random_bin(feature_names,min_bin_size,max_bin_init_size,random)

                new_bin.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,
                                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration)
                
                new_bin.calculate_fitness(pareto_fitness,group_strata_min,penalty)

                self.bin_pop.append(new_bin)
        #print("Random Seed Check - Post Bin Init: "+ str(random.random()))


    def select_parent_pair(self,tournament_prop,random):
        #Tournament Selection
        parent_list = [None, None]
        tSize = int(len(self.bin_pop) * tournament_prop) #Tournament Size

        currentCount = 0
        while currentCount < 2:
            random.shuffle(self.bin_pop)
            parent_list[currentCount] = max(self.bin_pop[:tSize], key=lambda x: x.fitness)
            currentCount += 1
        return parent_list


    def generate_offspring(self,crossover_prob,mutation_prob,iterations,iteration,parent_list,feature_names,threshold_evolving,min_bin_size,
                           max_bin_init_size,min_thresh,max_thresh,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,
                           outcome_label,censor_label,int_thresh,group_thresh,pareto_fitness,group_strata_min,penalty,random):
        #print("Random Seed Check - genoff: "+ str(random.random()))
        # Clone Parents
        offspring_1 = BIN()
        offspring_2 = BIN()
        offspring_1.copy_parent(parent_list[0],iteration)
        offspring_2.copy_parent(parent_list[1],iteration)

        # Crossover
        offspring_1.uniform_crossover(offspring_2,crossover_prob,threshold_evolving,random)
        #print("Random Seed Check - crossover: "+ str(random.random()))

        # Mutation - check for duplicate rules
        offspring_1.mutation(mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random)
        offspring_2.mutation(mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random)
        #print("Random Seed Check - mutation: "+ str(random.random()))

        # Check for duplicate rules based on feature list and threshold
        while self.equivalent_bin_in_pop(offspring_1): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_1.random_bin(feature_names,min_bin_size,max_bin_init_size,random)

        while self.equivalent_bin_in_pop(offspring_2): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_2.random_bin(feature_names,min_bin_size,max_bin_init_size,random)
        #print("Random Seed Check - duplicate: "+ str(random.random()))
        # Offspring Evalution 
        offspring_1.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration)
        offspring_2.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,log_rank_weighting,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration)
        #print("Random Seed Check - evatluate: "+ str(random.random()))
        offspring_1.calculate_fitness(pareto_fitness,group_strata_min,penalty)
        offspring_2.calculate_fitness(pareto_fitness,group_strata_min,penalty)

        #Add New Offspring to the Population
        self.offspring_pop.append(offspring_1)
        self.offspring_pop.append(offspring_2)


    def equivalent_bin_in_pop(self,new_bin):
        for existing_bin in self.bin_pop:
            if new_bin.is_equivalent(existing_bin):
                return True
        for existing_bin in self.offspring_pop:
            if new_bin.is_equivalent(existing_bin):
                return True
        return False
        
    
    def bin_deletion_probabilistic(self,pop_size,elitism,random):
        # Automatically delete bins with a fitness of 0
        delete_indexes = []
        i = 0
        for bin in self.bin_pop:
            if bin.fitness == 0:
                delete_indexes.append(i)
            i += 1
        delete_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in delete_indexes:
            del self.bin_pop[index]

        # Preseve any proportion of elite bins specified
        elite_count = int(pop_size*(elitism))
        self.bin_pop = sorted(self.bin_pop, key=lambda x: x.fitness,reverse=True)
        elite_bins = self.bin_pop[-elite_count:]
        remaining_bins = self.bin_pop[:-elite_count]

        # ROULETTE WHEEL SELECTION - deletion selection probability inversely related to bin fitness
        # Delete remaining bins required (from non-elite set) based on bin selection that is inversely proportional to bin fitness
        while len(remaining_bins)+len(elite_bins) > pop_size:
            #Calculate total fitness across all bins
            total_fitness = sum(1/bin.fitness for bin in remaining_bins)
            # Calculate deletion probabilities for each object
            deletion_probabilities = [(1/bin.fitness) / total_fitness for bin in remaining_bins]
            index = random.choices(range(len(remaining_bins)), weights=deletion_probabilities)[0]
            del remaining_bins[index]

        self.bin_pop = elite_bins + remaining_bins


    def bin_deletion_deterministic(self,pop_size):
        # Calculate number of bins to delete
        self.bin_pop = sorted(self.bin_pop, key=lambda x: x.fitness,reverse=True)
        while len(self.bin_pop) > pop_size:
            del self.bin_pop[-1]


    def add_offspring_into_pop(self):
        self.bin_pop = self.bin_pop + self.offspring_pop
        self.offspring_pop = []


    def sort_feature_lists(self):
        for bin in self.bin_pop:
            bin.feature_list = sorted(bin.feature_list)


    def report_pop(self):
        self.sort_feature_lists()
        pop_df = pd.DataFrame([vars(instance) for instance in self.bin_pop])
        print(pop_df)


    def get_pop(self):
        self.sort_feature_lists()
        pop_df = pd.DataFrame([vars(instance) for instance in self.bin_pop])
        return pop_df


    def get_all_top_bins(self):
        top_bin_list = [self.bin_pop[0]]
        highest_fitness = self.bin_pop[0].fitness
        bin_index = 1
        while self.bin_pop[bin_index].fitness == highest_fitness:
            top_bin_list.append(self.bin_pop[bin_index])
            bin_index += 1
        return top_bin_list



