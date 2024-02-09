import random
import pandas as pd
from .bin import BIN

class BIN_SET:
    def __init__(self,manual_bin_init,feature_df,outcome_df,censor_df,feature_names,pop_size,min_bin_size,max_bin_init_size,
                 group_thresh,min_thresh,max_thresh,int_thresh,outcome_type,fitness_metric,pareto_fitness,group_strata_min,
                 outcome_label,censor_label,threshold_evolving,penalty,random_seed):
        
        #Initialize bin population
        self.bin_pop = []

        if manual_bin_init != None:
            # Load manually curated or previously trained bin population
            print("manual bin intitialization not yet implemented")
        else:
            #Random bin initialization
            while len(self.bin_pop) < pop_size:
                new_bin = BIN()

                new_bin.initialize_random(feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,random_seed)
                # Check for duplicate rules based on feature list and threshold
                while self.equivalent_bin_in_pop(new_bin): # May slow down evolutionary cycles if new bins aren't found right away
                    new_bin.random_bin(feature_names,min_bin_size,max_bin_init_size,random_seed)

                new_bin.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,
                                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving)
                
                new_bin.calculate_fitness(pareto_fitness,group_strata_min,penalty)

                self.bin_pop.append(new_bin)


    def select_parent_pair(self,tournament_prop,random_seed):
        #Tournament Selection
        random.seed(random_seed)  # You can change the seed value as desired
        parent_list = [None, None]
        tSize = int(len(self.bin_pop) * tournament_prop) #Tournament Size

        currentCount = 0
        while currentCount < 2:
            random.shuffle(self.bin_pop)
            parent_list[currentCount] = max(self.bin_pop.head(tSize), key=lambda x: x.fitness)
            print("tournament size: "+str(len(self.bin_pop.head(tSize))))
            currentCount += 1
        return parent_list


    def generate_offspring(self,crossover_prob,mutation_prob,iteration,parent_list,feature_names,threshold_evolving,min_bin_size,
                           max_bin_init_size,min_thresh,max_thresh,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,
                           outcome_label,censor_label,int_thresh,group_thresh,random_seed):
        # Clone Parents
        offspring_1 = BIN()
        offspring_2 = BIN()
        offspring_1.copy_parent(parent_list[0],iteration)
        offspring_2.copy_parent(parent_list[1],iteration)

        # Crossover
        offspring_1.uniform_crossover(offspring_2,crossover_prob,threshold_evolving,random_seed)

        # Mutation - check for duplicate rules
        offspring_1.mutation(mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random_seed)
        offspring_2.mutation(mutation_prob,feature_names,min_bin_size,max_bin_init_size,threshold_evolving,min_thresh,max_thresh,random_seed)

        # Check for duplicate rules based on feature list and threshold
        while self.equivalent_bin_in_pop(offspring_1): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_1.random_bin(feature_names,min_bin_size,max_bin_init_size,random_seed)

        while self.equivalent_bin_in_pop(offspring_2): # May slow down evolutionary cycles if new bins arent' found right away
            offspring_2.random_bin(feature_names,min_bin_size,max_bin_init_size,random_seed)

        # Offspring Evalution 
        offspring_1.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving)
        offspring_2.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving)
        
        #Add New Offspring to the Population
        self.bin_pop.append(offspring_1)
        self.bin_pop.append(offspring_2)

    def equivalent_bin_in_pop(self,offspring):
        bin_exists = False
        for existing_bin in self.bin_pop:
            offspring.is_equivalent(existing_bin)
        return bin_exists
    
    def bin_deletion_probabilistic(self,pop_size):
        # ROULETTE WHEEL SELECTION - deletion selection probability inversely related to bin fitness
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

        # Calculate remaining number of bins to delete
        bins_to_delete = len(self.bin_pop) - pop_size
        #Calculate total fitness across all bins
        total_fitness = sum(1/obj.fitness for obj in self.bin_pop)
        # Calculate deletion probabilities for each object
        deletion_probabilities = [(1/obj.fitness) / total_fitness for obj in self.bin_pop]
        selected_indexes = random.choices(range(len(self.bin_pop)), weights=deletion_probabilities,k=bins_to_delete)
        selected_indexes.sort(reverse=True) #sort in descending order so deletion does not affect subsequent indexes
        for index in selected_indexes:
            del self.bin_pop[index]


    def bin_deletion_deterministic(self,pop_size):
        # Calculate number of bins to delete
        bins_to_delete = len(self.bin_pop) - pop_size
        self.bin_pop = sorted(self.bin_pop, )


        #Calculate total fitness across all bins
        total_fitness = sum(obj.fitness for obj in self.bin_pop)
        # Calculate deletion probabilities for each object
        deletion_probabilities = [obj.fitness / total_fitness for obj in self.bin_pop]
        selected_index = random.choices(range(len(self.bin_pop)), weights=deletion_probabilities)[0]

        parent_list[currentCount] = max(self.bin_pop.head(tSize), key=lambda x: x.fitness)
        sorted(self.bin_scores.items(), key=lambda item: item[1], reverse=True


    def report_pop(self):
        pop_df = pd.DataFrame([vars(instance) for instance in self.bin_pop])
        print(pop_df)