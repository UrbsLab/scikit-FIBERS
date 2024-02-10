import random
import pandas as pd
from .bin import BIN

class BIN_SET:
    def __init__(self,manual_bin_init,feature_df,outcome_df,censor_df,feature_names,pop_size,min_bin_size,max_bin_init_size,
                 group_thresh,min_thresh,max_thresh,int_thresh,outcome_type,fitness_metric,pareto_fitness,group_strata_min,
                 outcome_label,censor_label,threshold_evolving,penalty,iterations,iteration,random_seed):
        
        #Initialize bin population
        self.bin_pop = []
        self.offspring_pop = []

        if manual_bin_init != None:
            # Load manually curated or previously trained bin population
            print("manual bin intitialization not yet implemented")
        else:
            #Random bin initialization
            while len(self.bin_pop) < pop_size:
                #print("pop init: "+str(len(self.bin_pop)))
                new_bin = BIN()

                new_bin.initialize_random(feature_names,min_bin_size,max_bin_init_size,group_thresh,min_thresh,max_thresh,iteration,random_seed)
                # Check for duplicate rules based on feature list and threshold
                while self.equivalent_bin_in_pop(new_bin): # May slow down evolutionary cycles if new bins aren't found right away
                    new_bin.random_bin(feature_names,min_bin_size,max_bin_init_size,random_seed)

                new_bin.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,
                                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving,iterations,iteration)
                
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
            parent_list[currentCount] = max(self.bin_pop[:tSize], key=lambda x: x.fitness)
            currentCount += 1
        return parent_list


    def generate_offspring(self,crossover_prob,mutation_prob,iterations,iteration,parent_list,feature_names,threshold_evolving,min_bin_size,
                           max_bin_init_size,min_thresh,max_thresh,feature_df,outcome_df,censor_df,outcome_type,fitness_metric,
                           outcome_label,censor_label,int_thresh,group_thresh,pareto_fitness,group_strata_min,penalty,random_seed):
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
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration)
        offspring_2.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,censor_label,min_thresh,max_thresh,
                             int_thresh,group_thresh,threshold_evolving,iterations,iteration)
        
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

        while len(self.bin_pop) > pop_size:
            #Calculate total fitness across all bins
            total_fitness = sum(1/obj.fitness for obj in self.bin_pop)
            # Calculate deletion probabilities for each object
            deletion_probabilities = [(1/obj.fitness) / total_fitness for obj in self.bin_pop]
            index = random.choices(range(len(self.bin_pop)), weights=deletion_probabilities)[0]
            del self.bin_pop[index]



    def bin_deletion_deterministic(self,pop_size):
        # Calculate number of bins to delete
        self.bin_pop = sorted(self.bin_pop, key=lambda x: x.fitness,reverse=True)
        self.report_pop()
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

    def report_best_bin_in_pop(self):
        best_bin = max(self.bin_pop, key=lambda x: x.fitness)
