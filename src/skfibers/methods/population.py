
import pandas as pd
from .bin import BIN

class BIN_SET:
    def __init__(self,manual_bin_init,feature_df,outcome_df,censor_df,feature_names,pop_size,min_bin_size,max_bin_init_size,
                 group_thresh,min_thresh,max_thresh,int_thresh,outcome_type,fitness_metric,pareto_fitness,group_strata_min,
                 outcome_label,censor_label,threshold_evolving,penalty,random_seed):
        self.bin_pop = []

        #Initialize bin population
        if manual_bin_init != None:
            # Load manually curated or previously trained bin population
            print("manual bin intitialization not yet implemented")
        else:
            #Random bin initialization
            while len(self.bin_pop) < pop_size:
                new_bin = BIN()
                new_bin.initialize_random(feature_names,min_bin_size,max_bin_init_size,group_thresh,random_seed)
                new_bin.evaluate(feature_df,outcome_df,censor_df,outcome_type,fitness_metric,outcome_label,
                                 censor_label,min_thresh,max_thresh,int_thresh,group_thresh,threshold_evolving)
                new_bin.calculate_fitness(pareto_fitness,group_strata_min,penalty)

                self.bin_pop.append(new_bin)


    def report_pop(self):
        pop_df = pd.DataFrame([vars(instance) for instance in self.bin_pop])
        print(pop_df)