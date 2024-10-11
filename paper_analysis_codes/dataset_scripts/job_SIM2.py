import os
import sys
import argparse
sys.path.append('/project/kamoun_shared/code_shared/sim-study-harsh/')
from src.skfibers.experiments.SIM2 import survival_data_simulation_covariates #SOURCE CODE RUN
#from skfibers.experiments.survival_sim_simple import survival_data_simulation #PIP INSTALL CODE RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--o', dest='data_path', help='', type=str, default = 'myDataPath') #full path/filename
    parser.add_argument('--i', dest='instance', help='number of instances', type=int, default = 10000) #output folder name
    parser.add_argument('--p', dest='pred_feature', help='number of predictive features', type=int, default = 10) #output folder name
    parser.add_argument('--tf', dest='total_feature', help='total number of features', type=int, default=100)
    parser.add_argument('--l', dest='exp_name', help='experiment name dataset label', type=str, default='Sim')
    parser.add_argument('--c', dest='censor', help='censoring frequency in dataset', type=float, default=0.2)
    parser.add_argument('--n', dest='noise', help='proportion of noise in the dataset', type=float, default=0.0)
    parser.add_argument('--nc', dest='nc', help='whether to make this datasets a negative control', type=str, default = 'False') #full path/filename
    options=parser.parse_args(argv[1:])

    data_path = options.data_path
    instance = options.instance
    pred_feature = options.pred_feature
    if options.nc == 'True':
        nc = True
    else:
        nc = False
    nc = False
    noise = options.noise
    total_feature = options.total_feature
    censor = options.censor
    exp_name = options.exp_name
    threshold = 'NA'

    data_name = exp_name+'_i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_t_'+str(threshold)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)


    #Generate Example Simulated Dataset --------------------------------------------
    full_data_name_path = data_path +'/'+data_name+'.csv'

    print('Simulating Dataset')
    # data = survival_data_simulation(instances=instance, total_features=total_feature, predictive_features=pred_feature, low_risk_proportion=0.5, threshold=threshold, feature_frequency_range=(0.1, 0.4), 
    #         noise_frequency=noise, class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2), censoring_frequency=censor, 
    #         negative_control=nc, random_seed=42)
    _, data = survival_data_simulation_covariates(instances=instance,total_features=total_feature,
                                                           predictive_features=pred_feature,feature_frequency_range=(0.1, 0.4),noise_frequency=noise,censoring_frequency=censor,negative_control=nc,random_seed=42)
    data.to_csv(full_data_name_path, index=False)
    print('Dataset Simulation Complete')

if __name__=="__main__":
    sys.exit(main(sys.argv))
