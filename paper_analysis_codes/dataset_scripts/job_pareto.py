import os
import sys
import argparse
import numpy as np
sys.path.append('/project/kamoun_shared/code_shared/scikit-FIBERS-summer/')
from src.skfibers.experiments.survival_sim_simple_area import survival_data_area_simulation
#from skfibers.experiments.survival_sim_simple import survival_data_simulation #PIP INSTALL CODE RUN

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--o', dest='data_path', help='', type=str, default = 'myDataPath') #full path/filename
    parser.add_argument('--i', dest='instance', help='number of instances', type=int, default = 10000) #output folder name
    parser.add_argument('--p', dest='pred_feature', help='number of predictive features', type=int, default = 10) #output folder name
    parser.add_argument('--nc', dest='nc', help='whether to make this datasets a negative control', type=str, default = 'False') #full path/filename
    parser.add_argument('--n', dest='noise', help='proportion of noise in the dataset', type=float, default=0.0)
    parser.add_argument('--tf', dest='total_feature', help='total number of features', type=int, default=100)
    parser.add_argument('--sf', dest='survivability_feature', help='total number of features benefitting survivability', type=int, default=1)
    parser.add_argument('--t', dest='threshold', help='ground truth threshold of the dataset', type=int, default=0)
    parser.add_argument('--st', dest='survivability_threshold', help='ground truth threshold survivablity feature', type=int, default=0)
    parser.add_argument('--sbs', dest='sbs', help='survivability_benefit lowerbound', type=float, default=0.01)
    parser.add_argument('--sbm', dest='sbm', help='survivability_benefit upperbound', type=float, default=0.1)
    parser.add_argument('--c', dest='censor', help='censoring frequency in dataset', type=float, default=0.2)
    parser.add_argument('--cov', dest='covariates', help='covariates to sim', type=int, default=0)
    parser.add_argument('--l', dest='exp_name', help='experiment name dataset label', type=str, default='Sim')
    # :param survivability_features: total number of features benefitting survivability
    # :param survivability_benefit: percent increase in survivability for those with feature, regardless of risk group



    options=parser.parse_args(argv[1:])

    data_path = options.data_path
    instance = options.instance
    pred_feature = options.pred_feature
    if options.nc == 'True':
        nc = True
    else:
        nc = False
    noise = options.noise
    total_feature = options.total_feature
    survivability_feature = options.survivability_feature
    threshold = options.threshold
    survivability_threshold = options.survivability_threshold
    censor = options.censor
    survivability_benefit=(options.sbm, options.sbs)
    covariates = options.covariates
    exp_name = options.exp_name

    data_name = exp_name+'_i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_t_'+str(threshold)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)

    #Generate Example Simulated Dataset --------------------------------------------
    full_data_name_path = data_path +'/'+data_name+'.csv'

    print('Simulating Dataset')
    data = survival_data_area_simulation(instances=instance, total_features=total_feature, predictive_features=pred_feature, survivability_features=survivability_feature, low_risk_proportion=0.5, threshold=threshold, survivability_threshold=survivability_threshold, feature_frequency_range=(0.1, 0.4), 
                        noise_frequency=noise, class0_time_to_event_range=(1.5, 0.2), class1_time_to_event_range=(1, 0.2), survivability_benefit=survivability_benefit, censoring_frequency=censor, 
                        covariates_to_sim=covariates, covariates_signal_range=(0.2,0.4), negative_control=nc, random_seed=42)
    
    data.to_csv(full_data_name_path, index=False)
    print('Dataset Simulation Complete')

if __name__=="__main__":
    sys.exit(main(sys.argv))
