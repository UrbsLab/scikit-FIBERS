import os
import sys
import time
import argparse

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='Duration')  
    parser.add_argument('--i', dest='iterations', help='iterations', type=int, default=100)
    parser.add_argument('--ps', dest='pop_size', help='population size', type=int, default=50)
    parser.add_argument('--cp', dest='crossover_prob', help='crossover probability', type=float, default=0.5)
    parser.add_argument('--mup', dest='mutation_prob', help='mutation probability', type=float, default=0.5)
    parser.add_argument('--mp', dest='merge_prob', help='merge probability', type=float, default=0.1)
    parser.add_argument('--e', dest='elitism', help='elite proportion of population protected from deletion', type=float, default=0.1)
    parser.add_argument('--bi', dest='min_features_per_group', help='mininum features in a bin', type=int, default=2)
    parser.add_argument('--ba', dest='max_number_of_groups_with_feature', help='maximum number of bin with said features', type=int, default=2)
    # parser.add_argument('--ib', dest='max_bin_init_size', help='maximum bin intitilize size', type=int, default=10)
    parser.add_argument('--f', dest='fitness_metric', help='fitness metric', type=str, default='log_rank')
    parser.add_argument('--c', dest='censor_label', help='censor column label', type=str, default='Censoring')
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.2)
    parser.add_argument('--t', dest='group_thresh', help='group threshold', type=int, default=0)
    parser.add_argument('--it', dest='min_thresh', help='minimum threshold', type=int, default=0)
    parser.add_argument('--at', dest='max_thresh', help='maximum threshold', type=int, default=5)
    parser.add_argument('--ms', dest='mutation_strategy', help='mutation strategy (Regular or Simplifed)', type=str, default="Regular")
    parser.add_argument('--te', dest='thresh_evolve_prob', help='threshold evolution probability', type=float, default=0.5)

    options=parser.parse_args(argv[1:])

    datafolder= options.datafolder
    writepath = options.writepath
    outputfolder = options.outputfolder
    manual_bin_init = options.manual_bin_init
    run_cluster = options.run_cluster
    random_seeds = options.random_seeds
    reserved_memory = options.reserved_memory
    queue = options.queue
    outcome_label = options.outcome_label
    iterations = options.iterations
    pop_size = options.pop_size
    crossover_prob = options.crossover_prob
    mutation_prob = options.mutation_prob 
    merge_prob = options.merge_prob
    elitism = options.elitism
    min_features_per_group = options.min_features_per_group
    max_number_of_groups_with_feature = options.max_number_of_groups_with_feature
    fitness_metric = options.fitness_metric
    censor_label = options.censor_label
    group_strata_min = options.group_strata_min
    group_thresh = options.group_thresh
    min_thresh = options.min_thresh 
    max_thresh = options.max_thresh 
    thresh_evolve_prob = options.thresh_evolve_prob
    algorithm = 'FibersAT' #hard coded here

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    outputPath = writepath+'output/'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'scratch'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'logs'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 
    outputPath = outputPath+algorithm+'_'+outputfolder
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 

    jobCount = 0
    #For each simulated dataset
    for dataname in os.listdir(datafolder):
        if os.path.isfile(os.path.join(datafolder, dataname)):
            datapath = os.path.join(datafolder, dataname)
        data_name = os.path.splitext(dataname)[0]
        #Make output subfolder (to contain all random seed runs)
        if not os.path.exists(outputPath+'/'+data_name):
            os.mkdir(outputPath+'/'+data_name) 
        outputpath = outputPath+'/'+data_name #random seed run output saved in separate output folders named based on simulated dataset
        #For each random seed
        for seed in range(0,random_seeds):
            if run_cluster == 'LSF':
                submit_lsf_cluster_job(scratchPath,logPath,data_name,datapath,outputpath,manual_bin_init,reserved_memory,queue,outcome_label,
                                       iterations,pop_size,crossover_prob,mutation_prob,merge_prob,elitism,
                                       min_features_per_group,max_number_of_groups_with_feature,fitness_metric,censor_label,
                                       group_strata_min,group_thresh,min_thresh,max_thresh,thresh_evolve_prob,seed)
                jobCount +=1
            elif run_cluster == 'SLURM':
                submit_slurm_cluster_job(scratchPath,logPath,data_name,datapath,outputpath,manual_bin_init,reserved_memory,queue,outcome_label,
                                       iterations,pop_size,crossover_prob,mutation_prob,merge_prob,elitism,
                                       min_features_per_group,max_number_of_groups_with_feature,fitness_metric,censor_label,
                                       group_strata_min,group_thresh,min_thresh,max_thresh,thresh_evolve_prob,seed)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')
    print(str(jobCount)+' jobs submitted successfully')

#legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
def submit_slurm_cluster_job(scratchPath,logPath,data_name,datapath,outputpath,manual_bin_init,reserved_memory,queue,outcome_label,
                                       iterations,pop_size,crossover_prob,mutation_prob,merge_prob,elitism,
                                       min_features_per_group,max_number_of_groups_with_feature,fitness_metric,censor_label,
                                       group_strata_min,group_thresh,min_thresh,max_thresh,thresh_evolve_prob,random_seed): 
    job_ref = str(time.time())
    job_name = 'FIBERS_'+data_name+'_' +str(random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_sim_fibers_hpc.py'+' --d '+str(datapath)+' --o '+str(outputpath)+' --pi '+str(manual_bin_init)
        +' --ol '+str(outcome_label)
        +' --i '+str(iterations)+' --ps '+str(pop_size)+' --cp '+str(crossover_prob)+' --mup '+str(mutation_prob)
        +' --mp '+str(merge_prob)+' --e '+str(elitism)
        +' --bi '+str(min_features_per_group)+' --ba '+str(max_number_of_groups_with_feature)+' --f '+str(fitness_metric)
        +' --c '+str(censor_label)+' --g '+str(group_strata_min)+' --t '+str(group_thresh)+' --it '+str(min_thresh)+' --at '+str(max_thresh)
        +' --te '+str(thresh_evolve_prob)+' --r '+str(random_seed)+'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

#UPENN - Legacy mode (using shell file) - memory on head node
def submit_lsf_cluster_job(scratchPath,logPath,data_name,datapath,outputpath,manual_bin_init,reserved_memory,queue,outcome_label,
                                       iterations,pop_size,crossover_prob,mutation_prob,merge_prob,elitism,
                                       min_features_per_group,max_number_of_groups_with_feature,fitness_metric,censor_label,
                                       group_strata_min,group_thresh,min_thresh,max_thresh,thresh_evolve_prob,random_seed): 
    job_ref = str(time.time())
    job_name = 'FIBERS_'+data_name+'_' +str(random_seed)+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_sim_fibers_hpc.py'+' --d '+str(datapath)+' --o '+str(outputpath)+' --pi '+str(manual_bin_init)
        +' --ol '+str(outcome_label)
        +' --i '+str(iterations)+' --ps '+str(pop_size)+' --cp '+str(crossover_prob)+' --mup '+str(mutation_prob)
        +' --mp '+str(merge_prob)+' --e '+str(elitism)
        +' --bi '+str(min_features_per_group)+' --ba '+str(max_number_of_groups_with_feature)+' --f '+str(fitness_metric)
        +' --c '+str(censor_label)+' --g '+str(group_strata_min)+' --t '+str(group_thresh)+' --it '+str(min_thresh)+' --at '+str(max_thresh)
        +' --te '+str(thresh_evolve_prob)+' --r '+str(random_seed)+'\n')
    print('python job_sim_fibers_hpc.py'+' --d '+str(datapath)+' --o '+str(outputpath)+' --pi '+str(manual_bin_init)
        +' --ol '+str(outcome_label)
        +' --i '+str(iterations)+' --ps '+str(pop_size)+' --cp '+str(crossover_prob)+' --mup '+str(mutation_prob)
        +' --mp '+str(merge_prob)+' --e '+str(elitism)
        +' --bi '+str(min_features_per_group)+' --ba '+str(max_number_of_groups_with_feature)+' --f '+str(fitness_metric)
        +' --c '+str(censor_label)+' --g '+str(group_strata_min)+' --t '+str(group_thresh)+' --it '+str(min_thresh)+' --at '+str(max_thresh)
        +' --te '+str(thresh_evolve_prob)+' --r '+str(random_seed)+'\n')
    sh_file.close()
    # os.system('bsub < ' + job_path)


if __name__=="__main__":
    sys.exit(main(sys.argv))