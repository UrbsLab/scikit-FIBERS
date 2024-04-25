import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path


def main(argv):

    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    
    #Script Parameters
    parser.add_argument('--d', dest='dataset', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--pi', dest='manual_bin_init', help='directory path to population initialization file', type=str, default = 'None') #full path/filename
    parser.add_argument('--da', dest='data_process', help='boolean determining whether to process data using this script', type=str, default='True')

    #FIBERS Parameters
    parser.add_argument('--ol', dest='outcome_label', help='outcome column label', type=str, default='Duration')  
    parser.add_argument('--ot', dest='outcome_type', help='outcome type', type=str, default='survival')
    parser.add_argument('--i', dest='iterations', help='iterations', type=int, default=100)
    parser.add_argument('--ps', dest='pop_size', help='population size', type=int, default=50)
    parser.add_argument('--tp', dest='tournament_prop', help='trournament probability', type=float, default=0.2)
    parser.add_argument('--cp', dest='crossover_prob', help='crossover probability', type=float, default=0.5)
    parser.add_argument('--mi', dest='min_mutation_prob', help='minimum mutation probability', type=float, default=0.1)
    parser.add_argument('--ma', dest='max_mutation_prob', help='maximum mutation probability', type=float, default=0.5)
    parser.add_argument('--mp', dest='merge_prob', help='merge probability', type=float, default=0.1)
    parser.add_argument('--ng', dest='new_gen', help='proportion of max population used to deterimine offspring population size', type=float, default=1.0)
    parser.add_argument('--e', dest='elitism', help='elite proportion of population protected from deletion', type=float, default=0.1)
    parser.add_argument('--d', dest='diversity_pressure', help='diversity pressure (K in k-means)', type=int, default=0)
    parser.add_argument('--bi', dest='min_bin_size', help='minimum bin size', type=int, default=1)
    parser.add_argument('--ba', dest='max_bin_size', help='maximum bin size', type=str, default='None')
    parser.add_argument('--ib', dest='max_bin_init_size', help='maximum bin intitilize size', type=int, default=10)
    parser.add_argument('--f', dest='fitness_metric', help='fitness metric', type=str, default='log_rank')
    parser.add_argument('--we', dest='log_rank_weighting', help='log-rank test weighting', type=str, default='None')
    parser.add_argument('--c', dest='censor_label', help='censor column label', type=str, default='Censoring')
    parser.add_argument('--g', dest='group_strata_min', help='group strata minimum', type=float, default=0.2)
    parser.add_argument('--p', dest='penalty', help='group strata min penalty', type=float, default=0.5)
    parser.add_argument('--t', dest='group_thresh', help='group threshold', type=str, default=0)
    parser.add_argument('--it', dest='min_thresh', help='minimum threshold', type=int, default=0)
    parser.add_argument('--at', dest='max_thresh', help='maximum threshold', type=int, default=5)
    #int_thresh
    parser.add_argument('--te', dest='thresh_evolve_prob', help='threshold evolution probability', type=float, default=0.5)
    parser.add_argument('--cl', dest='pop_clean', help='clean population', type=str, default='None')
    parser.add_argument('--r', dest='random_seed', help='random seed', type=str, default='None')
    options=parser.parse_args(argv[1:])

    dataset = options.dataset
    writepath = options.writepath
    outputfolder = options.outputfolder
    if options.manual_bin_init == 'None':
        manual_bin_init = None
    else:
        manual_bin_init = pd.read_csv(manual_bin_init,low_memory=False)

    if options.data_process == 'True':
        data_process = True
    else:
        data_process = False 
    outcome_label = options.outcome_label
    outcome_type = options.outcome_type
    iterations = options.iterations
    pop_size = options.pop_size
    tournament_prop = options.tournament_prop
    crossover_prob = options.crossover_prob
    min_mutation_prob = options.min_mutation_prob 
    max_mutation_prob = options.max_mutation_prob
    merge_prob = options.merge_prob
    new_gen = options.new_gen
    elitism = options.elitism
    diversity_pressure = options.diversity_pressure
    min_bin_size = options.min_bin_size
    if options.max_bin_size == 'None':
        max_bin_size = None
    else:
        max_bin_size = int(options.max_bin_size)
    max_bin_init_size = options.max_bin_init_size
    fitness_metric = options.fitness_metric
    if options.log_rank_weighting == 'None':
        log_rank_weighting = None
    else:
        log_rank_weighting = str(options.log_rank_weighting)
    censor_label = options.censor_label
    group_strata_min = options.group_strata_min
    penalty = options.penalty
    if options.group_thresh == 'None':
        group_thresh = None
    else:
        group_thresh = str(options.group_thresh)
    min_thresh = options.min_thresh 
    max_thresh = options.max_thresh 
    #int_thresh = options.int_thresh
    thresh_evolve_prob = options.thresh_evolve_prob
    covariates = None #Manually included in script
    if options.pop_clean == 'None':
        pop_clean = None
    else:
        pop_clean = str(options.pop_clean)
    pop_clean = options.pop_clean
    random_seed = options.random_seed

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    outputPath = writepath+'output/'
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'scratch/'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'logs/'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 
    outputPath = outputPath+outputfolder
    if not os.path.exists(outputPath):
        os.mkdir(outputPath) 

    #load and process datasets

    #set up analysis of single dataset as a method that is called in the job submission below


    """ Submit Job to the cluster. """
    #MAKE CLUSTER JOBS###################################################################
    jobName = scratchPath+'ReBATE_'+dataName+'_'+str(algorithm)+'_'+str(discretelimit)+'_'+str(neighbors)+'_'+str(topattr)+'_'+str(turflimit)+'_'+str(time.time())+'_run.sh'                                                  
    shFile = open(jobName, 'w')
    shFile.write('#!/bin/bash\n')
    shFile.write('#BSUB -J '+dataName+'_'+str(algorithm)+'_'+str(discretelimit)+'_'+str(neighbors)+'_'+str(topattr)+'_'+str(turflimit)+'_'+str(time.time())+'\n')
    #shFile.write('#BSUB -M 45000'+'\n')
    shFile.write('#BSUB -o ' + logPath+'ReBATE_'+dataName+'_'+str(algorithm)+'_'+str(discretelimit)+'_'+str(neighbors)+'_'+str(topattr)+'_'+str(turflimit)+'_'+str(time.time())+'.o\n')
    shFile.write('#BSUB -e ' + logPath+'ReBATE_'+dataName+'_'+str(algorithm)+'_'+str(discretelimit)+'_'+str(neighbors)+'_'+str(topattr)+'_'+str(turflimit)+'_'+str(time.time())+'.e\n\n')
    shFile.write('python '+rebatePath+'rebate.py '+'-f '+str(filename)+' -o '+str(outputPath)+' -a '+str(algorithm)+' -c '+str(classname)+' -d '+str(discretelimit)+' -m '+str(missingdata)+' -n '+str(neighbors)+' -s '+str(separator)+' -T '+str(topattr)+' -t '+str(turflimit)+'\n') #HARD-CODING
    shFile.close()
    os.system('bsub < '+jobName)    
    #####################################################################################  

def get_cluster_params(self): 
    cluster_params = [self.output_path, self.experiment_name, None,
                        self.outcome_label, self.outcome_type, self.instance_label, self.sig_cutoff, self.show_plots]
    cluster_params = [str(i) for i in cluster_params]
    return cluster_params
    
def submit_slurm_cluster_job(self): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = self.output_path + '/' + self.experiment_name + '/jobs/P1_' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + self.queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_ref + '\n')
    sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write(
        '#SBATCH -o ' + self.output_path + '/' + self.experiment_name +
        '/logs/P7_' + job_ref + '.o\n')
    sh_file.write(
        '#SBATCH -e ' + self.output_path + '/' + self.experiment_name +
        '/logs/P7_' + job_ref + '.e\n')

    file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/CompareJobSubmit.py'
    cluster_params = self.get_cluster_params()
    command = ' '.join(['srun', 'python', file_path] + cluster_params)
    sh_file.write(command + '\n')
    sh_file.close()
    os.system('sbatch ' + job_name)


def submit_lsf_cluster_job(self): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = self.output_path + '/' + self.experiment_name + '/jobs/P7_' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + self.queue + '\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
    sh_file.write(
        '#BSUB -o ' + self.output_path + '/' + self.experiment_name +
        '/logs/P7_' + job_ref + '.o\n')
    sh_file.write(
        '#BSUB -e ' + self.output_path + '/' + self.experiment_name +
        '/logs/P7_' + job_ref + '.e\n')

    file_path = str(Path(__file__).parent.parent.parent) + "/streamline/legacy" + '/CompareJobSubmit.py'
    cluster_params = self.get_cluster_params()
    command = ' '.join(['python', file_path] + cluster_params)
    sh_file.write(command + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)


if __name__=="__main__":
    sys.exit(main(sys.argv))