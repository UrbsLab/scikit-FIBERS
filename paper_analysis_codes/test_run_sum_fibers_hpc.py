import os
import time

#https://github.com/UrbsLab/scikit-FIBERS/blob/ppsn/New%20FIBERS%20Experiment%20-%20All%20FIBERS2%20Exp.ipynb

class SumFIBERS:
    def __init__(self):
        #Run Parameters
        self.data_path = '/project/kamoun_shared/ryanurb/data'
        self.write_path = '/project/kamoun_shared/ryanurb/'
        self.data_name = 'example_sim1'
        self.output_folder = 'sim_test1'
        self.run_cluster = 'LSF' #LSF or SLURM
        self.random_seeds = 10
        self.reserved_memory = 4
        self.queue = 'i2c2_normal'
        self.dataset = self.data_path+'/'+self.data_name+'.csv'

        #Folder Management------------------------------
        #Main Write Path-----------------
        if not os.path.exists(self.write_path):
            os.mkdir(self.write_path)  
        #Output Path--------------------
        if not os.path.exists(self.write_path+'output/'):
            os.mkdir(self.write_path+'output/') 
        #Output Folder ------------------
        self.outputPath = self.write_path+'output/'+self.output_folder
        if not os.path.exists(self.outputPath):
            os.mkdir(self.outputPath) 
        #Scratch Path-------------------- 
        self.scratchPath = self.write_path+'scratch/'
        if not os.path.exists(self.scratchPath):
            os.mkdir(self.scratchPath) 
        #LogFile Path--------------------
        self.logPath = self.write_path+'logs/'
        if not os.path.exists(self.logPath):
            os.mkdir(self.logPath) 

        #Apply FIBERS multiple times with different random seeds. 
        jobCount = 0
        if self.run_cluster == 'LSF':
            submit_lsf_cluster_job(self)
            jobCount +=1
        elif self.run_cluster == 'SLURM':
            submit_slurm_cluster_job(self)
            jobCount +=1
        else:
            print('ERROR: Cluster type not found')
        print(str(jobCount)+' jobs submitted successfully')

    #load and process datasets
    
def submit_slurm_cluster_job(self): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'FIBERS_'+self.data_name+'_' +'sum'+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + self.queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python test_job_sum_fibers_hpc.py'+' --d '+ self.dataset +' --o '+self.outputPath +' --r '+ str(self.random_seeds) + '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)

def submit_lsf_cluster_job(self): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'FIBERS_'+self.data_name+'_' +'sum'+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + self.queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('python test_job_sum_fibers_hpc.py'+' --d '+ self.dataset +' --o '+self.outputPath +' --r '+ str(self.random_seeds) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

run_obj = SumFIBERS()
