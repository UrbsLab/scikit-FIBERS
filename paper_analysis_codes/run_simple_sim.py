import os
import time
from src.skfibers.experiments.survival_sim_simple import survival_data_simulation #SOURCE CODE RUN
#from skfibers.experiments.survival_sim_simple import survival_data_simulation #PIP INSTALL CODE RUN

#CLUSTER NOTES:
# module load git
# git clone --single-branch --branch dev https://github.com/UrbsLab/scikit-FIBERS

class RunFIBERS:
    def __init__(self):
        #Run Parameters
        self.write_path = '/project/kamoun_shared/ryanurb/'
        self.data_path = '/project/kamoun_shared/ryanurb/data/simple_sim_datasets'
        self.run_cluster = 'LSF' #LSF or SLURM
        self.reserved_memory = 4
        self.queue = 'i2c2_normal'

        #Baseline Parameters

        self.censoring = [0,0.2,0.5,0.8]
        self.instances = [500,1000,10000]
        self.pred_features = [5,10,20]
        self.ncs = ['False','True']
        self.noises = [0,0.1,0.2,0.3,0.4,0.5]
        self.total_features = [100,200,500,1000]
        self.thresholds = [0,1,2,3,4,5]

        #Folder Management------------------------------
        #Main Write Path-----------------
        if not os.path.exists(self.write_path):
            os.mkdir(self.write_path)  
        #Data Folder ------------------
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path) 
        #Scratch Path-------------------- 
        self.scratchPath = self.write_path+'scratch/'
        if not os.path.exists(self.scratchPath):
            os.mkdir(self.scratchPath) 
        #LogFile Path--------------------
        self.logPath = self.write_path+'logs/'
        if not os.path.exists(self.logPath):
            os.mkdir(self.logPath) 

        jobCount = 0
        #Censoring Assessment
        instance = 10000
        pred_feature = self.pred_features[0]
        nc = self.ncs[0]
        noise = self.noises[0]
        total_feature = self.total_features[0]
        threshold = self.thresholds[0]
        for censor in self.censoring:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        #Initial Negative Control
        nc = self.ncs[1] #True
        censor = 0.2
        if self.run_cluster == 'LSF':
            submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
            jobCount +=1
        elif self.run_cluster == 'SLURM':
            submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
            jobCount +=1
        else:
            print('ERROR: Cluster type not found')

        # Instances Assessment
        nc = self.ncs[0] #False
        for instance in self.instances:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')
        
        # Predictive Features Assessment
        instance = 10000
        for pred_feature in self.pred_features:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Basic Noise Assessment
        pred_feature = 10
        for noise in self.noises:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Basic Total Features Assessment
        noise = 0
        for total_feature in self.total_features:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Noisy Total Features Assessment
        noise = 0.2
        for total_feature in self.total_features:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Basic Thresholds Assessment
        noise = 0
        total_feature = 100
        pred_feature = 20
        for threshold in self.thresholds:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Noisy Thresholds Assessment
        noise = 0.2
        for threshold in self.thresholds:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')


        print(str(jobCount)+' jobs submitted successfully')

    
def submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'FIBERS_data_sim_'+'i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_t_'+str(threshold)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + self.queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_simple_sim.py' +' --o '+self.data_path+' --i '+str(instance)+' --p '+ str(pred_feature) +' --nc '+str(nc) +' --n '+str(noise) +' --tf '+str(total_feature)+' --t '+str(threshold)+' --c '+str(censor)+ '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,threshold,censor): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'FIBERS_data_sim_'+'i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_t_'+str(threshold)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + self.queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_simple_sim.py'+' --o '+self.data_path+' --i '+str(instance)+' --p '+ str(pred_feature) +' --nc '+str(nc) +' --n '+str(noise) +' --tf '+str(total_feature)+' --t '+str(threshold)+' --c '+str(censor)+ '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)


run_obj = RunFIBERS()
