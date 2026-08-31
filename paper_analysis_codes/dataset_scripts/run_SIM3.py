import os
import time

#CLUSTER NOTES:
# module load git
# git clone --single-branch --branch dev https://github.com/UrbsLab/scikit-FIBERS

class RunFIBERS:
    def __init__(self):
        #Run Parameters
        #self.write_path = '/project/kamoun_shared/ryanurb/'
        self.write_path = '/project/kamoun_shared/output_shared/bandheyh/'
        #self.data_path = '/project/kamoun_shared/ryanurb/data/simple_sim_datasets'
        self.data_path = '/project/kamoun_shared/data_shared/SIM3'
        self.run_cluster = 'LSF' #LSF or SLURM
        self.reserved_memory = 4
        self.queue = 'i2c2_normal'

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

        #Baseline Parameters
        self.instance = 10000
        self.pred_feature = 5
        self.nc = 'False'
        self.censor = 0.2
        self.noise = 0.0
        self.total_feature = 100

        #Varied Parameters
        self.total_features = [200, 500, 1000]
        self.instances = [500,1000]
        self.noises = [0.1,0.2,0.3,0.4,0.5]

        jobCount = 0

        #Baseline Positive Control
        exp_name = 'BasePC'
        if self.run_cluster == 'LSF':
            submit_lsf_cluster_job(self,self.instance,self.pred_feature,self.nc,self.noise,self.total_feature,self.censor,exp_name)
            jobCount +=1
        elif self.run_cluster == 'SLURM':
            submit_slurm_cluster_job(self,self.instance,self.pred_feature,self.nc,self.noise,self.total_feature,self.censor,exp_name)
            jobCount +=1
        else:
            print('ERROR: Cluster type not found')

        #Baseline Negative Control
        exp_name = 'BaseNC'
        if self.run_cluster == 'LSF':
            submit_lsf_cluster_job(self,self.instance,self.pred_feature,'True',self.noise,self.total_feature,self.censor,exp_name)
            jobCount +=1
        elif self.run_cluster == 'SLURM':
            submit_slurm_cluster_job(self,self.instance,self.pred_feature,'True',self.noise,self.total_feature,self.censor,exp_name)
            jobCount +=1
        else:
            print('ERROR: Cluster type not found')

        # Total Instances Assessment
        exp_name = "Instances"
        for instance in self.instances:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,instance,self.pred_feature,self.nc,self.noise,self.total_feature,self.censor,exp_name)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,instance,self.pred_feature,self.nc,self.noise,self.total_feature,self.censor,exp_name)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Basic Total Features Assessment (no noise)
        exp_name = "Features"
        for total_feature in self.total_features:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,self.instance,self.pred_feature,self.nc,self.noise,total_feature,self.censor,exp_name)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,self.instance,self.pred_feature,self.nc,self.noise,total_feature,self.censor,exp_name)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        # Baseline Noise Assessment
        exp_name = "BaseNoise"
        for noise in self.noises:
            if self.run_cluster == 'LSF':
                submit_lsf_cluster_job(self,self.instance,self.pred_feature,self.nc,noise,self.total_feature,self.censor,exp_name)
                jobCount +=1
            elif self.run_cluster == 'SLURM':
                submit_slurm_cluster_job(self,self.instance,self.pred_feature,self.nc,noise,self.total_feature,self.censor,exp_name)
                jobCount +=1
            else:
                print('ERROR: Cluster type not found')

        print(str(jobCount)+' jobs submitted successfully')

    
def submit_slurm_cluster_job(self,instance,pred_feature,nc,noise,total_feature,censor,exp_name): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'FIBERS_data_SIM3_'+'i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + self.queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(self.reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_SIM3.py' +' --o '+self.data_path +' --i '+str(instance)+' --p '+ str(pred_feature) +' --nc '+str(nc) +' --n '+str(noise)+ ' --tf '+str(total_feature) + ' --c '+str(censor) + ' --l '+str(exp_name)+ '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(self,instance,pred_feature,nc,noise,total_feature,censor,exp_name): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'FIBERS_data_SIM3_'+'i_'+str(instance)+'_tf_'+str(total_feature)+'_p_'+str(pred_feature)+'_n_'+str(noise)+'_c_'+str(censor)+'_nc_'+str(nc)+'_'+job_ref
    job_path = self.scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + self.queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(self.reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(self.reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + self.logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + self.logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_SIM3.py'+' --o '+self.data_path+' --i '+str(instance)+' --p '+ str(pred_feature) +' --nc '+str(nc) +' --n '+str(noise)+' --tf '+str(total_feature) + ' --c '+str(censor) +' --l '+str(exp_name)+ '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)

if __name__ == "__main__":
    run_obj = RunFIBERS()
