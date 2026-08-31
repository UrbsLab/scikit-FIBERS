import os
import time
import argparse
import sys

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    parser.add_argument('--cv', dest='cv', help='cross validation partitions', type=int, default= 10)
    options=parser.parse_args(argv[1:])

    writepath = options.writepath
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue
    cv = options.cv

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    outputpath = writepath+'output/'
    if not os.path.exists(outputpath):
        os.mkdir(outputpath) 
    #Scratch Path-------------------- 
    scratchPath = writepath+'scratch/'
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath) 
    #LogFile Path--------------------
    logPath = writepath+'logs/'
    if not os.path.exists(logPath):
        os.mkdir(logPath) 

    jobCount = 0
    if run_cluster == 'LSF':
        submit_lsf_cluster_job(scratchPath,logPath,outputpath,cv,reserved_memory,queue)
        jobCount +=1
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(scratchPath,logPath,outputpath,cv,reserved_memory,queue)
        jobCount +=1
    else:
        print('ERROR: Cluster type not found')

    print(str(jobCount)+' job submitted successfully')

    
def submit_slurm_cluster_job(scratchPath,logPath,outputpath,cv,reserved_memory,queue): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'Sum_FIBERS_CV_'+'sum'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_sum_fibers_hpc_cv.py' +' --o '+outputpath +' --cv '+ str(cv) + '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(scratchPath,logPath,outputpath,cv,reserved_memory,queue): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'Sum_FIBERS_CV_'+'sum'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_sum_fibers_hpc_cv.py'+' --o '+outputpath +' --cv '+ str(cv) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)


if __name__=="__main__":
    sys.exit(main(sys.argv))
