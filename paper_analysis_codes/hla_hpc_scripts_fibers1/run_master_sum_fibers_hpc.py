import os
import argparse
import sys
import time

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 10)
    parser.add_argument('--re', dest='replicates', help='number of data replicates', type=int, default= 10)
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')

    options=parser.parse_args(argv[1:])

    datafolder = options.datafolder
    writepath = options.writepath
    outputfolder = options.outputfolder
    replicates = options.replicates
    random_seeds = options.random_seeds
    loci_list = options.loci_list
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

    #Folder Management------------------------------
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
        submit_lsf_cluster_job(scratchPath,logPath,datafolder,writepath,outputfolder,replicates,random_seeds,loci_list,reserved_memory,queue)
        jobCount +=1
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(scratchPath,logPath,datafolder,writepath,outputfolder,replicates,random_seeds,loci_list,reserved_memory,queue)
        jobCount +=1
    else:
        print('ERROR: Cluster type not found')

    print(str(jobCount)+' jobs submitted successfully')

def submit_slurm_cluster_job(scratchPath,logPath,datafolder,writepath,outputfolder,replicates,random_seeds,loci_list,reserved_memory,queue): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'Master_Sum_FIBERS_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_master_sum_fibers_hpc.py'+' --d '+ datafolder +' --w '+ writepath +' --o '+outputfolder +' --rs '+ str(random_seeds)+' --rs '+ str(replicates) +' --loci-list '+ str(loci_list)+ '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(scratchPath,logPath,datafolder,writepath,outputfolder,replicates,random_seeds,loci_list,reserved_memory,queue): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'Master_Sum_FIBERS_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_master_sum_fibers_hpc.py'+' --d '+ datafolder +' --w '+ writepath +' --o '+outputfolder +' --rs '+ str(random_seeds)+' --rs '+ str(replicates) +' --loci-list '+ str(loci_list) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)


if __name__=="__main__":
    sys.exit(main(sys.argv))
