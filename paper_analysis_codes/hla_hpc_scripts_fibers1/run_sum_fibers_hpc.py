import os
import time
import argparse
import sys

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--d', dest='datafolder', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 10)
    parser.add_argument('--re', dest='replicates', help='number of data replicates', type=int, default= 10)
    parser.add_argument('--loci-list', dest='loci_list', help='loci to include', type=str, default= 'A,B,C,DRB1,DRB345,DQA1,DQB1')
    parser.add_argument('--cov-list', dest='cov_list', help='loci covariates to include',type=str, default= 'A,B,C,DRB1,DQA1,DQB1')
    parser.add_argument('--ra', dest='rare_filter', help='rare frequency used for data cleaning', type=float, default=0)


    options=parser.parse_args(argv[1:])

    datafolder= options.datafolder
    writepath = options.writepath
    outputfolder = options.outputfolder
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue
    random_seeds = options.random_seeds
    replicates = options.replicates
    loci_list = options.loci_list
    if options.cov_list == 'None':
        cov_list = 'None'
    else:
        cov_list = options.cov_list
    rare_filter = options.rare_filter

    algorithm = 'Fibers1.0' #hard coded here

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    if not os.path.exists(writepath+'output/'):
        os.mkdir(writepath+'output/') 
    #Output Folder ------------------
    outputpath = writepath+'output/'+algorithm+'_'+outputfolder 
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
    datanames = []
    for dataname in os.listdir(datafolder):
        datanames.append(dataname)
    base_name = datanames[0].split('.')[0] #remove file extension
    base_name = base_name.split('_')[0] #remove replicate number

    for replicate in range(1,replicates+1): #one indexed datasets
        datapath = datafolder+'/'+base_name+'_'+str(replicate)+'.csv'
        data_name = base_name+'_'+str(replicate)

        if run_cluster == 'LSF':
            submit_lsf_cluster_job(scratchPath,logPath,outputpath,data_name,datapath,random_seeds,reserved_memory,queue,loci_list,cov_list,rare_filter)
            jobCount +=1
        elif run_cluster == 'SLURM':
            submit_slurm_cluster_job(scratchPath,logPath,outputpath,data_name,datapath,random_seeds,reserved_memory,queue,loci_list,cov_list,rare_filter)
            jobCount +=1
        else:
            print('ERROR: Cluster type not found')

    print(str(jobCount)+' jobs submitted successfully')

    
def submit_slurm_cluster_job(scratchPath,logPath,outputpath,data_name,datapath,random_seeds,reserved_memory,queue,loci_list,cov_list, rare_filter): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'Sum_FIBERS_'+data_name+'_' +'sum'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_sum_fibers_hpc.py'+' --d '+ datapath +' --o '+outputpath +' --r '+ str(random_seeds) 
                  +' --loci-list '+str(loci_list)+' --cov-list '+str(cov_list)+' --ra '+str(rare_filter)+'\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(scratchPath,logPath,outputpath,data_name,datapath,random_seeds,reserved_memory,queue,loci_list, cov_list, rare_filter): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'Sum_FIBERS_'+data_name+'_' +'sum'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_sum_fibers_hpc.py'+' --d '+ datapath +' --o '+outputpath +' --r '+ str(random_seeds) 
                  +' --loci-list '+str(loci_list)+' --cov-list '+str(cov_list)+' --ra '+str(rare_filter)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_path)


if __name__=="__main__":
    sys.exit(main(sys.argv))
