import os
import time
import argparse
import sys

# GOAl make all possible desired tables with statistical comparisons to a baseline run (selectd)
# - one job that 

#experimentfolder/datasetname/summary/datasetname_summary.csv  --target files with all 30 top bin runs. - for grabbing averages
#experimentfolder/datasetname/summary/datasetname_master_summary.csv --files to grab counts from

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')

    #Script Parameters
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--rs', dest='random_seeds', help='number of random seeds to run', type=int, default= 30)
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str, default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')

    options=parser.parse_args(argv[1:])

    writepath = options.writepath
    outputfolder = options.outputfolder
    random_seeds = options.random_seeds
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

    #Folder Management------------------------------
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.mkdir(writepath)  
    #Output Path--------------------
    if not os.path.exists(writepath+'output/'):
        os.mkdir(writepath+'output/') 
    #Output Folder ------------------
    outputpath = writepath+'output/'+outputfolder 
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

    if run_cluster == 'LSF':
        submit_lsf_cluster_job(scratchPath,logPath,writepath,outputpath,random_seeds,reserved_memory,queue)
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(scratchPath,logPath,writepath,outputpath,random_seeds,reserved_memory,queue)
    else:
        print('ERROR: Cluster type not found')

    print(str(1)+' job submitted successfully')

    
def submit_slurm_cluster_job(scratchPath,logPath,writepath,outputpath,random_seeds,reserved_memory,queue): #legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
    job_ref = str(time.time())
    job_name = 'Table_FIBERS_' +'SIM1'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('srun python job_sim1_table_maker_hpc.py'+' --w '+writepath+' --o '+outputpath +' --rs '+ str(random_seeds) + '\n')
    sh_file.close()
    os.system('sbatch ' + job_path)


def submit_lsf_cluster_job(scratchPath,logPath,writepath,outputpath,random_seeds,reserved_memory,queue): #UPENN - Legacy mode (using shell file) - memory on head node
    job_ref = str(time.time())
    job_name = 'Table_FIBERS_' +'SIM1'+'_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('python job_sim1_table_maker_hpc.py'+' --w '+writepath +' --o '+outputpath +' --rs '+ str(random_seeds) + '\n')
    sh_file.close()
    os.system('bsub < ' + job_path)


if __name__=="__main__":
    sys.exit(main(sys.argv))
