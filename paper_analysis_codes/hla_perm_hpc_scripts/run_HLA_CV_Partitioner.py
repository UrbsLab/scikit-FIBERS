import os
import shlex
import subprocess
import sys
import time
import argparse

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='')
    #Script Parameters
    parser.add_argument('--d', dest='datafile', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--cv', dest='partitions', help='number of cv partitions', type=int, default= 10)
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str.upper, choices=('LSF', 'SLURM'), default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 4)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    parser.add_argument('--dry-run', action='store_true', help='Write job files without submitting')
    parser.add_argument('--python-executable', default='python', help='Compute-node Python')

    options=parser.parse_args(argv[1:])

    datafile= os.path.abspath(options.datafile)
    writepath = os.path.abspath(options.writepath) + os.sep
    outputfolder = os.path.abspath(options.outputfolder)
    partitions = options.partitions
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

    outputfolder = outputfolder+'_'+str(partitions)
    #Main Write Path-----------------
    if not os.path.exists(writepath):
        os.makedirs(writepath)
    #Output Path--------------------
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    #Scratch Path--------------------
    scratchPath = writepath+'scratch'
    if not os.path.exists(scratchPath):
        os.makedirs(scratchPath)
    #LogFile Path--------------------
    logPath = writepath+'logs'
    if not os.path.exists(logPath):
        os.makedirs(logPath)

    if run_cluster == 'LSF':
        submit_lsf_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=options.dry_run,python_executable=options.python_executable)
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=options.dry_run,python_executable=options.python_executable)
    else:
        print('ERROR: Cluster type not found')
    print(str(1)+(' jobs prepared' if options.dry_run else ' jobs submitted successfully'))

#legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
def submit_slurm_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=False,python_executable='python'):
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('set -euo pipefail\nexport MPLBACKEND=Agg\n')
    sh_file.write(('srun '+shlex.quote(python_executable)+' '+shlex.quote(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_HLA_CV_Partitioner.py'))+' --d '+shlex.quote(str(datafile))+' --o '+shlex.quote(str(outputfolder))+' --cv '+str(partitions)+ '\n').rstrip('\n') + '\n')
    sh_file.close()
    if dry_run:
        print('Prepared: '+job_path)
    else:
        subprocess.run(['sbatch', job_path], check=True)


#UPENN - Legacy mode (using shell file) - memory on head node
def submit_lsf_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=False,python_executable='python'):
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    sh_file.write('#BSUB -J ' + job_name + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + '.e\n')
    sh_file.write('set -euo pipefail\nexport MPLBACKEND=Agg\n')
    sh_file.write((''+shlex.quote(python_executable)+' '+shlex.quote(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_HLA_CV_Partitioner.py'))+' --d '+shlex.quote(str(datafile))+' --o '+shlex.quote(str(outputfolder))+' --cv '+str(partitions)+ '\n').rstrip('\n') + '\n')
    sh_file.close()
    if dry_run:
        print('Prepared: '+job_path)
    else:
        with open(job_path) as job_file:
            subprocess.run(['bsub'], stdin=job_file, check=True)

if __name__=="__main__":
    sys.exit(main(sys.argv))
