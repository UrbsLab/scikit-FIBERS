import os
import shlex
import subprocess
import sys
import time
import argparse

def main(argv):
    #ARGUMENTS:------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Submit RRS survival CV partition jobs')
    #Script Parameters
    parser.add_argument('--d', dest='datafile', help='name of data file (REQUIRED)', type=str, default = 'myData') #output folder name
    parser.add_argument('--w', dest='writepath', help='', type=str, default = 'myWritePath') #full path/filename
    parser.add_argument('--o', dest='outputfolder', help='directory path to write output (default=CWD)', type=str, default = 'myOutput') #full path/filename
    parser.add_argument('--cv', dest='partitions', help='number of cv partitions', type=int, default= 10)
    parser.add_argument('--rc', dest='run_cluster', help='cluster type', type=str.upper, choices=('LSF', 'SLURM'), default='LSF')
    parser.add_argument('--rm', dest='reserved_memory', help='reserved memory for job', type=int, default= 16)
    parser.add_argument('--q', dest='queue', help='cluster queue name', type=str, default= 'i2c2_normal')
    parser.add_argument('--dry-run', action='store_true', help='Write job files without submitting')
    parser.add_argument('--python-executable', default='python', help='Compute-node Python')
    parser.add_argument('--event-col', '--c', default='grf_fail', help='0=censored, 1=event')
    parser.add_argument('--time-col', '--ol', default='graftyrs', help='Observed event/censoring follow-up time')
    parser.add_argument('--seed', type=int, default=42, help='Partition seed (default: 42)')

    parser.add_argument('--parallel-folds', action='store_true', help='Submit one array task per CV fold')
    parser.add_argument('--max-concurrent-folds', type=int, default=10,
                        help='Maximum simultaneous fold jobs with --parallel-folds (default: 10)')

    options=parser.parse_args(argv[1:])
    if options.partitions < 2:
        parser.error('--cv must be at least 2')
    if options.max_concurrent_folds < 1:
        parser.error('--max-concurrent-folds must be positive')
    if not 0 <= options.seed < 2 ** 32:
        parser.error('--seed must be from 0 through 4294967295')
    if options.event_col == options.time_col:
        parser.error('--event-col and --time-col must be different columns')
    partition_options = ['--event-col', options.event_col, '--time-col', options.time_col,
                         '--seed', str(options.seed)]

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
        submit_lsf_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=options.dry_run,python_executable=options.python_executable,parallel_folds=options.parallel_folds,max_concurrent_folds=options.max_concurrent_folds,partition_options=partition_options)
    elif run_cluster == 'SLURM':
        submit_slurm_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=options.dry_run,python_executable=options.python_executable,parallel_folds=options.parallel_folds,max_concurrent_folds=options.max_concurrent_folds,partition_options=partition_options)
    else:
        print('ERROR: Cluster type not found')
    count = partitions if options.parallel_folds else 1
    print(str(count)+(' jobs prepared' if options.dry_run else ' jobs submitted successfully')
          + (' in one job array' if options.parallel_folds else ''))

#legacy mode just for cedars (no head node) note cedars has a different hpc - we'd need to write a method for (this is the more recent one)
def submit_slurm_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=False,python_executable='python',parallel_folds=False,max_concurrent_folds=10,partition_options=()):
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH -p ' + queue + '\n')
    sh_file.write('#SBATCH --job-name=' + job_name + '\n')
    if parallel_folds:
        sh_file.write(f'#SBATCH --array=1-{partitions}%{min(partitions, max_concurrent_folds)}\n')
    sh_file.write('#SBATCH --mem=' + str(reserved_memory) + 'G' + '\n')
    # sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#SBATCH -o ' + logPath+'/'+job_name + ('_%A_%a' if parallel_folds else '') + '.o\n')
    sh_file.write('#SBATCH -e ' + logPath+'/'+job_name + ('_%A_%a' if parallel_folds else '') + '.e\n')
    sh_file.write('set -euo pipefail\nexport MPLBACKEND=Agg\n')
    command = [python_executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_HLA_CV_Partitioner.py'),
               '--d', str(datafile), '--o', str(outputfolder), '--cv', str(partitions), *partition_options]
    sh_file.write('srun ' + shlex.join(command) + (' --fold "${SLURM_ARRAY_TASK_ID}"' if parallel_folds else '') + '\n')
    sh_file.close()
    if dry_run:
        print('Prepared: '+job_path)
    else:
        subprocess.run(['sbatch', job_path], check=True)


#UPENN - Legacy mode (using shell file) - memory on head node
def submit_lsf_cluster_job(datafile,outputfolder,logPath,scratchPath,reserved_memory,queue,partitions,dry_run=False,python_executable='python',parallel_folds=False,max_concurrent_folds=10,partition_options=()):
    job_ref = str(time.time())
    job_name = 'DataCV_'+job_ref
    job_path = scratchPath+'/'+job_name+ '_run.sh'
    sh_file = open(job_path, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q ' + queue + '\n')
    array_name = f'{job_name}[1-{partitions}]%{min(partitions, max_concurrent_folds)}' if parallel_folds else job_name
    sh_file.write('#BSUB -J ' + shlex.quote(array_name) + '\n')
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"' + '\n')
    sh_file.write('#BSUB -M ' + str(reserved_memory) + 'GB' + '\n')
    sh_file.write('#BSUB -o ' + logPath+'/'+job_name + ('_%J_%I' if parallel_folds else '') + '.o\n')
    sh_file.write('#BSUB -e ' + logPath+'/'+job_name + ('_%J_%I' if parallel_folds else '') + '.e\n')
    sh_file.write('set -euo pipefail\nexport MPLBACKEND=Agg\n')
    command = [python_executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_HLA_CV_Partitioner.py'),
               '--d', str(datafile), '--o', str(outputfolder), '--cv', str(partitions), *partition_options]
    sh_file.write(shlex.join(command) + (' --fold "${LSB_JOBINDEX}"' if parallel_folds else '') + '\n')
    sh_file.close()
    if dry_run:
        print('Prepared: '+job_path)
    else:
        with open(job_path) as job_file:
            subprocess.run(['bsub'], stdin=job_file, check=True)

if __name__=="__main__":
    sys.exit(main(sys.argv))
