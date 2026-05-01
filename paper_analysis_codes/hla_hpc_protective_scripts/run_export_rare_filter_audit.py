import os
import sys
import time
import argparse


def main(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cv-datafolder", dest="cv_datafolder", type=str, default="None")
    parser.add_argument("--noncv-datafile", dest="noncv_datafile", type=str, default="None")
    parser.add_argument("--save-dir", dest="save_dir", type=str, required=True)
    parser.add_argument("--ra", dest="rare_filter", type=float, default=0.1)
    parser.add_argument("--loci-list", dest="loci_list", type=str, default="A,B,C,DRB1,DRB345,DQA1,DQB1")
    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF")
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=64)
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal")

    options = parser.parse_args(argv[1:])

    cv_datafolder = None if options.cv_datafolder == "None" else options.cv_datafolder
    noncv_datafile = None if options.noncv_datafile == "None" else options.noncv_datafile
    save_dir = options.save_dir
    rare_filter = options.rare_filter
    loci_list = options.loci_list
    run_cluster = options.run_cluster
    reserved_memory = options.reserved_memory
    queue = options.queue

    if cv_datafolder is None and noncv_datafile is None:
        raise ValueError("Provide at least one of --cv-datafolder or --noncv-datafile.")

    write_root = os.path.dirname(save_dir.rstrip("/")) + "/"
    if not os.path.exists(write_root):
        os.makedirs(write_root, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    scratch_path = write_root + "scratch"
    if not os.path.exists(scratch_path):
        os.mkdir(scratch_path)

    log_path = write_root + "logs"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if run_cluster == "LSF":
        submit_lsf_cluster_job(
            scratch_path,
            log_path,
            cv_datafolder,
            noncv_datafile,
            save_dir,
            reserved_memory,
            queue,
            rare_filter,
            loci_list,
        )
    elif run_cluster == "SLURM":
        submit_slurm_cluster_job(
            scratch_path,
            log_path,
            cv_datafolder,
            noncv_datafile,
            save_dir,
            reserved_memory,
            queue,
            rare_filter,
            loci_list,
        )
    else:
        raise ValueError("Cluster type not found")

    print("1 audit job submitted successfully")


def build_command(cv_datafolder, noncv_datafile, save_dir, rare_filter, loci_list):
    command = "python job_export_rare_filter_audit.py"
    if cv_datafolder is not None:
        command += " --cv-datafolder " + str(cv_datafolder)
    if noncv_datafile is not None:
        command += " --noncv-datafile " + str(noncv_datafile)
    command += " --save-dir " + str(save_dir)
    command += " --ra " + str(rare_filter)
    command += " --loci-list " + str(loci_list)
    return command


def submit_slurm_cluster_job(
    scratch_path,
    log_path,
    cv_datafolder,
    noncv_datafile,
    save_dir,
    reserved_memory,
    queue,
    rare_filter,
    loci_list,
):
    job_ref = str(time.time())
    job_name = "RARE_AUDIT_" + job_ref
    job_path = scratch_path + "/" + job_name + "_run.sh"
    sh_file = open(job_path, "w")
    sh_file.write("#!/bin/bash\n")
    sh_file.write("#SBATCH -p " + queue + "\n")
    sh_file.write("#SBATCH --job-name=" + job_name + "\n")
    sh_file.write("#SBATCH --mem=" + str(reserved_memory) + "G\n")
    sh_file.write("#SBATCH -o " + log_path + "/" + job_name + ".o\n")
    sh_file.write("#SBATCH -e " + log_path + "/" + job_name + ".e\n")
    sh_file.write("cd " + os.path.dirname(os.path.abspath(__file__)) + "\n")
    sh_file.write("srun " + build_command(cv_datafolder, noncv_datafile, save_dir, rare_filter, loci_list) + "\n")
    sh_file.close()
    os.system("sbatch " + job_path)


def submit_lsf_cluster_job(
    scratch_path,
    log_path,
    cv_datafolder,
    noncv_datafile,
    save_dir,
    reserved_memory,
    queue,
    rare_filter,
    loci_list,
):
    job_ref = str(time.time())
    job_name = "RARE_AUDIT_" + job_ref
    job_path = scratch_path + "/" + job_name + "_run.sh"
    sh_file = open(job_path, "w")
    sh_file.write("#!/bin/bash\n")
    sh_file.write("#BSUB -q " + queue + "\n")
    sh_file.write("#BSUB -J " + job_name + "\n")
    sh_file.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"\n')
    sh_file.write("#BSUB -M " + str(reserved_memory) + "GB\n")
    sh_file.write("#BSUB -o " + log_path + "/" + job_name + ".o\n")
    sh_file.write("#BSUB -e " + log_path + "/" + job_name + ".e\n")
    sh_file.write("cd " + os.path.dirname(os.path.abspath(__file__)) + "\n")
    sh_file.write(build_command(cv_datafolder, noncv_datafile, save_dir, rare_filter, loci_list) + "\n")
    sh_file.close()
    os.system("bsub < " + job_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
