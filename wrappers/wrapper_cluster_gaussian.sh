#!/bin/bash
#SBATCH -p c23g
#SBATCH --account=thes1701
#SBATCH --job-name=wrapper
#SBATCH --output=autoslurm/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=mailslurm@gmail.com

[[ -e env.sh ]] && . env.sh

module load intel-compilers/2022.1.0
module load impi/2021.6.0
module load imkl/2022.1.0
module load imkl-FFTW/2022.1.0
module load intel/2022a
module load CUDA/11.8.0
module load OpenSSL/1.1
module load GCCcore/.11.3.0
module load binutils/2.38
module load GCC/11.3.0
module load XZ/5.2.5
module load libxml2/2.9.13
module load gettext/0.21
module load DB/18.1.40
module load expat/2.4.8
module load zlib/1.2.12
module load ncurses/6.3
module load libreadline/8.1.2
module load Perl/5.34.1
module load git/2.36.0-nodocs
module load bzip2/1.0.8
module load cURL/7.83.0
module load libarchive/3.6.1
module load CMake/3.26.3
module load Ninja/1.10.2
module load numactl/2.0.14

source /home/wc101072/miniconda/etc/profile.d/conda.sh
conda activate jax118

export OMP_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export PET_NPROC_PER_NODE=1

srun python gecco-torch/example_configs/gaussians_conditional.py --name r_dist_proc_rgb_ctxsplat10000