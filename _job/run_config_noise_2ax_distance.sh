#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.10/3.10.14 cuda/11.8/11.8.0 cudnn/8.6/8.6.0
cd ~/repo/lab-abci
source ./env/bin/activate
python train/train.py config_noise_2ax_distance.yaml