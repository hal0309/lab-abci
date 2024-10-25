#$ -l rt_G.small=1
#$ -l h_rt=12:00:00
#$ -N run_config
#$ -o run_config.out
#$ -e run_config.err
#$ -cwd



module load python/3.10/3.10.14 cuda/11.8/11.8.0 cudnn/8.6/8.6.0
cd ~/repo/lab-abci
source ./env/bin/activate
python train/train.py config.yaml 10