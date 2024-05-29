cd /mnt/pami23/zengyi/DIP/DM2F/
conda activate dm2f
# 或者
source activate dm2f
# pip3 install -r requirements.txt
# python3 train_ohaze.py --gpus=0

# 后台训练
nohup python3 train_2_My.py --gpus=3 > output.log 2>&1 &

# run ddp
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train_MyModel_DDP.py --gpus 0,1,2,3,4,5,6,7

# baseline算法1——O-Haze
python3 train_1_baseline.py --gpus=3

# baseline算法2——RESIDE
python3 train_2_baseline.py --gpus=3

# 改进算法1——O-Haze
python3 train_1_My.py --gpus=3

# 改进算法2——RESIDE
python3 train_2_My.py --gpus=3

# 消融实验1——O-Haze
python3 train_1_My_ablation.py --gpus=3

# 消融实验2——RESIDE
python3 train_2_My_ablation.py --gpus=3
