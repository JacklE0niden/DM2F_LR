cd /mnt/pami23/zengyi/DIP/DM2F/
conda activate dm2f
source activate dm2f
# pip3 install -r requirements.txt
# python3 train_ohaze.py --gpus=0

python3 train.py --gpus=3

python3 train_ohaze_MyModel.py --gpus=0
python3 train_MyModel.py --gpus=3
python3 train_ohaze.py --gpus=2

# run ddp
python -m torch.distributed.launch --nproc_per_node=8 --use_env train_MyModel_DDP.py --gpus 0,1,2,3,4,5,6,7


# ablation experiment
python3 train_ohaze_ablation.py --gpus=3

# ablation experiment
python3 train_MyModel_ablation.py --gpu=3