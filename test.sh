# test.sh

# conda activate dm2f
# cd ../DIP/DM2F/
# pip install -r requirements.txt

cd /mnt/pami23/zengyi/DIP/DM2F/
conda activate dm2f

# 自己采集的数据集
python3 test_1_selfcollected.py --gpus=3
python3 test_2_selfcollected.py --gpus=3
# baseline算法1——O-Haze
python3 test_1_baseline.py --gpus=3

# baseline算法2——RESIDE
python3 test_2_baseline.py --gpus=3

# 改进算法1——O-Haze
python3 test_1_My.py --gpus=3

# 改进算法2——RESIDE
python3 test_2_My.py --gpus=3

# 后台操作
# nohup python3 test_2_My.py --gpus=3 > output_test_11.log 2>&1 &
