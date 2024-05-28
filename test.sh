# test.sh

# conda activate dm2f
# cd ../DIP/DM2F/
# pip install -r requirements.txt
cd /mnt/pami23/zengyi/DIP/DM2F/
conda activate dm2f
python3 test_test.py --gpus=2
python3 test_ohaze.py --gpus=3
python3 test.py --gpus=0