# test.sh

# conda activate dm2f
# cd ../DIP/DM2F/
# pip install -r requirements.txt
cd /mnt/pami23/zengyi/DIP/DM2F/
conda activate dm2f
python test_ohaze.py --gpus=0
python test.py --gpus=1