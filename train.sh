export HF_HOME=/scratch0/zx22/zijie/cache/
CUDA_VISIBLE_DEVICES=4,5,6,7, nohup python trainer.py > log/bio10.log 2>&1 &
