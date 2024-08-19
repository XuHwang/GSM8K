export CUDA_VISIBLE_DEVICES=0,2

python evaluate.py --model-path /data1/home/huangxu/work/OpenLLMs/Meta-Llama-3.1-8B-Instruct
python evaluate.py --model-path /data1/home/huangxu/work/OpenLLMs/Meta-Llama-3-8B-Instruct
python evaluate.py --model-path /data1/home/huangxu/work/OpenLLMs/ToolACE-8B