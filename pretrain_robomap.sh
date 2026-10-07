export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=$(pwd)

port=23666
GPUS_PER_NODE=4
NNODES=1

torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --master_port=$port robomap/pretrain_robomap.py \
  --branches 2 \
  --config_path "config/pretrain_robomap_config.yaml" \
  --vis_flag detection_1
