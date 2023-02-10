CUDA_VISIBLE_DEVICES=2
#runwayml/stable-diffusion-v1-5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dog1_512"
export CLASS_DIR="./data/class_borderCollie_dog"
export OUTPUT_DIR="./models/dog1000_2.5e6_constant_0.5_xformer"

accelerate launch train_dreambooth_fast.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --instance_prompt="a photo of a <dog1> border collie dog" \
  --class_prompt="a photo of a border collie dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=2.5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=1000 \
  --mixed_precision="fp16" \
  --xformer

  
  
  

