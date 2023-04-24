CUDA_VISIBLE_DEVICES=2
#runwayml/stable-diffusion-v1-5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/mingqi/diffusers/examples/dreambooth/data/zxc_512"
export CLASS_DIR="/home/mingqi/diffusers/examples/dreambooth/data/class_man"
export OUTPUT_DIR="./models/zxc_0.3_xformer"

accelerate launch train_dreambooth_fast.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.3 \
  --instance_prompt="a photo of <zxc> man" \
  --class_prompt="a photo of man" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=20 \
  --max_train_steps=1000 \
  --mixed_precision="fp16" \
  --xformer \
  --train_text_encoder \

  
  
  

