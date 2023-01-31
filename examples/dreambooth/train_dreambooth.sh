CUDA_VISIBLE_DEVICES=1
#runwayml/stable-diffusion-v1-5
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/mingqi/diffusers/examples/dreambooth/data/cat_512"
export CLASS_DIR="/home/mingqi/diffusers/examples/dreambooth/data/class_cat"
export OUTPUT_DIR="./models/cat1000_3e6_constant_0.3_xformer"

accelerate launch train_dreambooth_fast.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=0.3 \
  --instance_prompt="a photo of a <cat> cat" \
  --class_prompt="a photo of a cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=3e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=800 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  
  
  

