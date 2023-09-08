
#python3 -m torch.distributed.run --nproc_per_node=1 --master_port=1234 finetune.py \
python3 finetune.py \
    --base_model tiiuae/falcon-180B \
    --data-path garage-bAInd/Open-Platypus \
    --output_dir ./falcon-platypus-13b \
    --batch_size 8 \
    --micro_batch_size 8 \
    --num_epochs 1 \
    --learning_rate 0.0003 \
    --cutoff_len 2048 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '["dense"]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --wandb_project huggingface \
    --wandb_run_name falcon-platypus-180b \
    --wandb_watch false \
    --wandb_log_model true \
    --resume_from_checkpoint /home/ubuntu/Platypus-QLORA/falcon-platypus-13b/checkpoint-8000 \
    --warmup_steps 100
