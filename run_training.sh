accelerate launch --config_file "src/config/config_fsdp.yaml" src/models/train.py 	\
  --num_train_epochs 10 \
	--output_dir " "  \
	--train_files  " " " " \
	--val_files   " " " "  \
	--batch_size    2 \
	--max_train_samples 200 \
	--max_eval_samples  50  \
	--num_beams   4 \
	--weight_decay  0.3 \
	--mixed_precision fp16  \
	--gradient_accumulation_steps 2 \
	--with_tracking True  \
	--report_to wandb \
	--checkpointing_steps epoch


