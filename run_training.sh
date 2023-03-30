accelerate launch --config_file src/config/config_defaultMultiGPU.yaml src/models/train.py 	\
  --num_train_epochs 10 \
	--output_dir "./output"  \
	--train_files  "./ketod/train_sample.json" "./fused_chat/train_sample.json" "./woi/train_sample.json"\
	--val_files   "./ketod/valid_sample.json" "./fused_chat/valid_sample.json" "./woi/valid_sample.json" \
	--batch_size    2 \
	--num_beams   4 \
	--weight_decay  0.3 \
	--learning_rate 5e-5 \
	--mixed_precision fp16  \
	--gradient_accumulation_steps 2 \
	--with_tracking True  \
	--report_to mlflow \
	--checkpointing_steps epoch \
	--do_eval_per_epoch True


