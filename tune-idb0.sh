export WANDB_PROJECT=instructdial
cd scripts
deepspeed main.py \
    --model_name_or_path prakharz/DIAL-BART0 \
    --do_train \
    --do_eval \
    --train_file text2textfiles/sample_doc2dial.json \
    --validation_file text2textfiles/test_sample_doc2dial.json \
    --text_column prompt \
    --target_column output \
    --output_dir ./saved_model/tunedmodel_m1 \
    --per_device_train_batch_size=9 \
    --per_device_eval_batch_size=9 \
    --gradient_accumulation_steps 12 \
    --learning_rate 5e-05 \
    --overwrite_output_dir \
    --predict_with_generate \
    --gradient_checkpointing \
    --save_total_limit 3\
    --deepspeed ./configs/ds-config.json \
    --evaluation_strategy steps\
    --num_train_epochs 5\
    --fp16 \
    --load_best_model_at_end\
    --metric_for_best_model f1 \
    --save_steps 100\
    --eval_steps 100\
    --logging_steps 25\
    --max_source_length 512\
    --max_target_length 30
