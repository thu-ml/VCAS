export TASK_NAME=sst2
export RESULTS_DIR=./results/$TASK_NAME/

mkdir -p $RESULTS_DIR
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $RESULTS_DIR \
  --overwrite_output_dir \
  --save_strategy no \
  --act_var_tau 0.025 \
  --w_var_tau 0.025 \
  --s_update_step 0.01 \
  --w_ratio_mul 0.95 \
  --cal_var_freq 100 \