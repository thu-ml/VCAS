export TASK_NAME=sst2
export RESULTS_DIR=./results/baseline_$TASK_NAME/

mkdir -p $RESULTS_DIR
python run_glue_baseline.py \
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
  --save_strategy no