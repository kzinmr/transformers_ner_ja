export BERT_MODEL=cl-tohoku/bert-base-japanese
# export WORK_DIR=${PWD}
export DATA_DIR=${WORK_DIR}/data/
export OUTPUT_DIR=${WORK_DIR}/outputs/
export CACHE=${WORK_DIR}/cache/
export LABEL_PATH=$DATA_DIR/label_types.txt
export SEED=42
mkdir -p $OUTPUT_DIR
# In Docker, the following error occurs due to not big enough memory:
# `RuntimeError: DataLoader worker is killed by signal: Killed.`
# Try to reduce NUM_WORKERS or MAX_LENGTH or BATCH_SIZE or increase docker memory
export NUM_WORKERS=1
export GPUS=0

export MAX_LENGTH=64
export BATCH_SIZE=32
export LEARNING_RATE=5e-5

export NUM_EPOCHS=1
export NUM_SAMPLES=100

python3 ner.py \
--model_name_or_path=$BERT_MODEL \
--output_dir=$OUTPUT_DIR \
--accumulate_grad_batches=1 \
--max_epochs=$NUM_EPOCHS \
--seed=$SEED \
--do_train \
--do_predict \
--cache_dir=$CACHE \
--gpus=$GPUS \
--data_dir=$DATA_DIR \
--labels=$LABEL_PATH \
--num_workers=$NUM_WORKERS \
--max_seq_length=$MAX_LENGTH \
--train_batch_size=$BATCH_SIZE \
--eval_batch_size=$BATCH_SIZE \
--learning_rate=$LEARNING_RATE \
--adam_epsilon=1e-8 \
--weight_decay=0.0 \
--num_samples=$NUM_SAMPLES
