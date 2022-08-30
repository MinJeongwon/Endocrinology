#Define the path of each parameter
python train.py \
     --model bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12 \
     --source_data_dir ./source/ \
     --max_sequence_len 512\
     --type title_abst \
     --num_labels 6\
     --epoch 50 \
     --train_batch_size 128\
     --valid_batch_size 128 \
     --res outputs \
     --log log \
     --checkpoint checkpoint \
     --lr 2e-5 \
     --n_warmup_steps 0