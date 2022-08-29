#Define the path of each parameter
python inference.py \
    --model bert-base-uncased \
    --test source/test.csv \
    --log log \
    --checkpoint checkpoint \
    --type title_abst \
    --res outputs \
    --num_labels 6 \
    --max_sequence_len 512 \
    --test_batch_size 64 