accelerate launch train_st_embeddings_ddp.py \
    --data_path fikriokan/ygty-qposneg-2k \
    --model_name ytu-ce-cosmos/turkish-e5-large \
    --output_dir out_dbg_2k \
    --batch_size 64 \
    --epochs 3 \
    --lr 2e-5 \
    --max_query_len 96 \
    --max_passage_len 384 \
    --use_triplet_if_negs \
    --margin 0

