export WORKSPACE=$(dirname $(dirname $(pwd)))
export DATADIR="$WORKSPACE/data"

python $WORKSPACE/nlp_2020/train.py --data_dir $DATADIR/classification \
--model_name_or_path $DATADIR/classification/model \
--output_dir $DATADIR/classification/output \
--cache_dir $DATADIR/cache \
--embed_path $DATADIR/classification/sgns.sogounews.bigram-char