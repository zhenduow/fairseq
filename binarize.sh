fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "cnn_dm/train.bpe" \
  --validpref "cnn_dm/val.bpe" \
  --destdir "cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;