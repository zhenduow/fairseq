import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'bart.large.cnn/',
    checkpoint_file='model.pt',
    data_name_or_path='cnn_dm-bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 1
with open('cnn_dm/test.source') as source, open('cnn_dm/test.target') as target, open('cnn_dm/test.hypo', 'w') as fout:
    source = source.readlines()
    target = target.readlines()
    for sample_id in range(len(source)):
        article, summary = source[sample_id].strip(), target[sample_id].strip()
        print(article)
        print(summary)
        with torch.no_grad():
            score = bart.sample_for_evaluation([article], [summary], beam=1, lenpen=2.0, max_len_b=len(summary), min_len=len(summary), no_repeat_ngram_size=3)
        fout.write(summary +'\n' + str(score.data.tolist())+ '\n')
        fout.flush()