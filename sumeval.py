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
bsz = 32
with open('cnn_dm/test.source') as source, open('cnn_dm/test.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch, scores_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for id in range(len(hypotheses_batch)):
                fout.write(hypotheses_batch[id] + '\n' + str(scores_batch[id]) + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch, scores_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for id in range(len(hypotheses_batch)):
                fout.write(hypotheses_batch[id] + '\n' + str(scores_batch[id]) + '\n')
                fout.flush()