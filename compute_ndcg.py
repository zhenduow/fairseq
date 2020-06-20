from sklearn.metrics import ndcg_score

original_lines = open('cnn_dm/test.hypo').readlines()
grammar_lines = open('cnn_dm/test_grammar.hypo').readlines()
syntax_lines = open('cnn_dm/test_syntax.hypo').readlines()
semantic_lines = open('cnn_dm/test_semantic.hypo').readlines()

topn = min(len(original_lines), len(grammar_lines), len(syntax_lines), len(semantic_lines))
original_score, grammar_score, syntax_score, semantic_score = [], [], [], []

print(topn)

for l in range(topn):
    if l%2:
        original_score.append(float(original_lines[l]))
        grammar_score.append(float(grammar_lines[l]))
        syntax_score.append(float(syntax_lines[l]))
        semantic_score.append(float(semantic_lines[l]))
            
ndcg_total = ndcg_score([[5,4,2,1]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score)))
print(ndcg_total)



