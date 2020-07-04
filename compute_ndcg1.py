from sklearn.metrics import ndcg_score

original_lines = open('cnn_dm/test.hypo').readlines()
grammar_lines = open('cnn_dm/test_grammar.hypo').readlines()
syntax_lines = open('cnn_dm/test_syntax.hypo').readlines()
semantic_lines = open('cnn_dm/test_semantic.hypo').readlines()
lead3_lines = open('cnn_dm/test_lead3.hypo').readlines()
irrelevant_lines = open('cnn_dm/test_irrelevant.hypo').readlines()

topn = min(len(original_lines), len(grammar_lines), len(syntax_lines), len(semantic_lines), len(lead3_lines), len(irrelevant_lines))
original_score, grammar_score, syntax_score, semantic_score, lead3_score, irrelevant_score = [], [], [], [], [], []


for l in range(topn):
    if l%2:
        original_score.append(float(original_lines[l]))
        grammar_score.append(float(grammar_lines[l]))
        syntax_score.append(float(syntax_lines[l]))
        semantic_score.append(float(semantic_lines[l])) 
        lead3_score.append(float(lead3_lines[l]))
        irrelevant_score.append(float(irrelevant_lines[l]))

print("Bart")
print(topn)

print("original-irrelevant")
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-lead3")
ndcg_total = ndcg_score([[1,0]]*len(original_score),list(zip(original_score, lead3_score)), k=1)
print(ndcg_total)

print("original-lead3-irrelevant")
ndcg_total = ndcg_score([[1,0,0]]*len(original_score),list(zip(original_score,lead3_score, irrelevant_score)), k=1)
print(ndcg_total)

print("original-grammar-syntax-semantic")
ndcg_total = ndcg_score([[1,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score)), k =1)
print(ndcg_total)

print("original-grammar-syntax-semantic-irrelevant")
ndcg_total = ndcg_score([[1,0,0,0,0]]*len(original_score),list(zip(original_score,grammar_score,syntax_score, semantic_score, irrelevant_score)), k=1)
print(ndcg_total)



