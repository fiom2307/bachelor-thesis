from src.pipelines.csp_lda_flow import run_csp_lda_for_subject

res = []

# 9 subjects
for subj in range(1, 10):
    acc = run_csp_lda_for_subject(subj)
    text = f"A0{subj} -- Accuracy: {acc:.4f} ({acc*100:.1f}%)"
    res.append(text)

for text in res:
    print(text)