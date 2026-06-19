from src.pipelines.eegnet_flow import run_eegnet_for_subject

res = []

# 9 subjects
for subj in range(1, 10):
    acc = run_eegnet_for_subject(subj)
    text = f"A0{subj} -- Accuracy: {acc:.4f} ({acc*100:.1f}%)"
    res.append(text)

for text in res:
    print(text)