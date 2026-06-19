from src.pipelines.csp_lda_flow import run_csp_lda_for_subject
from src.pipelines.eegnet_flow import run_eegnet_for_subject


def main():
    results = []

    for subj in range(1, 10):
        subject_name = f"A0{subj}"

        print(f"\nRunning {subject_name}...")

        csp_acc = run_csp_lda_for_subject(subj)
        eegnet_acc = run_eegnet_for_subject(subj)

        results.append((subject_name, csp_acc, eegnet_acc))

    print("\nFinal results:")
    print("-" * 55)
    print(f"{'Subject':<10} {'CSP+LDA':<20} {'EEGNet':<20}")
    print("-" * 55)

    for subject_name, csp_acc, eegnet_acc in results:
        csp_text = f"{csp_acc:.4f} ({csp_acc * 100:.1f}%)"
        eegnet_text = f"{eegnet_acc:.4f} ({eegnet_acc * 100:.1f}%)"

        print(f"{subject_name:<10} {csp_text:<20} {eegnet_text:<20}")


if __name__ == "__main__":
    main()