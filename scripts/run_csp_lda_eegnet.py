from src.pipelines.csp_lda_flow import run_csp_lda_for_subject
from src.pipelines.eegnet_flow import run_eegnet_for_subject
from src.utils.results import save_accuracy_comparison, load_accuracy_comparison

def print_results(results):
    print("\nFinal results:")
    print("-" * 70)
    print(f"{'Subject':<10} {'CSP+LDA':<20} {'EEGNet':<20} {'Difference':<20}")
    print("-" * 70)

    for subject_name, csp_acc, eegnet_acc in results:
        csp_text = f"{csp_acc:.4f} ({csp_acc * 100:.1f}%)"
        eegnet_text = f"{eegnet_acc:.4f} ({eegnet_acc * 100:.1f}%)"
        difference_text = f"{eegnet_acc - csp_acc:.4f}"

        print(f"{subject_name:<10} {csp_text:<20} {eegnet_text:<20} {difference_text:<20}")

def run_experiment():
    results = []

    for subj in range(1, 10):
        subject_name = f"A0{subj}"

        print(f"\nRunning {subject_name}...")

        csp_acc = run_csp_lda_for_subject(subj)
        eegnet_acc = run_eegnet_for_subject(subj)

        results.append((subject_name, csp_acc, eegnet_acc))

    return results

def main():
    results = load_accuracy_comparison()

    if results is not None:
        print_results(results)
        return

    results = run_experiment()
    output_file = save_accuracy_comparison(results)

    print_results(results)
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()