from src.pipelines.csp_lda_pipeline import run_csp_lda_for_subject
from src.pipelines.eegnet_pipeline import run_eegnet_for_subject
from src.utils.results import save_accuracy_comparison, load_accuracy_comparison
from src.data.dataset import get_data_for_subject
from src.data.preprocessing import apply_car

def print_results(results):
    print("-" * 70)
    print(f"{'Subject':<10} {'CSP+LDA':<20} {'EEGNet':<20} {'Difference':<20}")
    print("-" * 70)

    csp_accs = []
    eegnet_accs = []

    for subject_name, csp_acc, eegnet_acc in results:
        csp_text = f"{csp_acc:.4f} ({csp_acc * 100:.1f}%)"
        eegnet_text = f"{eegnet_acc:.4f} ({eegnet_acc * 100:.1f}%)"
        difference_text = f"{eegnet_acc - csp_acc:.4f}"

        print(f"{subject_name:<10} {csp_text:<20} {eegnet_text:<20} {difference_text:<20}")

        csp_accs.append(csp_acc)
        eegnet_accs.append(eegnet_acc)
    
    mean_csp = sum(csp_accs) / len(csp_accs)
    mean_eegnet = sum(eegnet_accs) / len(eegnet_accs)
    mean_difference = mean_eegnet - mean_csp

    print("-" * 70)
    print(f"{'Mean':<10} "
          f"{mean_csp:.4f} ({mean_csp * 100:.1f}%){'':<6} "
          f"{mean_eegnet:.4f} ({mean_eegnet * 100:.1f}%){'':<6} "
          f"{mean_difference:.4f}")

def run_experiment():
    results = []

    for subj in range(1, 10):
        subject_name = f"A0{subj}"

        print(f"\nRunning {subject_name}...")

        data = get_data_for_subject(subj)

        if data is None:
            print(f"Skipping {subject_name}: data not found")
            continue

        X_train, y_train, X_eval, y_eval = data

        X_train = apply_car(X_train)
        X_eval = apply_car(X_eval)

        data_car = X_train, y_train, X_eval, y_eval

        csp_acc = run_csp_lda_for_subject(subj, data_car)
        eegnet_acc = run_eegnet_for_subject(subj, data_car)

        results.append((subject_name, csp_acc, eegnet_acc))

    return results

def main():
    results = load_accuracy_comparison()

    if results is not None:
        print_results(results)
        return

    results = run_experiment()
    
    save_accuracy_comparison(results)

    print_results(results)


if __name__ == "__main__":
    main()