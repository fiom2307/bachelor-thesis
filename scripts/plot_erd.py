"""Generate ERD plots for one or more subjects."""

from src.analysis.erd import select_trials_by_class
from src.utils.paths import ERD_RESULTS_DIR

SUBJECT = 1
CLASS_ID = 0


def main() -> None:

    # Aquí cargarás los evaluation trials.
    #
    # Ejemplo conceptual:
    #
    # X_eval, y_eval = load_evaluation_data(SUBJECT)
    #
    # left_hand_trials = select_trials_by_class(
    #     X=X_eval,
    #     y=y_eval,
    #     class_id=CLASS_ID,
    # )
    #
    # Después:
    # compute_erd(...)
    # plot_erd(...)
    # save figure in results/erd/

    print(f"Preparing ERD analysis for Subject {SUBJECT:02d}")


if __name__ == "__main__":
    main()