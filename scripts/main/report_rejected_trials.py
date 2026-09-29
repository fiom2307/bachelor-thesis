from src.data.dataset import get_data_for_subject


N_SUBJECTS = 9
ORIGINAL_TRIALS_PER_SESSION = 288


def main() -> None:
    rows: list[str] = []

    total_train_rejected = 0
    total_train_retained = 0
    total_eval_rejected = 0
    total_eval_retained = 0

    for subject in range(1, N_SUBJECTS + 1):
        data = get_data_for_subject(subject)

        if data is None:
            rows.append(f"A{subject:02d}  data unavailable")
            continue

        X_train, y_train, X_eval, y_eval = data

        train_retained = len(y_train)
        eval_retained = len(y_eval)

        train_rejected = (
            ORIGINAL_TRIALS_PER_SESSION - train_retained
        )
        eval_rejected = (
            ORIGINAL_TRIALS_PER_SESSION - eval_retained
        )

        assert len(X_train) == train_retained
        assert len(X_eval) == eval_retained

        total_train_rejected += train_rejected
        total_train_retained += train_retained
        total_eval_rejected += eval_rejected
        total_eval_retained += eval_retained

        rows.append(
            f"A{subject:02d}"
            f"{train_rejected:>16}"
            f"{train_retained:>16}"
            f"{eval_rejected:>16}"
            f"{eval_retained:>16}"
        )

    header = (
        f"{'Subject':<10}"
        f"{'Train rejected':>16}"
        f"{'Train retained':>16}"
        f"{'Eval rejected':>16}"
        f"{'Eval retained':>16}"
    )

    separator = "-" * len(header)

    total_row = (
        f"{'Total':<4}"
        f"{total_train_rejected:>16}"
        f"{total_train_retained:>16}"
        f"{total_eval_rejected:>16}"
        f"{total_eval_retained:>16}"
    )

    output = "\n".join(
        [
            header,
            separator,
            *rows,
            separator,
            total_row,
        ]
    )

    print(output)


if __name__ == "__main__":
    main()