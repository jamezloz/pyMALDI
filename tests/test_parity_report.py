from pymaldi.parity.report import ParityThresholds, summarize_results


def test_summarize_results_marks_pass_fail() -> None:
    items = [
        {
            "sample_id": "ok",
            "metrics": {
                "mz_median_abs_err_da": 0.1,
                "intensity_pearson_r": 0.995,
            },
        },
        {
            "sample_id": "bad",
            "metrics": {
                "mz_median_abs_err_da": 0.9,
                "intensity_pearson_r": 0.80,
            },
        },
    ]

    out = summarize_results(items, thresholds=ParityThresholds())

    assert out["total"] == 2
    assert out["passed"] == 1
    assert out["failed"] == 1
