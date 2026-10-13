from pathlib import Path

from pymaldi.process_api import PROCESSING_API_VERSION, ProcessingConfig, run_processing_records


def test_run_processing_records_csv_smoke(tmp_path: Path) -> None:
    src = tmp_path / "x.csv"
    src.write_text("mz,intensity\n100,1\n200,2\n300,3\n", encoding="utf-8")

    res = run_processing_records(src, config=ProcessingConfig(process_peaklists=True))
    assert res.api_version == PROCESSING_API_VERSION
    assert len(res.records) == 1
    rec = res.records[0]
    assert rec.mz.size == 3
    assert rec.intensity.size == 3
