from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    background_rows: int = 1000000
    signal_rows: int = 100000
    injected_signal_rows: int = 1000
    sculpting_test_rows: int = 333334
    sic_background_rows: int = 612858
    sic_background_validation_stop: int = 266666
    preparation_seed: int = 1
