import argparse
import json
from pathlib import Path

from picai_prep.data_utils import atomic_image_write

from run_oof_fold_v2 import image_paths, predict_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-list", type=Path, required=True)
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-folds", nargs="+", default=["0", "1", "2", "3"])
    args = parser.parse_args()

    subjects = json.loads(args.subject_list.read_text())["subject_list"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in subjects:
        output = args.output_dir / f"{case_id}_detection_map.mha"
        if output.exists():
            continue
        detection, score = predict_case(
            args.ensemble_folds,
            args.results_dir,
            image_paths(args.gc_cases_dir / case_id),
        )
        atomic_image_write(detection, output)
        (args.output_dir / f"{case_id}_case_level_likelihood.json").write_text(json.dumps(score) + "\n")


if __name__ == "__main__":
    main()
