"""Usage: python scripts/eval_mae.py +run_id=<wandb_run_id>
Runs the evaluation on the validation or test set for a given run_id.
The predictions from that run must have been previously saved using
scripts/make_predictions_mae.py.

The predictions are saved in the following format:
targets:
- root_dir / targets / source_name / <batch_index.npy>
predictions:
- root_dir / outputs / source_name / <batch_index.npy>
info dataframe:
- root_dir / info.csv

Each batch is an array of shape (batch_size, channels, height, width).
Note that a given batch may not be included in all sources, and a batch
can be included in the targets but not in the predictions.

The info dataframe contains the following columns:
source_name (str), batch_idx (int), index_in_batch (int), dt (float).

The hydra configuration used to run this script must include
the entry 'evaluation_classes', of the form
evaluation_classes:
  eval_1:
    _target_: 'multi_sources.eval.path.to.evaluation_class'
    arg1: val1
    arg2: val2
  ...
where 'multi_sources.eval.path.to.evaluation_class' is the path to the
evaluation class to be used. The class must inherit from
AbstractEvaluationMetric and implement the method evaluate_source.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import hydra
import pandas as pd
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from concurrent.futures import ProcessPoolExecutor
from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
    num_workers = cfg["num_workers"]
    if "run_id" not in cfg:
        raise ValueError("Usage: python scripts/eval_mae.py run_id=<wandb_run_id>")
    run_id = cfg["run_id"]

    root_dir = Path(cfg["paths"]["predictions"]) / run_id
    if not root_dir.exists():
        raise ValueError(
            f"Predictions for run_id {run_id} do not exist.\
                Please run scripts/make_predictions_mae.py first."
        )
    info_filepath = root_dir / "info.csv"
    # Load the info dataframe written by the writer.
    info_df = pd.read_csv(info_filepath)
    info_df['dt'] = pd.to_timedelta(info_df['dt']).dt.round('min')
    # Create the results directory
    results_dir = Path(cfg["paths"]["results"]) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate the evaluation classes
    evaluation_classes = cfg["evaluation_classes"]
    evaluators = {}
    print("Building evaluation classes")
    for eval_name, evaluator_cfg in evaluation_classes.items():
        evaluator = instantiate(evaluator_cfg, predictions_dir=root_dir, results_dir=results_dir)
        evaluators[eval_name] = evaluator

    # For each evaluator, evaluate each source
    for eval_name, evaluator in evaluators.items():
        print(f"Evaluating {eval_name}")
        if isinstance(evaluator, AbstractMultisourceEvaluationMetric):
            # This evaluator processes all sources together
            evaluator.evaluate_sources(info_df)
        else:
            # This evaluator processes each source separately.
            # This allows for parallel processing of sources, if requested.
            if num_workers > 1:
                with ProcessPoolExecutor(num_workers) as executor:
                    futures = []
                    for source_name in info_df["source_name"].unique():
                        futures.append(
                            executor.submit(
                                evaluator.evaluate_source,
                                source_name,
                                info_df[info_df["source_name"] == source_name],
                                verbose=False,
                            )
                        )
                    for future in futures:
                        future.result()
            else:
                for source_name in info_df["source_name"].unique():
                    print(f"Processing source {source_name}")
                    evaluator.evaluate_source(
                        source_name, info_df[info_df["source_name"] == source_name]
                    )


if __name__ == "__main__":
    main()
