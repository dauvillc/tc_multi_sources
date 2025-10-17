"""Usage: python scripts/eval_mae.py +models.model1="(run_id1, pred_name1)" +models.model2="run_id2" +eval_name="evaluation_name"
Runs the evaluation on the validation or test set for one or more models.
The models are specified as a dictionary in the format:
models:
  model_id1: (run_id1, pred_name1)  # Run ID and prediction name as a tuple
  model_id2: run_id2                # Just run ID, pred_name defaults to "default"

You can also specify a custom name for the evaluation using the 'eval_name' parameter.
If not provided, a timestamp will be used.

The predictions from those runs must have been previously saved using
scripts/make_predictions.py.

The predictions are saved in the following format:
targets:
- root_dir / <rank> / targets / source_name / <batch_index.npy>
predictions:
- root_dir / <rank> / outputs / source_name / <batch_index.npy>
info dataframe:
- root_dir / <rank> / info.csv

Each batch is an array of shape (batch_size, channels, height, width).
Note that a given batch may not be included in all sources, and a batch
can be included in the targets but not in the predictions.

The info dataframe contains the following columns:
source_name (str), batch_idx (int), index_in_batch (int), sample_index (int), dt (float).
Since the predictions made have been made with different batch sizes, we need to work with
the sample_index to match the samples between different models.

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

import os
from datetime import datetime
from pathlib import Path
from time import localtime, strftime

import hydra
import pandas as pd
import submitit
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


class EvalJob(submitit.helpers.Checkpointable):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self):
        cfg = self.cfg
        num_workers = cfg["num_workers"]

        # Get the model dictionary from the config
        if "models" not in cfg:
            raise ValueError("'models' must be specified in the config.")

        model_dict = cfg["models"]

        # Get the evaluation name, or use a timestamp if not provided
        eval_name = cfg.get("eval_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"Evaluation name: {eval_name}")

        # Create the parent results directory with the evaluation name
        base_results_dir = Path(cfg["paths"]["results"])
        parent_results_dir = base_results_dir / eval_name
        parent_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {parent_results_dir}")

        # Create a dictionary to store loaded data for each model
        model_data = {}

        # Load data for each model
        for model_id, model_spec in model_dict.items():
            # Handle different model specification formats
            if isinstance(model_spec, tuple) or (
                isinstance(model_spec, list) and len(model_spec) == 2
            ):
                # Format: model_id: (run_id, pred_name)
                run_id, pred_name = model_spec
            elif isinstance(model_spec, str):
                # Format: model_id: run_id
                run_id = model_spec
                pred_name = "default"
            else:
                raise ValueError(
                    f"Invalid model specification for {model_id}. "
                    "Expected either (run_id, pred_name) tuple or run_id string."
                )

            print(f"Processing model: {model_id} (run_id: {run_id}, prediction: {pred_name})")

            root_dir = Path(cfg["paths"]["predictions"]) / run_id / pred_name
            if not root_dir.exists():
                raise ValueError(
                    f"Predictions for run_id {run_id} with pred_name {pred_name} do not exist.\
                        Please run scripts/make_predictions_mae.py first."
                )

            # Load the info dataframe for each rank
            rank_csvs = sorted(root_dir.glob("info_*.csv"))
            if not rank_csvs:
                raise ValueError(f"No info CSV files found in {root_dir}.")
            info_dfs = []
            for rank, rank_csv in enumerate(rank_csvs):
                # Load the info dataframe written by the writer.
                rank_df = pd.read_csv(rank_csv)
                rank_df["dt"] = pd.to_timedelta(rank_df["dt"]).dt.round("min")
                # Convert the "spatial_shape" from string to tuple
                rank_df["spatial_shape"] = rank_df["spatial_shape"].apply(
                    lambda x: tuple(map(int, x.strip("()").split(", "))) if x != "()" else ()
                )
                rank_df["rank"] = rank  # Add the rank column
                info_dfs.append(rank_df)

            # Concatenate all rank dataframes into a single dataframe
            info_df = pd.concat(info_dfs, ignore_index=True).reset_index(drop=True)

            # Add model identification to the dataframe
            info_df["run_id"] = run_id
            info_df["pred_name"] = pred_name
            info_df["model_id"] = model_id

            # Store data for this model
            model_data[model_id] = {
                "info_df": info_df,
                "root_dir": root_dir,
                "run_id": run_id,
                "pred_name": pred_name,
            }

        # Instantiate the evaluation classes
        evaluation_classes = cfg["eval_class"]
        if evaluation_classes is None:
            raise ValueError(
                "No evaluation classes specified. Please specify the evaluation_classes in the config."
            )

        # For each evaluation class, evaluate all models
        print("Building evaluation classes")
        for eval_cls_name, eval_class in evaluation_classes.items():
            print(f"Evaluating with {eval_cls_name}")

            # Instantiate the evaluator with all models
            evaluator = instantiate(
                eval_class,
                model_data=model_data,
                parent_results_dir=parent_results_dir,
                num_workers=num_workers,
            )

            # Run the evaluation
            evaluator.evaluate(num_workers=num_workers)

        print(f"Evaluation '{eval_name}' completed. Results saved to {parent_results_dir}")


def _make_executor(cfg: DictConfig) -> submitit.AutoExecutor:
    # Where submitit writes logs/stdout/err and its internal state
    folder = Path("submitit") / (
        "eval_" + cfg["eval_name"] + f"_{strftime('%Y%m%d_%H-%M-%S', localtime())}"
    )
    folder.mkdir(parents=True, exist_ok=True)

    ex = submitit.AutoExecutor(folder=str(folder), slurm_max_num_timeout=1)
    ex.update_parameters(**cfg["setup"])
    return ex


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    # Enable the full errors in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("nan", lambda: float("nan"))
    cfg = OmegaConf.to_object(cfg)

    # Create the job object and submit it to the auto executor.
    job = EvalJob(cfg)

    if cfg.get("launch_without_submitit", False):
        return job()
    else:
        executor = _make_executor(cfg)
        job = executor.submit(job)
        print(f"Submitted job {job.job_id}")
        return 0


if __name__ == "__main__":
    main()
