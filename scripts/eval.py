"""Usage: python scripts/eval_mae.py +models.model1="(run_id1, pred_name1)" +models.model2="run_id2" +eval_name="evaluation_name"
Runs the evaluation on the validation or test set for one or more models.
The models are specified as a dictionary in the format:
models:
  model_id1: (run_id1, pred_name1)  # Run ID and prediction name as a tuple
  model_id2: run_id2                # Just run ID, pred_name defaults to "default"

You can also specify a custom name for the evaluation using the 'eval_name' parameter.
If not provided, a timestamp will be used.

The predictions from those runs must have been previously saved using
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

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from multi_sources.eval.abstract_evaluation_metric import AbstractMultisourceEvaluationMetric


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_object(cfg)
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

        info_filepath = root_dir / "info.csv"
        # Load the info dataframe written by the writer.
        info_df = pd.read_csv(info_filepath)
        info_df["dt"] = pd.to_timedelta(info_df["dt"]).dt.round("min")
        # Convert the "spatial_shape" from string to tuple
        info_df["spatial_shape"] = info_df["spatial_shape"].apply(
            lambda x: tuple(map(int, x.strip("()").split(", "))) if x != "()" else ()
        )

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
            eval_class, model_data=model_data, parent_results_dir=parent_results_dir
        )

        # Run the evaluation
        if isinstance(evaluator, AbstractMultisourceEvaluationMetric):
            # This evaluator processes all sources together
            evaluator.evaluate_sources(num_workers=num_workers)
        else:
            # This evaluator processes each source separately
            for model_id, data in model_data.items():
                run_id = data["run_id"]
                pred_name = data["pred_name"]
                info_df = data["info_df"]

                print(f"Evaluating model: {model_id} (run_id: {run_id}, prediction: {pred_name})")

                # Process each source for this model
                if num_workers > 1:
                    with ProcessPoolExecutor(num_workers) as executor:
                        futures = []
                        for source_name in info_df["source_name"].unique():
                            futures.append(
                                executor.submit(
                                    evaluator.evaluate_source,
                                    source_name,
                                    info_df[info_df["source_name"] == source_name],
                                    run_id=run_id,
                                    pred_name=pred_name,
                                    model_id=model_id,
                                    verbose=False,
                                )
                            )
                        for future in futures:
                            future.result()
                else:
                    for source_name in info_df["source_name"].unique():
                        print(f"Processing source {source_name}")
                        evaluator.evaluate_source(
                            source_name,
                            info_df[info_df["source_name"] == source_name],
                            run_id=run_id,
                            pred_name=pred_name,
                            model_id=model_id,
                        )

    print(f"Evaluation '{eval_name}' completed. Results saved to {parent_results_dir}")


if __name__ == "__main__":
    # Enable the full errors in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
