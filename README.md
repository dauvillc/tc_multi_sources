# A DL model for multi-sources forecasting and interpolation with an application to tropical cyclones
This repository implements a DL architecture adapted to learning from multiple sources. The possibilities of inputs include:
* A flexible number of sources: while a large set of sources can be used as input to the model, a specific sample may contain any subset of the sources.
* Sources of different dimensionalities and natures (0D, e.g. station measurements; 1D, e.g. vertical profile; 2D, e.g. remote sensing images).
* Sources misaligned in space and time, for example remote sensing images covering different geographical areas (which may even be disjoint), and at irregular time intervals.
* Sources of the same type with different characteristics, e.g. remote sensing images in the same band from different satellites with different exact frequencies and ground sampling distance.


The framework is built with [PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/docs/pytorch/stable/), [Hydra](https://hydra.cc/docs/intro/) and [Weights and Biases](https://wandb.ai/site/). It includes multiple blocks to be able to learn in that general setting:
* A `MultiSourceDataset` and custom `collate_fn` function to quickly assemble batches with a flexible number of sources while limiting the required memory. The returned samples include the sources' values, spatio-temporal coordinates, characteristic vectors, and availability masks.
* A multi-source backbone based on the transformer to process multiple sources. The backbone decomposes the attention into a spatial attention based on the Swin Transformer and a cross-sources / temporal attention that uses anchor points to limit its cost. The backbone uses two separate sequences *for each source*, one for the values and one for the coordinates.
* A `Lightning` module that implements that receives the output of the `MultiSourceDataset` as input and performs the following tasks:
  * Embedding each source into two common latent spaces: values and coordinates;
  * Randomly mask one of the embedded sources in each sample;
  * Process the embedded sequences through the backbone;
  * Project the updated values sequences to their original spaces;
  * Compute the loss between the masked source's original values and the reconstructed values.
* Many options to perform different experiments, such as:
  * Training in a self-supervised setting: randomly mask and reconstruct a source for each sample;
  * Training or fine-tuning in a supervised manner by always masking the same source. The embedding layers, backbone and output layers can be frozen or reset.
  * Fine-tuning or testing on a source unseen during training.

All of the experiments and scripts are based on `Hydra` and require to be familiar with its functioning. This comes at the cost of some complexity of not familiar with it, but offers great flexibility.

# Running experiments
## Setting up the dataset
The data must first be put in the following format:

```bash
dataset_dir
└── processed
   ├── <source_1>
   │   ├── <filename_1.nc>
   │   ├── <filename_2.nc>
   │   ├── ...
   │   ├── <last_filename.nc>
   │   ├── samples_metadata.json
   │   ├── source_metadata.json
   ├── <source_2>
   │   ├── ...
   ├── ...
   └── <last_source>
      └── ...

```

where:
* The files `<filename.nc>` can be any netCDF4 files whose dimensions are the spatial dimensions of the source, for example (latitude, longitude) for a remote sensing image. The data variables should be the source's channels (e.g. the brightness temperature). The dataset must also include the following variables `latitude` and `longitude`. The latitude and longitude variables must have exactly the same shape as the data variables and give the spatial coordinates of each point in the dataset.
* The file `samples_metadata.json` is a pandas DataFrame saved in JSON format with `orient=records` and `lines=True` (see the [to_json](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html) method). It describes the elements available for that source (each row corresponds to a netCDF file). It must include the following columns:
  * 'source_name' (str): Name of the source.
  * 'source_type' (str): Type of the source. Different sources may have the same source type.
  * 'sid' (str): Storm ID, unique to each storm.
  * 'time' (pd.Timestamp): time of the observation.
  * 'data_path' (str): path to the corresponding netCDF4 file.
  * characteristic_variables (dict of str to float): One column for each characteristic variable of the source. The *characteristic variables* of a source are variables that characterize a source within its source type (all sources of a common type have values for the same characteristic variables, although those values can differ). Those characteristic variables are given in each row as a dict {channel_name: value}, so that each channel can have its own characteristic variables. For example, for the "frequency" variable:\
  `{'TB_89.0V': 89.0, 'TB_157.0V': 157.0, 'TB_183.31_1.0H': 183.31, 'TB_183.31_3.0H': 183.31, 'TB_190.31V': 190.31}`

The preprocessing options must then be set in `conf/preproc.yaml`. At this point, all paths must be indicated in a config file under `conf/paths/` and selected it in `preproc.yaml`.\
The dataset can then be formatted by running `python preproc/train_val_test_split.py` followed by `python preproc/compute_normalization_constants`.

## Training a model
A model can be trained by running
```bash
python scripts/train_mae +experiment=<experiment_name> wandb.name=<name> model=<model> trainer=<trainer>
```
where `<experiment_name>` is the name of an experiment config file under `conf/experiments`, `<name>` is an arbitrary name given to the training run, `<model>` is a model configuration under `conf/model/` and trainer is a trainer config under `conf/trainer`.

Experiments are logged using WandB; by default wandb is run offline and thus requires to run `wandb sync` to upload the run. Wandb can be set to online or disabled via the `wandb.mode` argument with hydra. For each run, a run id is generated. This id is used to name the checkpoint directories, as well as for wandb. 

## Making predictions
Given a run id, one can make predictions with the corresponding model via\
```bash
python scripts/make_predictions.py run_id=<run_id> split=<val_or_test>
```

The scripts will look for a checkpoint under `<checkpoints_dir>/run_id`. The checkpoints dir must be set in `conf/paths.yaml`.

## Evaluating a model
The evaluation of a model is done by:\
```bash
python scripts/eval_mae.py run_id=<run_id>
```

The predictions must have been made using `scripts/make_predictions.py` for that run id beforehand. Which evaluation is
run depends on the classes instantiated in `conf/eval.yaml`. For example,
```yaml
evaluation_classes:
  visual:
    _target_: multi_sources.eval.visualization.VisualEvaluation
    eval_fraction: 0.05
```
Runs the visual evaluation of a model on 5\% of the predictions.

## Running the self-supervised experiment on TC-PRIMED
### Preparing TC-PRIMED
The TC-PRIMED dataset must first be downloaded into a root directory, in the same format as in its AWS bucket: `season/basin/number/<files.nc>`. Then, the root directory and the directory to which the preprocessed data will saved must be indicated in `conf/paths.yaml`. Then, run:
```bash
python preproc/prepare_tc_primed_satellite.py num_workers=<num_workers>
python preproc/prepare_tc_primed_env.py num_workers=<num_workers>
python preproc/regrid.py num_workers=<num_workers>
```
The number of workers can be set for parallel processing. If 0 or 1, the data will be processed sequentially.\
Those scripts format the TC-PRIMED dataset to the format required by the general data preprocessing describe earlier. Various options can be set either as command line arguments or in `conf/preproc.yaml`, such as the target resolution for regridding.

### Running the self-supervised training experiment
To train a model on the default 37GHz self-supervised reconstruction task, use
```bash
python scripts/train_mae.py +experiment=mae_37GHz_improved model=msvit_8b_d256 trainer=4gpus wandb.name=<run_name> dataloader.batch_size=<batch_size>
```
The number of gpus and the batch size should be adapted to the system used. The experiment can also be launched as a job on a SLURM cluster using the submitit
plugin for Hydra:
```bash
python scripts/train_mae.py +experiment=mae_37GHz_improved model=msvit_8b_d256 trainer=4gpus wandb.name=<run_name> dataloader.batch_size=<batch_size> 
hydra/launcher=<launcher> --multirun
```
Where the launcher is a config file under `conf/hydra/launcher` (see examples in the folder).