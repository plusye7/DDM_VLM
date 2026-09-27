# DDM-VLM

DDM-VLM contains wafer-map defect classification experiments based on direct
vision-language model (VLM) inference, CLIP retrieval, and multi-round DDM-VLM
reasoning.

> The project name is **DDM-VLM**. The import package remains `DDM` because a
> hyphen cannot be used in a Python module name.

## Repository Layout

- `models/`: model wrappers for Qwen, Owl3, Llama, and InternVL3.
- `Qwen/`, `Owl3/`, `Llama/`, `Internvl3/`: model-specific configs and experiment entry points.
- `data/`: public data splitting and FAISS index-building utilities. Dataset files and generated artifacts are not committed.
- `ddm_vlm.yml`: the shared Conda environment for all supported model backends.

## Environment

The included `ddm_vlm.yml` defines the `ddm_vlm` Conda environment. It targets
64-bit Linux with Python 3.10 and an NVIDIA GPU stack (PyTorch, CUDA 12 runtime,
FAISS-GPU, Transformers, and vLLM). Use a recent NVIDIA driver compatible with
the CUDA packages pinned in the file. Windows and CPU-only installations require
platform-appropriate replacements for the GPU-specific packages.

Create and activate the environment from this directory:

```bash
conda env create -f ddm_vlm.yml
conda activate ddm_vlm
```

If the environment already exists, synchronize it with the checked-in file:

```bash
conda env update -n ddm_vlm -f ddm_vlm.yml --prune
conda activate ddm_vlm
```

Verify the main runtime and GPU access:

```bash
python -c "import torch, transformers, faiss, clip, vllm; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

Large model weights are not included. Download the model you intend to use and
set its local path in the corresponding `config.yaml`.

## Open-Source Path Policy

All local paths have intentionally been left as empty strings in each model's
`config.yaml`.

Before running an experiment, fill the relevant values in the config file for
the model you want to run:

- `model.path`: local model directory or model identifier.
- `data.test_path`: test `.npz` file.
- `data.database_path`: retrieval database `.npz` file.
- `retrieval.index_path`: FAISS index file.
- `outputs.*`: output directories for generated results.
- `examples.image_path`: optional image used by model smoke-test scripts.

Do not commit local datasets, retrieval indexes, result files, model weights, or API keys.

Most entry points also accept command-line overrides. Use `--config` to point to
an alternate YAML file when you do not want to edit the default model config.

## Data Format

This project expects MixedWM38-style `.npz` files with two arrays:

- `arr_0`: wafer maps shaped `(N, 52, 52)`. Values `0` and `1` are treated as
  non-defective/background pixels, and value `2` is treated as a defect pixel.
- `arr_1`: multi-hot labels shaped `(N, 8)`.

The label order is:

```text
center, donut, edge_loc, edge_ring, loc, near_full, scratch, random
```

The raw dataset is not included in this repository. Place it locally, then use
the scripts below to generate split files and retrieval indexes.

## Running Commands

The commands below use the import package name `DDM`, so run them from the
parent directory of this source directory. For example, if this README is at
`<workspace>/DDM/README.md`, first run:

```bash
cd <workspace>
```

### Data Splits

Create 9:1 and 7:3 database/test splits from a MixedWM38-style `.npz` file:

```bash
python -m DDM.data.split --input "<source_npz>" --output-dir "<output_dir>" --ratios 9:1 7:3
```

Generated `.npz` files and split metadata are ignored by Git.

### Retrieval Index

Build the FAISS index from the retrieval database split before running RAG or
DDM-VLM experiments:

```bash
python -m DDM.data.build_index \
  --database-path "<database_npz>" \
  --output-dir "<retrieval_output_dir>" \
  --overwrite
```

The index builder uses CLIP `ViT-B/32`, matching the retrieval code used by the
experiment scripts. The command writes:

- `global_faiss_cosine.bin`: FAISS cosine index. Use this path as
  `retrieval.index_path` in the model `config.yaml`.
- `global_labels.npy`: text labels aligned with FAISS rows.
- `global_label_vectors.npy`: one-hot labels aligned with FAISS rows.
- `global_faiss_metadata.json`: index build metadata.

Set `data.database_path` in the model `config.yaml` to the same database `.npz`
used to build the index.

### DDM-VLM Example

Run the Qwen DDM-VLM entry point with its default configuration:

```bash
python -m DDM.Qwen.ddm_vlm
```

To use an explicit configuration file:

```bash
python -m DDM.Qwen.ddm_vlm --config DDM/Qwen/config.yaml
```

The same entry-point name is available under `Owl3`, `Llama`, and `Internvl3`.
