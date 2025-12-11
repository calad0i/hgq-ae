# HGQ FPGA'26 Artifact Evaluation

## Requirements

- `python>=3.10` (`3.11+` recommended)
- `gnu make` and `g++` for building
- Reasonably modern linux system (e.g., OpenSUSE Leap 16) with reasonable amount of RAM (At least 32GB+)
- `Vivado 2025.1` with license for synthesis and implementation for `xcvu13p-flga2577-2-e`
- Optional, but strongly recommended: `gnu_parallel` for parallel execution (usually available via package manager under name `parallel`)

For python dependencies, you may use (with [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)):

```shell
# Install micromamba if you don't have it, or follow the official instructions
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

## Environment setup

```bash
micromamba create -n hgq-ae -f env.yml
micromamba activate hgq-ae
```

and clone the repository containing codes:

## Dataset preparation

Quick script: `bash dataset/prepare_datasets.sh`

See detailed instructions if you wish to do it manually.

## Results replication

Quick script: `bash run_all.sh x` where `x` is the number of parallel jobs for vivado synthesis.

See detailed instructions if you wish to run individual experiments.

## Detailed Instructions

### Dataset Preparation 

The following assumes you are in the root directory of this repository.

#### JSC

##### HLF, OpenML

No manual steps needed. The dataset (openml id: 42468) will be automatically downloaded in the dataloader.

##### HLF, CERNBox

```bash
curl -L -o dataset/cernbox.h5 "https://cernbox.cern.ch/s/jvFd5MoWhGs1l5v/download"
```

##### PLF

Two steps are needed: downloading the dataset and extract & combine the features.

```bash
# Download the dataset (train and test)
curl -L -o hls4ml_LHCjet_150p_train.tar.gz "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1"
curl -L -o hls4ml_LHCjet_150p_val.tar.gz "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz?download=1"

# Extract, combine features, and cleanup
tar -xvzf hls4ml_LHCjet_150p_train.tar.gz
tar -xvzf hls4ml_LHCjet_150p_val.tar.gz

mkdir -p jsc_plf

python3 jsc_plf_dataset.py -i ./train/ -o jsc_plf/150c-train.h5 -j 4
python3 jsc_plf_dataset.py -i ./val/ -o jsc_plf/150c-test.h5 -j 4

rm -rf ./train/
rm -rf ./val/
```

#### TGC

```bash
curl -L -o dataset/tgc_dataset.h5 "https://huggingface.co/datasets/Calad/fake-TGC/resolve/main/fake_TGC_0.041_pruned.h5?download=true"
```

### Experiment Execution

Training scripts are commented out to save time. You may uncomment them if you wish to retrain the models.

`-j` in `run_test.py` specifies the number of parallel processes for model testing RTL conversion
`-j` in `parallel` specifies the number of parallel Vivado synthesis jobs.

### JSC

#### HLF, OpenML

```bash
# python3 code/jsc_hlf/run_train.py -i dataset/openml_cache.h5 -o models/jsc_openml
python3 code/jsc_hlf/run_test.py -d dataset/openml_cache.h5 -i models/jsc_openml -o models/jsc_openml/verilog -j 5

# Vivado synth
ls -d models/jsc_openml/verilog/* | parallel -j 1 --bar --eta 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'

# Print results
da4ml report models/jsc_openml/verilog/* -c comb_metric latency 'latency(ns)' LUT DSP FF Fmax[MHz]
```

#### HLF, CERNBox

```bash
# python3 code/jsc_hlf/run_train.py -i dataset/cernbox.h5 -o models/jsc_cernbox --cern-box
python3 code/jsc_hlf/run_test.py -d dataset/cernbox.h5 -i models/jsc_cernbox -o models/jsc_cernbox/verilog --cern-box -j 5

# Vivado synth
ls -d models/jsc_cernbox/verilog/* | parallel -j 1 --bar --eta 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'

# Print results
da4ml report models/jsc_cernbox/verilog/* -c comb_metric latency 'latency(ns)' LUT DSP FF Fmax[MHz]
```

#### PLF, 3 feature, 32 particles

**note**: The lower accuracy models in the paper were pipelined every 5 adders intstead of 2 (`-lc`) here. FF and latency will differ accordingly. To recover those numbers, change `-lc 2` to `-lc 5` in the commands below. 

```bash
# python3 code/jsc_plf/run_train.py -i dataset/jsc_plf -o models/jsc_plf/32-3 -n 32 --ptetaphi
python3 code/jsc_plf/run_test.py -d dataset/jsc_plf -i models/jsc_plf/32-3 -o models/jsc_plf/32-3/verilog -n 32 --ptetaphi -j 2 -lc 2

# Vivado synth
ls -d models/jsc_plf/32-3/verilog/* | parallel -j 1 --bar --eta 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'

# Print results
da4ml report models/jsc_plf/32-3/verilog/* -c comb_metric latency 'latency(ns)' LUT DSP FF Fmax[MHz]
```

#### PLF, 3 feature, 32 particles

```bash
# python3 code/jsc_plf/run_train.py -i dataset/jsc_plf -o models/jsc_plf/32-16 -n 32
python3 code/jsc_plf/run_test.py -d dataset/jsc_plf -i models/jsc_plf/32-16 -o models/jsc_plf/32-16/verilog -n 32 -j 2 -lc 2

# Vivado synth
ls -d models/jsc_plf/32-16/verilog/* | parallel -j 1 --bar --eta 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'

# Print results
da4ml report models/jsc_plf/32-16/verilog/* -c comb_metric latency 'latency(ns)' LUT DSP FF Fmax[MHz]
```

#### PLF, 3 feature, 32 particles

```bash
# python3 code/jsc_plf/run_train.py -i dataset/jsc_plf -o models/jsc_plf/64-16 -n 64
python3 code/jsc_plf/run_test.py -d dataset/jsc_plf -i models/jsc_plf/64-16 -o models/jsc_plf/64-16/verilog -n 64 -j 2 -lc 2

# Vivado synth
ls -d models/jsc_plf/64-16/verilog/* | parallel -j 1 --bar --eta 'cd {} && vivado -mode batch -source build_vivado_prj.tcl > synth.log'

# Print results
da4ml report models/jsc_plf/64-16/verilog/* -c comb_metric latency 'latency(ns)' LUT DSP FF Fmax[MHz]
```