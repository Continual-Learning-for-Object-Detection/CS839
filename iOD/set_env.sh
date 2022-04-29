#!/bin/bash

echo 'Date: ' `date` 
echo 'Host: ' `hostname` 
echo 'System: ' `uname -spo` 
echo 'GPU: ' `lspci | grep NVIDIA`

# # Prepare the dataset
# tar zxf MNIST_data.tar.gz

# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Update conda as workaround for https://github.com/conda/conda/issues/9681
# Will no longer be needed once conda >= 4.8.3 is available from repo.anaconda.com
conda install conda=4.10.1

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate iod

pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# conda install -c anaconda memory_profiler
# conda remove msgpack-python
# conda install -c anaconda msgpack-python=0.6.1
# conda install -c anaconda memory_profiler

conda list