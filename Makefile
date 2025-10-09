prep: venv
.PHONY: venv mamba data

SHELL:=/bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
PROJNAME := partnn
MAMBABASE := ${PROJBASE}/mamba

#######################################
##########   Virtual Env   ############
#######################################

venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install -r requirements.txt
	source venv/bin/activate && python -m pip install tsnecuda==3.0.1+cu122 \
		-f https://tsnecuda.isx.ai/tsnecuda_stable.html
	source venv/bin/activate && python -m pip install -e .
	rm -rf *.egg-info

mamba:
	mkdir -p ${MAMBABASE}
	cd ${MAMBABASE}; \
	curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba;
	set -e; \
	export MAMBA_ROOT_PREFIX=${MAMBABASE}; \
	eval "$$(${MAMBABASE}/bin/micromamba shell hook -s posix)"; \
	micromamba activate; \
	micromamba create  -y -n ${PROJNAME} python=3.11 -c conda-forge; \
	micromamba activate ${PROJNAME}; \
	micromamba install -y -c conda-forge openssh; \
	export TMPDIR=${PROJBASE}/pip; mkdir -p $${TMPDIR}; \
	python -m pip install --upgrade pip; \
	python -m pip install jupyter; \
	rm -r $${TMPDIR};

mambatf:
	set -e; shopt -s expand_aliases; \
	export MAMBA_ROOT_PREFIX=${MAMBABASE}; \
	eval "$$(${MAMBABASE}/bin/micromamba shell hook -s posix)"; \
	micromamba activate ${PROJNAME}; \
	micromamba install -y -c conda-forge openssh; \
	micromamba install -y -c conda-forge cudatoolkit=11.8.0; \
	export TMPDIR=${PROJBASE}/pip; mkdir -p $${TMPDIR}; \
	python -m pip install nvidia-cudnn-cu11==8.6.0.163; \
	CUDNN_PATH=$$(dirname $$(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")); \
	export LD_LIBRARY_PATH=$${CONDA_PREFIX}/lib/:$${CUDNN_PATH}/lib:$${LD_LIBRARY_PATH}; \
	python -m pip install nvidia-cublas-cu11==11.11.3.6 nvidia-cudnn-cu11==8.6.0.163; \
	rm -r $${TMPDIR};

careshist:
	source ./activate mamba; \
	jupytext --to py notebooks/n02_histdata.ipynb; \
	NWORKERS=20; \
	for MYRANK in $$(seq 0 $$((NWORKERS-1))); do \
		python notebooks/n02_histdata.py -f 01_cares -r $${MYRANK} -s $${NWORKERS} & \
	done; wait;

nest: mamba
	source ./activate mamba && micromamba install -y -c conda-forge packaging=24.1 numpy=1.26.4 mkl=2023
	source ./activate mamba && micromamba install -y -c pytorch -c nvidia faiss-gpu=1.8.0
	source ./activate mamba && micromamba install -y --no-deps tsnecuda=3.0.2 cudatoolkit=11.8.0 \
		gflags=2.2.2 libsodium=1.0.20 zeromq=4.3.5 -c conda-forge
	source ./activate mamba && mkdir -p ./trash/piptmp && TMPDIR=./trash/piptmp \
		python -m pip install torch torchvision torchaudio \
		&& rm -rf ./trash/piptmp
	source ./activate mamba && python -m pip uninstall numpy -y
	source ./activate mamba && python -m pip install -r requirements.txt
	source ./activate mamba && python -m pip uninstall numpy -y
	source ./activate mamba && python -m pip install numpy==1.26.4
	source ./activate mamba && python -m pip install -e .
	rm -rf *.egg-info
	mkdir -p ${PROJBASE}/trash ${PROJBASE}/.cache ${PROJBASE}/storage ${PROJBASE}/results

# Package Installation Steps
#
# Step 1: Install `packaging`, `numpy`, and `mkl` using conda-forge. These are 
#         needed for the next `faiss-gpu` micromamba install.
#
# 		  It is important to install everything that needs the conda-forge 
#         channel before installing `faiss-gpu`. 
#
#         When installing `faiss-gpu`, the conda-forge channel should not be accessible 
#         since it will cause the `faiss` cpu package to be installed and the gpu features 
#         of faiss will become inaccessible.
#
# Step 2: Install `faiss-gpu=1.8.0` from the official pytroch channel. 
#         Make sure you do not reveal/use the `conda-forge` channel in this step.
#
#         The `conda-forge` version of `faiss-gpu=1.8.0` installs the cpu version 
#         of `faiss` as well, and you will end up losing access to the gpu features.
#
# Step 3: Carefully install `tsnecuda` without changing the `faiss-gpu=1.8.0` installation
#         from the official pytorch channel. 
#
#         Unfortunately, `conda install tsnecuda=3.0.2 -c conda-forge` removes the 
#         official installation and uses conda-forge to re-install `faiss-gpu=1.8.0`. 
#
#         Again, The unofficial `conda-forge` version of `faiss-gpu=1.8.0` installs 
#         the cpu version of `faiss` as well, and you will end up losing access to 
#         the gpu features.
#
#         To avoid this, I collected what packages needed to be installed for `tsnecuda` 
#         in conda, and then filtered out the `libfaiss` and `faiss*` libraries, and 
#         installed the rest. 
#
#         That is how you have the `--no-deps cudatoolkit=11.8.0 gflags=2.2.2 
#		  libsodium=1.0.20 zeromq=4.3.5` options.
#
# Step 4: Install `torch` and friends. 
#
#         If you face an "out of space" error, make sure you read the note.
# 
# Step 5: Install the rest of requirements.
#
# Step 6: Uninstall and re-install numpy to a non-breaking version.
#
# Step 7: Install your own package, and be done!

# Note:
#
#   If `pip install` is throwing a "no space left on device" error, there are two possible solutions:
#
#       1. The pip cache directory may be running out of space.
# 
#          To temporarily change the pip cache directory for a single install, you can 
#          either specify of the following
#           
#          `PIP_CACHE_DIR=/path/to/cache python -m pip install ....`, or
#          
#          `python -m pip --cache-dir=/path/to/cache install ....`.
#
#   	2. The most likely case is that the "TMPDIR" path has no space left:
#
#		   ```
#          mkdir -p ./trash/piptmp
#          TMPDIR=./trash/piptmp python -m pip install ...
#          rm -rf ./trash/piptmp
#          ``` 
