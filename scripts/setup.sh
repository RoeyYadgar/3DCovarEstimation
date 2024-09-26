set -e
cd ~


if ! command -v conda &> /dev/null  #Conda installation
then
	wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O miniconda.sh
	bash miniconda.sh -b -p $HOME/miniconda
	rm miniconda.sh

	echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
	source $HOME/.bashrc
fi


if [ ! -e "ASPIRE-Python" ];
then
	git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git
fi

if ! command -v nvcc &> /dev/null #adding nvcc to path
then
	CUDA_PATH = "/usr/local/cuda-12.1" #change version of cuda if needed
	echo "export PATH=$CUDA_PATH/bin:\$PATH" >> $HOME/.bashrc
	echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> $HOME/.bashrc
	source $HOME/.bashrc
fi


if ! conda env list | grep -q "\baspire\b"; #Aspire installation
then
	cd ASPIRE-Python
	source $HOME/miniconda/etc/profile.d/conda.sh
	conda create --name aspire python=3.8 pip

	conda activate aspire

	pip install jupyter
	pip install spyder
	pip install torch
	pip install starfile
	
	pip install -e ".[gpu-12x]"


	cd ~
fi

if ! conda info --envs | grep -q "cryodrgn";
then
    conda create -n cryodrgn -y
    conda activate cryodrgn
    pip install cryodrgn
    conda deactivate
fi

if ! conda info --envs | grep -q "recovar";
then
    conda create --name recovar python=3.11 -y
    conda activate recovar
    pip install -U "jax[cuda12_pip]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    git clone https://github.com/ma-gilles/recovar.git
    pip install --no-deps -r  recovar/recovar_install_requirements.txt
    python -m ipykernel install --user --name=recovar 
    conda deactivate
fi

if [ ! -e "aspire" ]; #Matlab aspire installation
then
	git clone https://github.com/PrincetonUniversity/aspire.git
	cd aspire
	matlab -nodisplay -r "initpath;install;exit;"	
	cd ~
fi

if [ ! -e "relion" ]; #relion installation
then
	git clone https://github.com/3dem/relion.git
	cd relion
	git checkout master # or ver4.0; see below
	mkdir build
	cd build
	cmake ..
	make
	cd ~
fi