# Step by step instruction how to install CUDA 11 Ubuntu 20.04

## Remove previous environment
```
conda remove -n UpPoinTr --all
```
## NVidia Ubuntu 20.04 repository for CUDA 11


Add Ubuntu 20.04 repository 

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

```

## Install cuda toolkit

Update and install cuda toolkit, you will gain access to many version of cuda and cuda toolkit. 

```
sudo apt update
sudo apt install cuda-toolkit-11-0

```


## Install cuDNN

[Download cuDNN from NVidia](https://developer.nvidia.com/cudnn). You'll have to log in, answer a few questions then you will be redirected to download. Find the right cuDNN binary packages and save it on you computer.

```
tar -xvf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar

mv cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive cudnn

sudo cp cudnn/include/cudnn*.h /usr/local/cuda-11.0/include
sudo cp cuda/lib/libcudnn* /usr/local/cuda-11.0/lib64
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*
```

## Link CUDA 11.0 as CUDA
```
sudo rm -rf /usr/local/cuda
sudo ln -snf /usr/local/cuda-11.0/ /usr/local/cuda
```

## Add CUDA_HOME to PATH environmet


Edit `/home/$USER/.bashrc` file


```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
```


## Restart PC

## Create environment
```
conda create -n UpPoinTr.yml
```

## Install pointnet libraries
```
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Build Pytorch extensions
```
bash install.sh
```


