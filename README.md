# Multi-modal Instance Refinement for Action Recognition
A implementation of MMIR using pytorch.


## Data Preparation
Follow [MM-SADA](https://github.com/jonmun/MM-SADA-code/tree/master) to prepare data from P01, P08 and P22 of [EPIC-Kitchens](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and the data structure should be constructed as follows.

```
├── rgb
|   ├── P01
|   |   ├── P01_01
|   |   |   ├── frame_0000000000.jpg
|   |   |   ├── ...
|   |   ├── P01_02
|   |   ├── ...
|   ├── P08
|   |   ├── P08_01
|   |   ├── ...
|   ├── P22
|   |   ├── P22_01
|   |   ├── ...

├── flow
|   ├── P01
|   |   ├── P01_01
|   |   |   ├── u 
|   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   ├── ...
|   |   |   ├── v
|   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   ├── ...
|   |   ├── P01_02
|   ├── P08
|   |   ├── P08_01
|   |   ├── ..
|   ├── P22
|   |   ├── P22_01
|   |   ├── ..
```
## Environment Setup
There are two ways to setup the environment.
### Build docker image (Recommended)
1. Clone this repository.
2. Run the following command:
```
cd build_mmir_env
bash build_env.sh
docker run -it -d --runtime=nvidia --shm-size 32G --name=mmir -v /MMIR:/workspace mmir:1.0
docker exec -it mmir /bin/bash
```
### Use conda
1. Clone this repository.
2. Run the following command:
```
conda create -n mmir python=3.7
conda activate mmir
pip install -r requirements.txt
```

## Steps to train

1. Modify the path inside the shell script ```MMIR.sh``` to fit your own setting.
2. Run the script.
```
bash MMIR.sh
```

## Steps to evaluate and conduct ablation studies

1. Modify the path inside the shell scripts to fit your own setting.
2. Use the following scripts to explore any component you like.
```
bash MMIR_test.sh
bash MMIR_woir.sh
bash MMIR_sourceonly.sh
bash MMIR_woself.sh
bash MMIR_DA.sh
bash MMIR_self.sh

```

## Acknoledgements

In this project, we borrowed some codes from the following projects:

[Multi-Modal Domain Adaptation for Fine-Grained Action Recognition](https://github.com/jonmun/MM-SADA-code/tree/master)

[Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition](https://github.com/IBM/action-recognition-pytorch)

[Pytorch-I3D](https://github.com/piergiaj/pytorch-i3d)

[SIR](https://github.com/ChenJinBIT/SIR)

Thanks for sharing your codes.


