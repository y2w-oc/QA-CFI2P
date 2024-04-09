# Learning Quantity-Aware 2D-3D Correspondence for Image-to-Point Cloud Registration

This repository is a 2-level hierarchical architecture implementation of Quantity-Aware CFI2P.

## Environments
- We run the code on an Nvidia RTX 3090 GPU and i7-12700K CPU. Other devices may also work.
- Install the required Python packages from ***requirements.txt*** .

## Demo
We provide a demo to perform image-to-point cloud registration and visualize cross-modal correspondences. Please run:
```shell script
cd demo
python demo.py
```

## Data Preparation
### KITTI Odometry
Download the KITTI Odometry dataset from its official website.
Run the data preprocessing scripts and arrange the processed data as:
```
./CFI2P/
./kitti/
   ├── calib/
   |    ├── 00/
   |    |    ├── calib.txt
   |    |    └── time.txt
   |    ├── 01/
   |    └── ...
   ├── data_odometry_color_npy/
   |    └── sequences/
   |         ├── 00/
   |         |   ├── image2/
   |         |   |    ├── 000000.npy
   |         |   |    └── ...
   |         |   └── image3/
   |         |        ├── 000000.npy
   |         |        └── ...
   |         ├── 01/
   |         └── ...
   └── data_odometry_velodyne_npy/
       └── sequences/
            ├── 00/
            |   └── voxel0.1-SNr0.6/
            |        ├── 000000.npy
            |        └── ...
            ├── 01/
            └── ...
```
### NuScenes
Download the NuScenes dataset from its official website.
Please refer to the public repository of CorrI2P (TCSVT 2023) to process the NuScenes dataset, or download the preprocessed data provided by them directly. Please arrange the data as:
```
./CFI2P/
./nuscenes2/
   ├── train/
   |    ├── img/
   |    |   ├── 000000.npy
   |    |   └── ...
   |    ├── PC/
   |    |   ├── 000000.npy
   |    |   └── ...
   |    └── K/
   |        ├── 000000.npy
   |        └── ...
   └── test/
        ├── img/
        |   ├── 000000.npy
        |   └── ...
        ├── PC/
        |   ├── 000000.npy
        |   └── ...
        └── K/
            ├── 000000.npy
            └── ...
```

## Training
Train the coarse part of CFI2P:
```shell script
python Train_coarse.py
```
Train the fine part of CFI2P:
```shell script
python Train_fine.py
```

## Testing
In ***checkpoint***, we provide a pretrained model _"kitti.pth"_ to reproduce the reported results on the KITTI Odometry dataset. Please run:
```shell script
python Test.py
```

## Acknowledgements
We thank the authors of DeepI2P (CVPR 2021), Point Transformer (CVPR 2021), CorrI2P (TCSVT 2023) and so on for their public codes.


