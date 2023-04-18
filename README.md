# Multi-modal Instance Refinement for Action Recognition
A implementation of MMIR using pytorch.


## Data Preparation
1. Follow [MM-SADA](https://github.com/jonmun/MM-SADA-code/tree/master) to prepare data from P01, P08 and P22 of [EPIC-Kitchens](https://github.com/epic-kitchens/epic-kitchens-download-scripts) and the data structure should be constructed as follows.

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
