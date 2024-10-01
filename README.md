# Design Automation: A Conditional VAE Approach to 3D Object Generation under Conditions

> **Abstract:** *Traditionally, engineering designs are created manually by experts. This process can be time-consuming and requires significant computing resources. Designs are iteratively created and simulated to satisfy physical constraints. Generative neural networks have the ability to learn the relationship between physical constraints and their geometric design. By leveraging this understanding, generative neural networks can generate innovative designs that satisfy these constraints. In the context of Industry 4.0, integrating these networks can significantly speed up the design process, reduce costs, and push the boundaries of traditional engineering practices. To achieve this goal, we propose a conditional variational autoencoder to learn the underlying relationship between geometry and physics. We validate this approach on the ShapeNetCore dataset, focusing on subsets that contain three-dimensional objects such as cars and airplanes, which contain both continuous and discrete data.*

# Content
- [Installation](#installation)
- [Data](#data)
- [Super-Resolution](#super-resolution)
- [Training](#training)
- [Evaluation](#evaluation)
- [Sources](#sources)

# Installation

```
conda create -n design-automation
conda activate design-automation
conda env update -n design-automation --file environment.yml
```

# Data

The ShapeNet dataset can be found at [4], the subclasses are copied from the official ShapeNet website and the drag coefficients from [5] and [6].

# Super-Resolution

Own PyTorch implementation of [3], please visit their repository for more details.

For training the Super-Resolution-Network:

```
python super_resolution_pytorch/train.py
```
If you use our PyTorch implementation, please refer to our repository `https://github.com/mhohmann1/super-resolution-pytorch` and of course to [3].

# Training

```
python train.py
```

# Evaluation

Before you evaluate, please be sure that you trained the Super-Resolution-Network, if not add `--super_res False`.

```
python eval.py
```

# Sources

`[1] https://github.com/AWehenkel/Normalizing-Flows`

`[2] https://github.com/seung-lab/connected-components-3d`

`[3] https://github.com/EdwardSmith1884/Multi-View-Silhouette-and-Depth-Decomposition-for-High-Resolution-3D-Object-Representation`

`[4] https://github.com/autonomousvision/shape_as_points`

`[5] https://decode.mit.edu/projects/dragprediction`

`[6] https://decode.mit.edu/projects/formfunction`