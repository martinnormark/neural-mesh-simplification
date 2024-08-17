# Neural Mesh Simplification

This repository contains an implementation of the paper "Neural Mesh Simplification" by Potamias et al. (CVPR 2022). The project aims to provide a fast, learnable method for mesh simplification that generates simplified meshes in real-time.

Research, methodology introduced in the [Neural Mesh Simplification paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Potamias_Neural_Mesh_Simplification_CVPR_2022_paper.pdf), with the updated info shared in [supplementary material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Potamias_Neural_Mesh_Simplification_CVPR_2022_supplemental.pdf).

This implementation could not have been done without the use of an LLM, specifically Claude Sonnet 3.5 by Anthropic. It was useful to create a project, upload the PDF of the papers there and use the custom instructions in [llm_instructions.txt](llm_instructions.txt). To steer the model, a copy of the file structure (which it helped create early on) is also useful. This can be created with the command `tree -F > file_structure.txt` in the root directory of the project.

## Overview

Neural Mesh Simplification is a novel approach to reduce the resolution of 3D meshes while preserving their appearance. Unlike traditional simplification methods that collapse edges in a greedy iterative manner, this method simplifies a given mesh in one pass using deep learning techniques.

The method consists of three main steps:

1. Sampling a subset of input vertices using a sophisticated extension of random sampling.
2. Training a sparse attention network to propose candidate triangles based on the edge connectivity of sampled vertices.
3. Using a classification network to estimate the probability that a candidate triangle will be included in the final mesh.

## Features

- Fast and scalable mesh simplification
- One-pass simplification process
- Preservation of mesh appearance
- Lightweight and differentiable implementation
- Suitable for integration into learnable pipelines

## Installation

```bash
git clone https://github.com/martinnormark/neural-mesh-simplification.git
cd neural-mesh-simplification
pip install -r requirements.txt
```

## Usage

```python
from neural_mesh_simplifier import NeuralMeshSimplifier

# Initialize the simplifier
simplifier = NeuralMeshSimplifier()

# Load a mesh
original_mesh = load_mesh("path/to/your/mesh.obj")

# Simplify the mesh
simplified_mesh = simplifier.simplify(original_mesh, target_faces=1000)

# Save the simplified mesh
save_mesh(simplified_mesh, "path/to/simplified_mesh.obj")
```

## Training

To train the model on your own dataset:

```bash
python ./scripts/train.py --data_path /path/to/your/dataset --epochs 100 --batch_size 32
```

## Evaluation

To evaluate the model on a test set:

```bash
python ./scripts/evaluate.py --model_path /path/to/saved/model --test_data /path/to/test/set
```

## Results

Our implementation achieves results comparable to those reported in the original paper:

- Up to 10x faster than traditional methods
- Competitive performance on appearance error measures
- Effective preservation of mesh structure and details

## Citation

If you use this code in your research, please cite the original paper:

```
@InProceedings{Potamias_2022_CVPR,
    author    = {Potamias, Rolandos Alexandros and Ploumpis, Stylianos and Zafeiriou, Stefanos},
    title     = {Neural Mesh Simplification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {18583-18592}
}
```

## Contributing

Contributions are welcome to improve this implementation. Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
