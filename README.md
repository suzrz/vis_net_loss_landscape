# Neural Network Training Progress Visualization
This tool provides a visualization of the training progress of a neural network model. It uses the Linear Path 
technique introduced by Goodfellow et al. in  [[1]](#1), Quadratic Path introduced in this work and PCA projection 
of a loss landscape, introduced by Hao Li et al. in [[2]](#2).

# Requirements
The tool requires following dependencies to run:
- Python>=3.8
- pip>=20

If CUDA will be used, then also:
- CUDA>=11.1

# Installation
For running this tool it is necessary to have installed PyTorch. Recommended way of installation is from the project 
[website](https://pytorch.org/), where it is possible to get cuda version. Other way is to use ```pip```:

Windows:
```py -m pip install -r requirements.txt```

Linux/MacOS: ```pip3 install -r requirements.txt```


# References
<a id="1">[1]</a>
Ian J. Goodfellow and Oriol Vinyals and Andrew M. Saxe, 2015,
Qualitatively characterizing neural network optimization problems,
doi 1412.6544,
Available [Here](https://arxiv.org/abs/1412.6544)

<a id="2">[2]</a>
Hao Li and Zheng Xu and Gavin Taylor and Christoph Studer and Tom Goldstein, 2018,
Visualizing the Loss Landscape of Neural Nets,
doi 1712.09913,
Available [Here](https://arxiv.org/abs/1712.09913)
