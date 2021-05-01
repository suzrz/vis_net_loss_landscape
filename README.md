# Neural Network Training Progress Visualization
This tool provides a visualization of the training progress of a neural network model. It uses the Linear Path 
technique introduced by Goodfellow et al. in  [[1]](#1), Quadratic Path introduced in this work and PCA projection 
of a loss landscape, introduced by Hao Li et al. in [[2]](#2).

# Prerequisities
The tool requires following dependencies to run:
- Python>=3.8 (Available [here](https://www.python.org/downloads/))
- pip>=20 (pip is installed together with Python>=3.4, upgrade pip before installing packages: [guide](https://pip.pypa.io/en/stable/installing/))

If CUDA will be used, then also:
- CUDA>=11.1 (Available [here](https://developer.nvidia.com/cuda-downloads))

# Installation
For running this tool it is necessary to have installed PyTorch. Recommended way of installation is from the project 
[website](https://pytorch.org/get-started/locally/), where it is possible to get CUDA version if you have a CUDA GPU.

All required Python packages can be installed with: 

Windows:
```py -m pip install -r requirements.txt```

Linux/MacOS: ```pip3 install -r requirements.txt```

# Usage
For help run:

Windows: ```py run.py -h```

Linux/MacOS: ```python3 run.py -h```

## Examples
For auto run all one dimensional experiments run:

Windows: ```py run.py --auto```

Linux/MacOS: ```python3 run.py --auto```

This will run linear and quadratic path experiments on the level of model, layer and parameter. Option ```--auto-n INT``` 
specifies number of randomly selected parameters to examine in each layer (e. g. ```--auto-n 10``` will examine 10 
randomly selected parameters). 

To examine the surface of the loss function run:

Windows: ```py run.py --surface```

Linux/MacOS: ```python3 run.py --surface```

This will execute the random directions surface examination.

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
