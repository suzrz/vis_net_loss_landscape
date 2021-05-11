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
Warning: The installation can take around 5GB of disk space.

Make sure that you are using the right version of Python. In Windows OS you can use option 
```py -v <version> ...``` and in Linux/MacOS ```python<version> ...``` to use version 
```<version>``` of Python.

## Virtual environment setup
Because of use of various Python Packages it is recommened to use virtual environment. 

Create virtual environment:

Windows: ```py -m venv <path_to_venv>```

Linux/MacOS: ```python3 -m venv <path_to_venv>```

Activate virtual environment:

Windows: ```<path_to_venv>\Scripts\activate```

Linux/MacOS: ```source <path_to_venv>/bin/activate```

Deactivation of the virtual environment:

Windows/Linux/MacOS: ```deactivate```

More details about virtual environments can be found [here](https://docs.python.org/3/library/venv.html).

### Using requirements file
Or you can use provided ```requirements.txt``` file: 

Windows:
```py -m pip install -r requirements.txt```

Linux/MacOS: ```pip3 install -r requirements.txt```

### Using pip
You can use pip to install the nnvis package:

Windows: ```py -m pip install nnvis```

Linux/MacOS: ```pip3 install nnvis```

# Usage
For help run:

Windows: ```py run.py -h```

Linux/MacOS: ```python3 run.py -h```

#### Logging
The tool uses ```<path_to_project>/vis_net.log``` to log the progress of the usage and it 
shows progress bars on terminal.

## Examples
The examples are demonstrating the use of tool on a simple CNN model based on LeNet 5, trained and validated on MNIST 
dataset.

#### Computational demands:
It takes around three minutes to examine one parameter using single-dimensional methods (linear and quadratic path) on 
a machine with Intel Core i5-6600K and NVIDIA GTX 1060 6GB with CUDA enabled.

#### Execution
To auto run all one dimensional experiments it is recommended to set number of examined parameters in each layer. The 
default value is ```10```, you can set the number using option ```--auto-n [INT]```.

Windows: ```py run.py --auto --auto-n [INT]```

Linux/MacOS: ```python3 run.py --auto --auto-n [INT]```


To visualize the surface of the loss function run:

Windows: ```py run.py --surface```

Linux/MacOS: ```python3 run.py --surface```

To visualize the path of SGD:

Windows: ```py run.py --path```

Linux/MacOS: ```python3 run.py --path```

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
