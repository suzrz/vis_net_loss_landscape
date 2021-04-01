import os
import sys
import numpy as np
import prep
import torch
import data_load
import random
import subprocess
import individual_param
from pathlib import Path
from paths import *


args = prep.parse_arguments()

init_dirs()

if args.single:
    individual_param.run_single(args)
