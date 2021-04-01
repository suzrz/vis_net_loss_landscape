import prep
import individual_param
import layer_params
from paths import *


args = prep.parse_arguments()

init_dirs()

if args.single:
    individual_param.run_single(args)

if args.layers:
    layer_params.run_layers(args)