import prep
import preliminary
import itertools
import layer_params
import individual_param
import random_directions
import quadr_interpolation
import numpy as np
from paths import *

logger = logging.getLogger("vis_net")

args = prep.parse_arguments()

if args.debug:
    lvl = logging.DEBUG
else:
    lvl = logging.INFO

logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    filename="vis_net.log")

init_dirs()

args.auto = False  # TODO vyresit proc se pokazde spousti (mam tam default hodnotu...)
if args.auto:
    logger.info("Executing experiments automatically")
    aux = [list(np.arange(0, 6)), [0], list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv1_idxs = list(itertools.product(*aux))
    args.layer = "conv1"
    layer_params.run_layers(args)
    count = 0
    for i in conv1_idxs:
        args.idxs = i
        logger.debug(f"Layer: {args.layer}, idxs: {args.idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)
        count += 1
        if count > args.auto:
            break

    aux = [list(np.arange(0, 6)), list(np.arange(0, 6)), list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv2_idxs = list(itertools.product(*aux))
    args.layer = "conv2"
    layer_params.run_layers(args)
    count = 0
    for i in conv2_idxs:
        args.idxs = i
        logger.debug(f"Layer: {args.layer}, idxs: {args.idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)
        count += 1
        if count > args.auto:
            break

    aux = [list(np.arange(0, 120)), list(np.arange(0, 576))]
    fc1_idxs = list(itertools.product(*aux))
    count = 0
    args.layer = "fc1"
    layer_params.run_layers(args)
    for i in fc1_idxs:
        args.idxs = i
        logger.debug(f"Layer: {args.layer}, idxs: {args.idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)
        count += 1
        if count > args.auto:
            break

    aux = [list(np.arange(0, 84)), list(np.arange(0, 120))]
    fc2_idxs = list(itertools.product(*aux))
    count = 0
    args.layer = "fc2"
    layer_params.run_layers(args)
    for i in fc2_idxs:
        args.idxs = i
        logger.debug(f"Layer: {args.layer}, idxs: {args.idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)
        count += 1
        if count > args.auto:
            break

    aux = [list(np.arange(0, 10)), list(np.arange(0, 84))]
    fc3_idxs = list(itertools.product(*aux))
    count = 0
    args.layer = "fc3"
    layer_params.run_layers(args)
    for i in fc3_idxs:
        args.idxs = i
        logger.debug(f"Layer: {args.layer}, idxs: {args.idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)
        count += 1
        if count > args.auto:
            break

if args.single:
    logger.info("Executing interpolation of individual parameter experiment")
    individual_param.run_single(args)

if args.layers:
    logger.info("Executing interpolation of parameters of a layer experiment")
    layer_params.run_layers(args)

if args.quadratic:
    logger.info("Executing quadratic interpolation of individual parameter")
    quadr_interpolation.run_quadr_interpolation(args)

if args.preliminary:
    logger.info("Executing preliminary experiments")
    preliminary.run_preliminary(args)

if args.surface:
    logger.info("Executing random directions experiment")
    random_directions.run_rand_dirs(args)
