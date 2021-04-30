import torch
from lib import preliminary, prep
import PCA_directions
import random_directions
import linear
import quadratic
from lib.paths import *

logger = logging.getLogger("vis_net")

args = prep.parse_arguments()

if args.debug:
    lvl = logging.DEBUG
else:
    lvl = logging.INFO

logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    filename="vis_net.log")

init_dirs()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.debug(f"Device: {device}")

if args.auto:
    logger.info("Executing 1D experiments automatically")
    prep.run_all(args, device)

if args.single:
    logger.info("Executing interpolation of individual parameter experiment")
    linear.run_single(args, device)

if args.layers:
    linear.run_complete(args, device)

    logger.info("Executing interpolation of parameters of a layer experiment")
    linear.run_layers(args, device)

if args.quadratic:
    quadratic.run_complete(args, device)

    logger.info("Executing quadratic interpolation of individual parameter")
    quadratic.run_individual(args, device)
    quadratic.run_layers(args, device)

if args.preliminary:
    logger.info("Executing preliminary experiments")
    preliminary.run_preliminary(args)

if args.surface:
    logger.info("Executing random directions experiment")
    random_directions.run_rand_dirs(args)

if args.path:
    logger.info("Executing optimizer path visualization using PCA directions")
    PCA_directions.run_pca_surface(args, device)

if args.plot:
    logger.info("Plotting available data")
    prep.plot_available(args)