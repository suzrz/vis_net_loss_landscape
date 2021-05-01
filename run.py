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
logger.info("========NEURAL NETWORK TRAINING PROGRESS VISUALIZATION TOOL========")

init_dirs()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.debug(f"Device: {device}")

if args.auto:
    logger.info(f"Executing one dimensional experiments automatically. (Number of samples: {args.auto_n})")
    prep.run_all(args, device)

if args.linear_i:
    logger.info(f"Executing parameter level linear path experiment. (Parameter {args.layer} {args.idxs})")
    linear.run_individual(args, device)

if args.linear_q:
    logger.info(f"Executing layer level linear path experiment. (Layer {args.layer})")
    linear.run_complete(args, device)

    linear.run_layer(args, device)

if args.quadratic_i:
    logger.info(f"Executing parameter level quadratic path experiment. (Parameter {args.layer} {args.idxs})")
    quadratic.run_individual(args, device)

if args.quadratic_l:
    logger.info(f"Executing layer level quadratic path experiment. (Layer {args.layer})")
    quadratic.run_complete(args, device)

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