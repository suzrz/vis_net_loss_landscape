import prep
import preliminary
import layer_params
import individual_param
import random_directions
import quadr_interpolation
import l_interpolate_all
import q_interpolate_all
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

logger.info(f"Executing interpolation of whole model.")
l_interpolate_all.run_complete_interpolation(args)
#q_interpolate_all.run_complete_quadratic_interpolation(args)

if args.auto:
    logger.info("Executing experiments automatically")
    prep.run_all(args)

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
