import prep
import preliminary
import layer_params
import individual_param
from paths import *


args = prep.parse_arguments()

logger = logging.getLogger("vis_net")

if args.debug:
    lvl = logging.DEBUG
else:
    lvl = logging.INFO

logging.basicConfig(level=lvl, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    filename="vis_net.log")

init_dirs()

if args.single:
    logger.info("Executing interpolation of individual parameter experiment")
    individual_param.run_single(args)

if args.layers:
    logger.info("Executing interpolation of parameters of a layer experiment")
    layer_params.run_layers(args)

if args.preliminary:
    logger.info("Executing preliminary experiments")
    preliminary.run_preliminary(args)