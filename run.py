import prep
import linear
import quadratic
import preliminary
import random_directions
import PCA_directions
import nnvis
import torch

args = prep.parse_arguments()

nnvis.init_dirs()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.auto:
    prep.run_all(args, device)

if args.single:
    linear.run_single(args, device)

if args.layers:
    linear.run_complete(args, device)

    linear.run_layers(args, device)

if args.quadratic:
    quadratic.run_complete(args, device)

    quadratic.run_individual(args, device)
    quadratic.run_layers(args, device)

if args.preliminary:
    preliminary.run_preliminary(args, device)

if args.surface:
    random_directions.run_rand_dirs(args)

if args.path:
    PCA_directions.run_pca_surface(args, device)

if args.plot:
    prep.plot_available(args)
