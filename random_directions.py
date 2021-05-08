import prep
import torch
import nnvis


def run_rand_dirs(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    dirs = nnvis.get_directions(model, device)
    nnvis.calc_loss(model, test_loader, dirs, device)
    nnvis.surface3d_rand_dirs()
    nnvis.surface_heatmap_rand_dirs()
