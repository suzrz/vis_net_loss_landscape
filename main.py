"""https://github.com/okn-yu/Visualizing-the-Loss-Landscape-of-Neural-Nets"""
import os
import net
import plot
import torch
import argparse
import data_load
import directions
import calculate_loss
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR


def main():
    """PARSE ARGUMENTS"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--interpolation-samples", type=int, default=13,
                        help="Set number of interpolation samples (default = 13)")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--single-param-only", action="store_true",
                        help="Only one single parameter will be interpolated")
    parser.add_argument("--two-params-only", action="store_true",
                        help="Only two parameters will be interpolated")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Set directory to which will be saved trained model and computed "
                             "loss/accuracy (default = \"results\")")
    parser.add_argument("--epochs", type=int, default=14, help="Set number of training epochs (default = 14)")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    init_state = os.path.join(args.results_dir, "init_state.pt")
    final_state = os.path.join(args.results_dir, "final_state.pt")

    """CREATE INSTANCE OF NEURAL NETWORK"""
    model = net.Net().to(device)

    """SAVE INIT STATE OF MODEL"""
    # Check if directory for results exists
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    # If initialized state does not exist, create new one. Else skip.
    if not os.path.isfile(init_state):
        torch.save(model.state_dict(), init_state)
        print("New initial state saved.")

    """PREPARE MODEL"""
    model.load_state_dict(torch.load(init_state))  # load initialized state from file

    torch.manual_seed(1)  # set seed

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # Get differences in loss and accuracy between different sizes of train subset
    if args.hist:
        calculate_loss.get_diff_in_success_for_subsets(model, optimizer, scheduler, device,
                                                       args.results_dir, args.epochs)
        plot.plot_subset_hist()

    # Get train and test data loader
    train_loader, test_loader = data_load.data_load()
    """TRAIN MODEL"""
    # check if exists trained model params
    if not os.path.isfile(final_state):
        print("Final state not found - beginning training")
        for epoch in range(1, args.epochs):
            net.train(model, train_loader, optimizer, device, epoch)
            net.test(model, test_loader, device)
            scheduler.step()
            print("Finished epoch no. ", epoch)

        torch.save(model.state_dict(), final_state)  # save final parameters of model

    model.load_state_dict(torch.load(init_state))

    """PREPARE FOR PLOT"""
    if not args.two_params_only:
        # prepare files for 2D plot
        calculate_loss.single(model, train_loader, test_loader, device, args.interpolation_samples,
                              optimizer, args.results_dir, final_state, init_state)

        """PLOT"""
        plot.plot_accuracy(args.interpolation_samples, args.results_dir)
        plot.plot_2d_loss(args.interpolation_samples, args.results_dir)

    if not args.single_param_only:
        # prepare files for 3D plot
        dirs = directions.random_directions(model)  # get random directions
        calculate_loss.double(model, test_loader, dirs, device, args.results_dir)  # calculate validation loss
        # plot
        plot.surface3d_rand_dirs(args.results_dir)


if __name__ == "__main__":
    main()  # run
