import argparse
from argparse import RawDescriptionHelpFormatter


def init_parser():
    parser = argparse.ArgumentParser(
        prog="train_cut.py",
        description=("A PyTorch implementation of Contrastive Unpaired "
                     "Translation (CUT).\n"
                     "CUT transfers the style of class B to an image from "
                     "class A.\n"
                     "For example, apply zebra style (B) to an image of a"
                     "horse (A)."),
        usage="python3 %(prog)s --trainA path/to/A --trainB path/to/B [options]",
        formatter_class=RawDescriptionHelpFormatter
    )

    # define training datasets
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--data", metavar="path", type=str, nargs=None,
                               help="path to training data folders",
                               required=True)
    required_args.add_argument("--trainA", metavar="str", type=str, nargs=None,
                               help="name of folder containing class A images",
                               required=True)
    required_args.add_argument("--trainB", metavar="str", type=str, nargs=None,
                               help="name of folder containing class B images",
                               required=True)

    # learning rates
    parser.add_argument("--D", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for discriminator network "
                              "(default 8e-8)"),
                        default=8e-8)
    parser.add_argument("--G", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for generator network "
                              "(default 5e-7)"),
                        default=5e-7)
    parser.add_argument("--P", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for patchNCE network "
                              "(default 2e-3)"),
                        default=2e-3)

    # print progress during training
    parser.add_argument("--epochs", metavar="int", type=int, nargs="?",
                        help="Number of epochs to train (default 400)",
                        default=400)

    parser.add_argument("--print", metavar="int", type=int, nargs="?",
                        help=("Print losses every `print_every` minibatches "
                              "(default once per epoch)."))

    # how often to save training checkpoints
    parser.add_argument("--save", metavar="int", type=int, nargs="?",
                        help=("Save models, optimizers, and losses every "
                              "`save` epochs (default 0)"),
                        default=0)

    # load checkpoints
    parser.add_argument("--loadD", metavar="path", type=str, nargs="?",
                        help=("path to .pt file for discriminator model state"))
    parser.add_argument("--loadG", metavar="path", type=str, nargs="?",
                        help=("path to .pt file for generator model state"))
    parser.add_argument("--loadP", metavar="path", type=str, nargs="?",
                        help=("path to .pt file for patchNCE model state"))
    parser.add_argument("--loadLoss", metavar="path", type=str, nargs="?",
                        help=("path to .pt file for list of minibatch losses"))

    # use gpu or cpu
    parser.add_argument("--device", metavar="str", type=str, nargs="?",
                        help="torch device (default cuda:0 if available)")

    return parser


if __name__ == "__main__":
    p = init_parser()
    p.print_help()
    args = p.parse_args()
