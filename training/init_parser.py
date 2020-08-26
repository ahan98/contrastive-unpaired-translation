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

    # training datasets
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--data", metavar="str", type=str, nargs=None,
                               help="path containing training data folders",
                               required=True)
    required_args.add_argument("--trainA", metavar="str", type=str, nargs=None,
                               help="name of folder containing class A images",
                               required=True)
    required_args.add_argument("--trainB", metavar="str", type=str, nargs=None,
                               help="name of folder containing class B images",
                               required=True)

    # learning rates
    parser.add_argument("--lr_D", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for discriminator network "
                              "(default 8e-8)"),
                        const=1, default=8e-8)
    parser.add_argument("--lr_G", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for generator network "
                              "(default 5e-7)"),
                        const=1, default=5e-7)
    parser.add_argument("--lr_P", metavar="float", type=float, nargs="?",
                        help=("Adam learning rate for patchNCE network "
                              "(default 2e-3)"),
                        const=1, default=2e-3)

    # print progress during training
    parser.add_argument("--epochs", metavar="int", type=int, nargs="?",
                        help="Number of epochs to train (default 400)",
                        const=1, default=400)

    parser.add_argument("--print", metavar="int", type=int, nargs="?",
                        help=("Print losses every `print_every` minibatches "
                              "(default once per epoch)."))

    # checkpoint training states
    parser.add_argument("--checkpoint", metavar="int", type=int, nargs="?",
                        help=("Save models, optimizers, and losses every "
                              "`checkpoint` epochs (default 0)"),
                        const=1, default=0)

    parser.add_argument("--device", metavar="str", type=str, nargs="?",
                        help="torch device (default cuda:0 if available)")

    # parser.print_help()
    return parser
