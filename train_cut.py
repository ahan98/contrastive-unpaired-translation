import torch
from training.checkpoint_utils import load_models_and_losses
from training.train_utils import make_dataloader, get_batches_from_path
from training.train import train
from training.init_parser import init_parser
from training.checkpoint_utils import save_models, save_losses

parser = init_parser()
args = parser.parse_args()
# parser.print_help()

if args.device:
    device = args.device
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device", device)

# define path to image folders
path_to_train = args.data
X_class_name = args.trainA
Y_class_name = args.trainB

print("Loading training data...")
batches_by_class = get_batches_from_path(path_to_train)
X_train_dataloader = make_dataloader(batches_by_class, X_class_name)
Y_train_dataloader = make_dataloader(batches_by_class, Y_class_name,
                                     replacement=True)

# Hyperparameters
lr_D = args.lr_D if args.lr_D else 8e-8
lr_G = args.lr_G if args.lr_G else 5e-7
lr_P = args.lr_P if args.lr_P else 2e-3

# init models
print("Initializing models...")
models_dict, loss_per_minibatch = \
    load_models_and_losses(lr_discriminator=lr_D, lr_generator=lr_G,
                           lr_patchNCE=lr_P, device=device)

n_epochs = args.epochs if args.epochs else 400
print_every = args.print if args.print else len(X_train_dataloader)
checkpoint_epoch = args.checkpoint if args.checkpoint else 0

models_dict, loss_per_minibatch = \
    train(models_dict, loss_per_minibatch, X_train_dataloader,
          Y_train_dataloader, device=device, n_epochs=n_epochs,
          print_every=print_every, checkpoint_epoch=checkpoint_epoch)

# Save final states after training
print("Training completed.")
save_models(models_dict, n_epochs)
save_losses(loss_per_minibatch, n_epochs)
