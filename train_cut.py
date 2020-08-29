import torch
from training.checkpoint_utils import load_models_and_losses
from training.train_utils import make_dataloader, get_batches_from_path
from training.train import train
from training.init_parser import init_parser
from training.checkpoint_utils import save_models, save_losses

parser = init_parser()
args = parser.parse_args()

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
lr_D = args.D if args.D else 2e-3
lr_G = args.G if args.G else 2e-3
lr_P = args.P if args.P else 2e-3

# init models
print("Initializing models...")
checkpoint_files_dict = None
if args.loadD and args.loadG and args.loadP and args.loadLoss:
    checkpoint_files_dict = {
        "discriminator": args.loadD,
        "generator": args.loadG,
        "patchNCE": args.loadP,
        "loss": args.loadLoss
    }

models_dict, loss_per_minibatch = \
    load_models_and_losses(lr_discriminator=lr_D, lr_generator=lr_G,
                           lr_patchNCE=lr_P, device=device,
                           checkpoint_files_dict=checkpoint_files_dict)

n_epochs = args.epochs if args.epochs else 400
print_every = args.print if args.print else len(X_train_dataloader)
checkpoint_epoch = args.save if args.save else 0

models_dict, loss_per_minibatch = \
    train(models_dict, loss_per_minibatch, X_train_dataloader,
          Y_train_dataloader, device=device, n_epochs=n_epochs,
          print_every=print_every, checkpoint_epoch=checkpoint_epoch)

# Save final states after training
print("Training completed.")
save_models(models_dict, n_epochs)
save_losses(loss_per_minibatch, n_epochs)
