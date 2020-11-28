from .models.GAN import *
from .models.PatchNCE import  *

args = None

# Hyperparameters
discriminator_learning_rate = args.D if args.D else 2e-3
generator_learning_rate = args.G if args.G else 2e-3
patchNCE_learning_rate = args.P if args.P else 2e-3

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
