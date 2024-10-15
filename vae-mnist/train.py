import torch
from torchvision import datasets
from torchvision.transforms import v2
from datetime import datetime
from tqdm import tqdm
import configargparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from openTSNE import TSNE
from torch.utils.tensorboard import SummaryWriter

from pytorchtools import EarlyStopping
from vae import VAE


def fix_seed(seed):
    # from https://discuss.pytorch.org/t/difference-between-torch-manual-seed-and-torch-cuda-manual-seed/13848/6
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random

    random.seed(seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 200  # 50
hidden_dim = 512
batch_size = 128

DATASET_INPUT_DIM = {
    "mnist": 784,
    "kmnist": 784,
    "fashion": 784,
    "cifar10": 1024,
}


def main():
    p = configargparse.ArgumentParser()
    p.add(
        "-c",
        "--config",
        required=False,
        is_config_file=True,
        help="Path to config file.",
    )
    # general
    p.add_argument("--dataset", type=str, default="mnist", help="Dataset to use")
    p.add_argument(
        "--latent_dim", type=int, default=20, help="Dimension of the latent embedding."
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for training")
    p.add_argument(
        "--beta", type=float, default=1.0, help="Weighting of kld loss term in VAE."
    )
    p.add_argument("--DEV", type=int, default=0, help="cuda device number")
    p.add_argument(
        "--delta", type=float, default=0.001, help="minimal early stopping improvement"
    )
    p.add_argument(
        "--patience", type=int, default=5, help="early stopping steps to tolerate"
    )
    opt = p.parse_args()

    print(f"[INFO] Latent dimension is set to {opt.latent_dim}.")
    fix_seed(opt.seed)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.contiguous().view(-1) - 0.5),
        ]
    )

    # Download and load the training data
    if opt.dataset == "mnist":
        train_data = datasets.MNIST(
            "MNIST_data/",
            download=True,
            train=True,
            transform=transform,
        )
        # Download and load the test data
        test_data = datasets.MNIST(
            "MNIST_data/",
            download=True,
            train=False,
            transform=transform,
        )
    elif opt.dataset == "kmnist":
        train_data = datasets.KMNIST(
            "KMNIST_data/",
            download=True,
            train=True,
            transform=transform,
        )
        test_data = datasets.KMNIST(
            "KMNIST_data/",
            download=True,
            train=False,
            transform=transform,
        )
    elif opt.dataset == "cifar10":
        transform.transforms.insert(2, v2.Grayscale(1))
        train_data = datasets.CIFAR10(
            "CIFAR10_data/",
            download=True,
            train=True,
            transform=transform,
        )
        test_data = datasets.CIFAR10(
            "CIFAR10_data/",
            download=True,
            train=False,
            transform=transform,
        )
    elif opt.dataset == "fashion":
        train_data = datasets.FashionMNIST(
            "FashionMNIST_data/",
            download=True,
            train=True,
            transform=transform,
        )
        test_data = datasets.FashionMNIST(
            "FashionMNIST_data/",
            download=True,
            train=False,
            transform=transform,
        )
    elif opt.dataset == "fashion":
        train_data = datasets.FashionMNIST(
            "FashionMNIST_data/",
            download=True,
            train=True,
            transform=transform,
        )
        test_data = datasets.FashionMNIST(
            "FashionMNIST_data/",
            download=True,
            train=False,
            transform=transform,
        )
    # elif opt.dataset == "imagenette":
    #     # TODO - think about size here?
    #     # size (string, optional) – The image size. Supports "full" (default), "320px", and "160px"
    #     train_data = datasets.Imagenette(
    #         'Imagenette_data/',
    #         download=True,
    #         train=True,
    #         transform=transform,
    #     )
    #     test_data = datasets.Imagenette(
    #         'Imagenette_data/',
    #         download=True,
    #         train=False,
    #         transform=transform,
    #     )
    else:
        raise NotImplementedError("Dataset is not implemented")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device(f"cuda:{opt.DEV}" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=DATASET_INPUT_DIM[opt.dataset],
        hidden_dim=hidden_dim,
        latent_dim=opt.latent_dim,
        beta=opt.beta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    run_name = f'runs/{opt.dataset}_till_convergence/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{opt.latent_dim}_seed_{opt.seed}_beta_{str(opt.beta).replace(".", "")}_patience_{opt.patience}_delta_{str(opt.delta).replace(".", "")}'
    writer = SummaryWriter(run_name)
    early_stopping = EarlyStopping(
        patience=opt.patience,
        delta=opt.delta,
        path=f"{run_name}/checkpoint.pth",
        verbose=True,
    )
    prev_updates = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        prev_updates = train(
            model, train_loader, optimizer, prev_updates, device, writer=writer
        )

        valid_loss = test(
            model,
            test_loader,
            prev_updates,
            device,
            opt,
            writer=writer,
            out_dim=int(np.sqrt(DATASET_INPUT_DIM[opt.dataset])),
        )
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        visualize_latents(
            model,
            test_loader,
            prev_updates,
            device,
            opt,
            run_name,
            writer=writer,
            save=False,
        )

    # save latent embeddings
    visualize_latents(
        model,
        test_loader,
        prev_updates,
        device,
        opt,
        run_name,
        writer=writer,
        save=True,
    )
    torch.save(model.state_dict(), f"{run_name}/model_final_epoch.pth")


def train(model, dataloader, optimizer, prev_updates, device, writer=None):
    """
    Trains the model on the given data.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        output = model(data)  # Forward pass
        loss = output.loss

        loss.backward()

        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)

            print(
                f"Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}"
            )

            if writer is not None:
                global_step = n_upd
                writer.add_scalar("Loss/Train", loss.item(), global_step)
                writer.add_scalar(
                    "Loss/Train/BCE", output.loss_recon.item(), global_step
                )
                writer.add_scalar("Loss/Train/KLD", output.loss_kl.item(), global_step)
                writer.add_scalar("GradNorm/Train", total_norm, global_step)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # Update the model parameters

    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, device, opt, writer=None, out_dim=28):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Testing"):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=True)  # Forward pass

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(
        f"====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})"
    )

    if writer is not None:
        writer.add_scalar("Loss/Test", test_loss, global_step=cur_step)
        writer.add_scalar(
            "Loss/Test/BCE", output.loss_recon.item(), global_step=cur_step
        )
        writer.add_scalar("Loss/Test/KLD", output.loss_kl.item(), global_step=cur_step)

        # Log reconstructions
        writer.add_images(
            "Test/Reconstructions",
            output.x_recon.view(-1, 1, out_dim, out_dim),
            global_step=cur_step,
        )
        writer.add_images(
            "Test/Originals", data.view(-1, 1, out_dim, out_dim), global_step=cur_step
        )

        # Log random samples from the latent space
        z = torch.randn(16, opt.latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images(
            "Test/Samples", samples.view(-1, 1, out_dim, out_dim), global_step=cur_step
        )
    return test_loss


def visualize_latents(
    model, dataloader, cur_step, device, opt, run_name, writer=None, save=False
):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode

    latents = []
    targets = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Visualizing"):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=False)  # Forward pass

            z = output.z_sample

            # save z:target
            latents.append(z.cpu())
            targets.append(target.cpu())

    # scatterplot z colored with target
    latents = torch.cat(latents)
    targets = torch.cat(targets)

    if save:
        df = pd.DataFrame(data={"latent": list(np.array(latents)), "label": targets})
        df.to_pickle(
            join(run_name, f"vae_embeddings_latent_dim_{str(opt.latent_dim)}.pkl")
        )

    if opt.latent_dim > 2:
        tsne = TSNE(
            perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=True
        )
        latents = tsne.fit(latents)

    fig = plt.figure(figsize=(15, 5))
    u_labels = np.unique(targets)
    colors = (
        plt.cm.tab10(np.arange(len(u_labels)))
        if len(u_labels) <= 10
        else plt.cm.tab20(np.arange(len(u_labels)))
    )
    for label, color in zip(u_labels, colors):
        plt.scatter(
            latents[targets == label, 0],
            latents[targets == label, 1],
            label=label,
            color=color,
        )
    plt.legend(bbox_to_anchor=(1, 1))

    if writer is not None:
        # Log reconstructions
        writer.add_figure("Latent distribution", fig, global_step=cur_step)


if __name__ == "__main__":
    main()
