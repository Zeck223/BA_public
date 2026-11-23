"""Command-line workflow for training spin-grid regression models."""

import argparse
import sys
import time
import random
import numpy as np
import torch

from spin_system import FileManager
from spin_system.core.config import NNConfigLoader
from spin_system.core.cli import CompactHelpFormatter
from spin_system.neural_network.data import SimulationDataService
from .artifacts.checkpoint import ModelCheckpoint
from .training.logger import NNLogger
from .training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set the random seeds and deterministic backend options.

    Args:
        seed: Seed applied to Python, NumPy, and PyTorch random generators.

    Returns:
        None: Configures the process-wide random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main(arguments=None, paths=None) -> int:
    """Run the neural-network training command-line workflow.

    Args:
        arguments: Optional command-line argument sequence.
        paths: Optional project paths used for configuration and data access.

    Returns:
        int: Exit status, where zero indicates successful training.
    """

    parser = argparse.ArgumentParser(
        description="Train a neural network model on simulation data.",
        formatter_class=CompactHelpFormatter
    )

    parser.add_argument(
        "--config",
        "-c",
        help="Name of the configuration (without .yaml extension)",
        required=True
    )

    parser.add_argument(
        "--model-save-name",
        "-m",
        help="Name of the model checkpoint file to save (without .pt extension)",
        required=True
    )

    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        default=["grids"],
        choices=["grids", "magnitude"],
        metavar="SOURCE",
        help="Data sources to use for training: grids, magnitude. "
             "Multiple sources are concatenated along the channel axis."
    )

    args = parser.parse_args(arguments)
    config_name = args.config
    model_save_name = args.model_save_name
    data_sources = args.data
    in_channels = len(data_sources) * 3
    file_manager = FileManager(paths=paths)
    data_service = SimulationDataService(
        config_name,
        file_manager=file_manager,
        data_sources=data_sources,
    )
    nn_config = NNConfigLoader(file_manager.paths).load(config_name)

    output_dir = file_manager.paths.model_directory(config_name, model_save_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(nn_config["seed"])

    output_file = output_dir / "output_stream.txt"

    # Redirect stdout and stderr
    tee_out = NNLogger(output_file, mode='w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_out
    sys.stderr = tee_out
    tee_out.initialization(output_file)

    config = data_service.get_config()

    # === Data Preprocessing (lazy — grids read on demand, only targets in RAM) ===
    train_size = nn_config["train_size"]
    scaler_path = output_dir / "scalers"
    target_cols, train_dataset, val_dataset = data_service.build_lazy_datasets(
        train_size=train_size,
        seed=nn_config["seed"],
        save_path=str(scaler_path),
    )

    # === Model Training ===
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        device=device,
        L=config["base_parameters"]["L"],
        num_out_features=len(target_cols),
        target_names=target_cols,
        epochs=nn_config["epochs"],
        batch_size=nn_config["batch_size"],
        patience=nn_config["patience"],
        lr=nn_config["lr"],
        weight_decay=nn_config["weight_decay"],
        min_lr=nn_config["min_lr"],
        warmup_epochs=nn_config["warmup_epochs"],
        dropout=nn_config["dropout"],
        print_every=nn_config["print_every"],
        in_channels=in_channels
    )

    model, optimizer, train_losses_per_var, val_losses_per_var, target_names = trainer.train(
        train_dataset, val_dataset
    )

    end_time = time.time()
    device_name = "CUDA" if device.type == "cuda" else "CPU"
    print(f"Training device: {device_name}, time: {end_time - start_time:.6f} seconds")

    # === Save the trained model ===
    checkpoint = ModelCheckpoint(model, optimizer)
    checkpoint.save(f"{output_dir}/checkpoint.pth", target_cols=target_cols, data_sources=data_sources)
    np.save(f"{output_dir}/train_losses_per_var.npy", train_losses_per_var)
    np.save(f"{output_dir}/val_losses_per_var.npy", val_losses_per_var)

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    tee_out.close()
    return 0


if __name__ == "__main__":
    main()




