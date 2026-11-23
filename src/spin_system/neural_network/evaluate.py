"""Command-line workflow for evaluating trained neural-network models."""

import argparse

import numpy as np

from spin_system import FileManager
from spin_system.core.cli import CompactHelpFormatter
from spin_system.generation.utils.plotter import Plotter
from .evaluation.evaluator import Evaluator


def percentile_mosaics(
    evaluator: Evaluator,
    model_save_name: str,
    test_set_name: str,
) -> None:
    """Create separate mosaics for the best 90% and worst 10% samples.

    Args:
        evaluator: Evaluator containing predictions, targets, and output paths.
        model_save_name: Name of the trained model checkpoint.
        test_set_name: Name of the test-set configuration.

    Returns:
        None: Saves mosaics for both prediction-quality subsets.
    """
    from .evaluation.error_analysis import filter_predictions

    good_path, bad_path = filter_predictions(
        evaluator=evaluator,
        test_set_name=test_set_name,
        model_save_name=model_save_name,
        output_dir=evaluator.save_path / "good-bad",
        file_manager=evaluator.file_manager,
    )
    plot_args = argparse.Namespace(
        config=test_set_name,
        first_sim="sim_000001",
        last_sim="sim_999999",
        sim_name=None,
        dpi=300,
        cmap="RdBu",
        ratio=[15, 15],
    )
    for subset_name, selection_path in (("good", good_path), ("bad", bad_path)):
        sim_names = np.load(selection_path)
        plotter = Plotter(
            plot_args,
            selected_sim_names=sim_names,
            file_manager=evaluator.file_manager,
        )
        plotter.plot_mosaic(
            output_name=f"{subset_name}_mosaic.png",
            output_dir=evaluator.save_path / "good-bad",
        )


def main(arguments=None, paths=None) -> int:
    """Run the neural-network evaluation command-line workflow.

    Args:
        arguments: Optional command-line argument sequence.
        paths: Optional project paths used by the file manager.

    Returns:
        int: Exit status, where zero indicates successful evaluation.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate a trained neural network model on simulation data.",
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
        help="Name of the model checkpoint file to load (without .pt extension)",
        required=True
    )

    parser.add_argument(
        "--test-set-name",
        "-t",
        help="Name of the test dataset (without .npy extension)",
        required=True
    )

    parser.add_argument(
        "--data",
        "-d",
        nargs="+",
        default=["grids"],
        choices=["grids", "magnitude"],
        metavar="SOURCE",
        help="Data sources to evaluate on: grids, magnitude. "
             "Must match the sources used during training."
    )

    args = parser.parse_args(arguments)

    evaluator = Evaluator(
        config_name=args.config,
        model_save_name=args.model_save_name,
        test_set_name=args.test_set_name,
        data_sources=args.data,
        file_manager=FileManager(paths=paths),
    )

    print("-" * 50)

    evaluator.plot_loss_curves()
    evaluator.plot_raw_loss_curves()
    evaluator.plot_total_loss_curve()
    evaluator.eval_pred_vs_true()
    evaluator.print_evaluation_metrics()
    evaluator.one_v_one_scatter_plot()
    evaluator.mae_all_combinations_plot()
    evaluator.good_bad_target_combinations_plot()
    evaluator.abs_error_histograms()
    evaluator.plot_single_loss_curves()

    percentile_mosaics(
        evaluator=evaluator,
        model_save_name=args.model_save_name,
        test_set_name=args.test_set_name,
    )
    return 0


if __name__ == "__main__":
    main()