import json
import click
import mlflow
import shutil
import os
import traceback

from ml_ids.models.gradient_boost.model import ModelWrapper


def merge(dict1, dict2):
    """
    Merge two dictionaries by creating copies of the dictionaries.
    :param dict1: First dictionary to merge
    :param dict2: Second dictionary to merge
    :return: Merged dictionary
    """
    d = dict(dict1)
    d.update(dict2)
    return d


@click.command()
@click.option('--train-path', type=click.Path(exists=True), required=True,
              help='Path to the training dataset (.h5 or .csv).')
@click.option('--val-path', type=click.Path(exists=True), required=True,
              help='Path to the validation dataset (.h5 or .csv).')
@click.option('--test-path', type=click.Path(exists=True), required=True,
              help='Path to the test dataset (.h5 or .csv).')
@click.option('--output-path', type=click.Path(), required=True,
              help='Path to store the output files.')
@click.option('--param-path', type=click.Path(exists=True), required=True,
              help='Path to the training parameters JSON file.')
def train(train_path, val_path, test_path, output_path, param_path):
    """Train Gradient Boosting model with MLflow tracking."""
    print("🚀 train.py started")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📦 Paths received:\n"
          f"  Train: {train_path}\n"
          f"  Val:   {val_path}\n"
          f"  Test:  {test_path}\n"
          f"  Output: {output_path}\n"
          f"  Params: {param_path}\n")

    # ✅ Load params
    try:
        with open(param_path, 'r') as f:
            params = json.load(f)
        print("✅ Parameters file loaded successfully.")
    except Exception as e:
        print("❌ Failed to load parameters JSON:")
        print(traceback.format_exc())
        return

    # ✅ Prepare output folder
    try:
        if os.path.exists(output_path):
            print(f"⚠️ Removing existing output directory: {output_path}")
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)
        print(f"📂 Output directory ready: {output_path}")
    except Exception as e:
        print("❌ Failed to create or clean output folder:")
        print(traceback.format_exc())
        return

    # ✅ Merge params for MLflow run
    run_params = merge(params, {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path,
        'output_path': output_path,
        'artifact_path': output_path,
    })

    print("\n🚀 Starting MLflow run with parameters:")
    print(json.dumps(run_params, indent=2))

    # ✅ Run MLflow project
    try:
        mlflow.run('models/gradient_boost/project',
                   parameters=run_params,
                   env_manager="local")
        print("🎯 MLflow run finished successfully!")
    except Exception as e:
        print("❌ MLflow run failed with error:")
        print(traceback.format_exc())
        return

    print("✅ Script finished without errors.")


if __name__ == '__main__':
    train()
