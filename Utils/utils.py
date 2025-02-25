import yaml
from argparse import Namespace

def load_args_from_file(config_path):
    """
    Load arguments from a YAML file and convert them into a Namespace.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Namespace: Arguments loaded as a Namespace object.
    """
    # Load YAML config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Convert dictionary to Namespace
    args = Namespace(**config)
    return args
