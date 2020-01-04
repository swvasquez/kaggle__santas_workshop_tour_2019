import pathlib


def project_paths():
    """Allows files in module know where to find the project config. file."""
    root_path = pathlib.Path(__file__).parents[1]
    config_path = root_path / 'config.yaml'
    return {'config': config_path, 'root': root_path}
