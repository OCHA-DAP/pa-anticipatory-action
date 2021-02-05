import yaml
import argparse
import coloredlogs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('adm_level', help='Admin level to calculate flood fraction')
    args = parser.parse_args()
    return args


def parse_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def config_logger(level="INFO"):
    # Colours selected from here:
    # http://humanfriendly.readthedocs.io/en/latest/_images/ansi-demo.png
    coloredlogs.install(
        level=level,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        field_styles={
            "name": {"color": 8},
            "asctime": {"color": 248},
            "levelname": {"color": 8, "bold": True},
        },
    )