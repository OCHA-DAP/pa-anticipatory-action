import yaml
import argparse
import coloredlogs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("country_iso3", help="Country ISO3")
    parser.add_argument("-a", "--admin_level", default=1)
    # Prefix for filenames
    parser.add_argument(
        "-s",
        "--suffix",
        default="",
        type=str,
        help="Suffix for output files, and if applicable input files",
    )
    return parser.parse_args()


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
