import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--country", help="Country name")
    # parser.add_argument("-a", "--admin_level", default=1)
    # Prefix for filenames
    parser.add_argument(
        "-s",
        "--suffix",
        default="",
        type=str,
        help="Suffix for output files, and if applicable input files",
    )
    parser.add_argument(
        "-d",
        "--download-data",
        action="store_true",
        help=(
            "Download the raw data. FewsNet and WorldPop are currently"
            " implemented"
        ),
    )
    return parser.parse_args()
