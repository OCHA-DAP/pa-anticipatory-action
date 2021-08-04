import os
from pathlib import Path

DATA_DIR = Path(os.environ["AA_DATA_DIR"])
DATA_DIR_PUBLIC = DATA_DIR / "public" / "exploration"
DATA_DIR_PRIVATE = DATA_DIR / "private" / "exploration"

RCO_DIR = DATA_DIR_PRIVATE / "npl" / "unrco"
