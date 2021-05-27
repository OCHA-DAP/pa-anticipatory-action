```python
from pathlib import Path
import os

DATA_DIR = Path(os.environ["AA_DATA_DIR"]) 
DATA_DIR_PUBLIC = DATA_DIR / 'public/exploration'
DATA_DIR_PRIVATE = DATA_DIR / 'private/exploration'

GLOFAS_STATIONS_SHAPEFILE = DATA_DIR_PUBLIC / 'npl/glofas/npl_glofas_stations.gpkg'


```
