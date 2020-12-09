from scripts.d02_processing import FE_flood_extent as fe
from scripts import utils

parameters = utils.parse_yaml('config.yml')['DIRS']

shp_dir = parameters['shp_dir']
output_dir = parameters['data_dir']

if __name__ == "__main__":
    arg = utils.parse_args()
    fe.main(arg.adm_level, shp_dir, output_dir)
