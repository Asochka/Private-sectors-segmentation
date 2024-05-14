import os
import pandas as pd
import geopandas as gpd

from glob import glob

import src.utils as utils
from src.geotif_predictor import GEOTIFPredictor
from src.rastr_predictor import RastrPredictor


INPUT_FOLDER = 'input'
OUTPUT_FOLDER = 'output'


if __name__ == '__main__':

    # Определение форматов входных данных
    geotif_files = []
    files = []
    for f in glob(os.path.join(INPUT_FOLDER, '*')):
        if utils.is_geotif(f):
            geotif_files.append(f)
        else:
            files.append(f)

    # Блок прогноза geotif
    predictor = GEOTIFPredictor()
    gdf_d = {f:predictor.process_image_optimized(f) for f in geotif_files}
    if gdf_d:
        combined_data = gpd.GeoDataFrame(pd.concat(gdf_d.values(), ignore_index=True))
        combined_data = utils.combined_polygons_gdfs(combined_data)
        combined_data = utils.delete_small_objects(combined_data, area_th=10)
        combined_data = utils.apply_rdp_to_gdf(combined_data, tolerance=0.000001)
        combined_data.to_file(os.path.join(OUTPUT_FOLDER, 'geo_predict.gpkg'), driver="GPKG")

    # Блок прогноза растровых изображений
    predictor = RastrPredictor()
    for f in files:
        mask = predictor.predict(f)
        f_name = os.path.basename(f).split('.')[0]
        mask.save(os.path.join(OUTPUT_FOLDER, f_name+'.png'))
        