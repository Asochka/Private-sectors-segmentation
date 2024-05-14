import os
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union


def apply_rdp_to_gdf(
        gdf: gpd.GeoDataFrame,
        tolerance: float = 0.00001,
        preserve_topology: bool = True
    ) -> gpd.GeoDataFrame:
    """
    Сглаживание
    """
    gdf_simplified = gdf.copy()
    gdf_simplified['geometry'] = gdf_simplified['geometry'].apply(
        lambda geom: geom.simplify(tolerance, preserve_topology)
    )
    return gdf_simplified


def delete_small_objects(gdf: gpd.GeoDataFrame, area_th: float = 10) -> gpd.GeoDataFrame:
    """
    Фильтрация полигонов по площади
    """
    gdf_projected = gdf.to_crs(epsg=32633)
    areas = gdf_projected['geometry'].area
    gdf['areas'] = areas
    return gdf[gdf['areas'] > area_th][['geometry']].reset_index(drop=True)


def combined_polygons_gdfs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Объединение (наложением) полигонов
    """
    combined_polygon = unary_union(gdf['geometry'])
    combined_gdf = gpd.GeoDataFrame(geometry=[combined_polygon], crs=gdf.crs)
    return combined_gdf.explode().reset_index(drop=True)
