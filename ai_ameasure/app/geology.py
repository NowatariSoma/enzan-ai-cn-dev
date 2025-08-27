import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon

GEOLOGY_LIST = pd.read_csv('app/master/seamlessV2/legend.tsv', sep='\t', index_col=0)
GEOLOGY_LIST.rename(columns={'formationAge_ja': '形成時代', 'group_ja': '大区分', 'lithology_ja': '岩相'}, inplace=True)
GEOLOGY_MAP = gpd.read_file('app/master/seamlessV2/seamlessV2_poly.shp', driver='ESRI Shapefile')


def get_geology(gdf):
    # Filter geology map by the area of geometries
    bounds = gdf.geometry.total_bounds
    gdf_geology_map_aoi = GEOLOGY_MAP.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    df_geology = pd.DataFrame()
    for geometry in gdf.geometry:
        for _, row in gdf_geology_map_aoi.iterrows():
            if geometry.within(row.geometry):
                symbol = row['symbol']
                geology = GEOLOGY_LIST[GEOLOGY_LIST['symbol']==symbol]
                df_geology = pd.concat([df_geology, geology])
                break
    gdf = pd.concat([gdf, df_geology.reset_index(drop=True)], axis=1)
    return gdf, df_geology.columns.to_list()
