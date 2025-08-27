import argparse
import unicodedata
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


ATTRIBUTE_COLUMNS = ['ﾎﾟｲﾝﾄ名称', 'TD(m)']

def get_xyz_columns(df):

    for col in df.columns:
        col_unicode = unicodedata.normalize('NFKC', col.lower())
        if 'x' in col_unicode:
            x_col = col
            continue
        if 'y' in col_unicode:
            y_col = col
            continue
        if 'z' in col_unicode or '高さ' in col_unicode:
            z_col = col

    return [x_col, y_col, z_col]

def load_alignment_as_4326gdf(csv_path, epsg):
    # Read CSV
    df = pd.read_csv(csv_path, encoding='shift-jis')
    try:
        xyz_names = get_xyz_columns(df)
    except:
        raise ValueError("Columns with x, y, and z coordinates not found in CSV.")

    # Create geometry from 3D points
    df['geometry'] = df.apply(lambda row: Point([(row[xyz_names])]), axis=1)
    
    # Create GeoDataFrame with attributes and geometry
    gdf = gpd.GeoDataFrame(data=df[ATTRIBUTE_COLUMNS], geometry=df['geometry'])
    gdf.set_crs(epsg=epsg, inplace=True)
    gdf.to_crs(epsg=4326, inplace=True)
    
    return gdf

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a LineString from a CSV file and plot it on a map.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing coordinates.")
    parser.add_argument("epsg", type=int, help="EPSG code of the coordinate system used in the CSV file.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create LineString and plot on map
    load_alignment_as_4326gdf(args.csv_path, args.epsg)
    

if __name__ == "__main__":
    main()
