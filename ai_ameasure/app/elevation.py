import os
import srtm
import gpxpy
import gpxpy.gpx


LOCAL_CACHE_DIR = "srtm_cache"
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

def convert_to_gpx(latitudes, longitudes, output_filename):
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for lat, lon in zip(latitudes, longitudes):
        point = gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon)
        gpx_segment.points.append(point)

    with open(output_filename, 'w') as f:
        f.write(gpx.to_xml())

def get_elevation_from_dem(lons: list, lats: list) -> list:

    #gpx_path = os.path.join(LOCAL_CACHE_DIR, "temp.gpx")
    #convert_to_gpx(lats, lons, gpx_path)
    #gpx = gpxpy.parse(open(gpx_path))
    elevation_data = srtm.get_data(local_cache_dir=LOCAL_CACHE_DIR)
    #elevation_data.add_elevations(gpx, smooth=True)
    elevations = []
    for lat, lon in zip(lats, lons):
        elevation = elevation_data.get_elevation(lat, lon)
        elevations.append(elevation)

    return elevations