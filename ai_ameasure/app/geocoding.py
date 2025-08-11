import requests
import pandas as pd

MUNICODE_LIST = pd.read_csv('app/master/municode.csv', index_col='id').to_dict(orient='index')


def reverse_geocoding(lons: list, lats: list):

    names = []
    for lon, lat in zip(lons, lats):
        endpoint_url = 'https://mreversegeocoder.gsi.go.jp/reverse-geocoder/LonLatToAddress'
        params={
            'lat':lat,
            'lon':lon
        }
        res = requests.get(endpoint_url, params=params)
        data = res.json()
        d = data['results']
        municde_area = MUNICODE_LIST[int(d['muniCd'])]
        name = f"{municde_area['chiriin_pref_name']} {municde_area['chiriin_city_name']}"
        names.append(name)

    return names