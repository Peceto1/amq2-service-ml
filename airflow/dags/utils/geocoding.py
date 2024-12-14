import time

import pandas as pd
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from .utilities import load_geocode_results, split_uppercase_word, save_geocode_results

geolocator = Nominatim(user_agent="try2_ceia_add_weatherAus")


def generate_lat_long_features_from_location(weather_df):
    unique_places = weather_df['Location'].unique()

    unique_places_corrected = [split_uppercase_word(place) for place in unique_places]

    unique_places_corrected
    try:
        geocode_results_loaded = load_geocode_results()
    except:
      print("No se encontraron geocode results")
      geocode_results_loaded = {}
    if(geocode_results_loaded):
      geocode_results = geocode_results_loaded
    else:
      # Query the geocoding API for each unique place
      geocode_results = {}  # Declare the dictionary
      for place in unique_places_corrected:
          lat, lon = geocode_place(place)
          geocode_results[place] = {'latitude': lat, 'longitude': lon}
          save_geocode_results(geocode_results)


    geocode_results = {k.strip(): v for k, v in geocode_results.items()}
    # Convert the results into a DataFrame
    geocode_df = pd.DataFrame.from_dict(geocode_results, orient='index')

    geocode_df.reset_index(inplace=True)
    geocode_df.rename(columns={'index': 'Location'}, inplace=True)
    # Apply trim operation to 'Location' column in geocode_df
    geocode_df['Location'] = geocode_df['Location'].str.replace(" ", "")
    # Set 'Location' as index in geocode_df after trim operation
    geocode_df = geocode_df.set_index('Location')

    # Update the mapping for 'latitud' in weather_df
    weather_df['latitude'] = weather_df['Location'].map(geocode_df['latitude'])

    weather_df['longitude'] = weather_df['Location'].map(geocode_df['longitude'])
    return geocode_df,weather_df

def geocode_place(place):
    try:
        location = geolocator.geocode(place + ", Australia")
        time.sleep(1)  # Delay to avoid overusing the API
        return location.latitude, location.longitude

    except (AttributeError, GeocoderTimedOut):
        print("geolocation not ofound for:%s",place)
        return None, None