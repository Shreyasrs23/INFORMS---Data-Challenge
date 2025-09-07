import os
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.statespace.sarimax import SARIMAX

SPLIT_DIR = "./../Data Given for Challenge/data/"

file_path = os.path.join(SPLIT_DIR, "train.nc")

try:
    ds = xr.open_dataset(file_path)

    counties = list(ds.location.values)
    timestamps = pd.to_datetime(ds.timestamp.values)

    df=pd.DataFrame()

    for county_code in counties:
        for timestamp in timestamps:
            weather_profile = ds['weather'].sel(
                location=county_code, 
                timestamp=timestamp
            )

            out_value = ds['out'].sel(
                location=county_code, 
                timestamp=timestamp
            ).item()
    
            # Get the data values and feature names
            data_values = weather_profile.values
            feature_names = weather_profile.feature.values

            dummy_df = pd.DataFrame([data_values], columns=feature_names)

            dummy_df['out'] = out_value
            dummy_df['Location'] = county_code
            dummy_df['Timestamp'] = timestamp

            dummy_df = dummy_df[['Location', 'Timestamp', 'out'] + [col for col in dummy_df.columns if col not in ['Location', 'Timestamp', 'out']]]

            df = pd.concat([df, dummy_df], ignore_index=True)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")

finally:
    # Always close the dataset to free up resources
    if 'ds' in locals():
        ds.close()

df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)