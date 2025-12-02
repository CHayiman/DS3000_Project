import pandas as pd
import requests
from io import StringIO
import numpy as np

# ==========================================
# PART 1: ROBUST MULTI-STATION WEATHER DOWNLOAD
# ==========================================
def download_station(station_id, start_year, end_year, station_name):
    print(f"Downloading {station_name} (ID: {station_id}) for {start_year}-{end_year}...")
    frames = []
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
    
    for year in range(start_year, end_year + 1):
        # FIX: Loop through all 12 months, not just Month=1
        for month in range(1, 13):
            # timeframe=1 is Hourly data (requires month-by-month download)
            url = f"{base_url}?format=csv&stationID={station_id}&Year={year}&Month={month}&timeframe=1&submit=Download+Data"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    content = response.content.decode('utf-8')
                    if "Station not found" not in content:
                        df = pd.read_csv(StringIO(content))
                        frames.append(df)
            except Exception as e:
                print(f"  - Error downloading {year}-{month} for {station_name}: {e}")
        
        # Print progress after each year so you know it's not frozen
        print(f"  - Completed {year}")
            
    if not frames:
        return pd.DataFrame() # Return empty if failed

    df_combined = pd.concat(frames)
    
    # Rename columns to be Python-friendly
    df_combined.rename(columns={
        'Date/Time (LST)': 'weather_datetime',
        'Temp (°C)': 'temperature',
        'Precip. Amount (mm)': 'precipitation',
        'Visibility (km)': 'visibility',
        'Weather': 'weather_desc'
    }, inplace=True)
    
    # Standardize time and index
    df_combined['weather_datetime'] = pd.to_datetime(df_combined['weather_datetime'])
    df_combined.set_index('weather_datetime', inplace=True)
    
    # Return only useful columns to save memory
    return df_combined[['temperature', 'precipitation', 'visibility', 'weather_desc']]

def get_combined_weather_data(start_year, end_year):
    # 1. Download Primary Station (Pearson Airport - 51459)
    df_primary = download_station(51459, start_year, end_year, "Pearson Airport")
    
    # 2. Download Backup Station (City Centre / Billy Bishop - 48549)
    df_backup = download_station(48549, start_year, end_year, "City Centre Airport")
    
    # 3. Patching: Fill gaps in Primary using Backup
    print("\nPatching missing data using backup station...")
    # combine_first() prioritizes df_primary. If a value is NaN, it grabs it from df_backup.
    weather_data = df_primary.combine_first(df_backup)
    
    return weather_data

# ==========================================
# PART 2: LOAD & PREP COLLISION DATA
# ==========================================
try:
    df_collisions = pd.read_csv('Traffic_Collisions.csv')
    print("Collision data loaded successfully.")
except FileNotFoundError:
    print("ERROR: Please download 'Traffic_Collisions.csv' and place it in this folder.")
    # Dummy data for testing if file missing
    df_collisions = pd.DataFrame({
        'Event_Unique_ID': ['GO-2022001'],
        'OCC_DATE': ['2022-01-01'],
        'OCC_HOUR': [14]
    })

# 1. Clean the Date
df_collisions['date_clean'] = pd.to_datetime(df_collisions['OCC_DATE']).dt.date

# 2. Clean the Hour
df_collisions['hour_clean'] = pd.to_numeric(df_collisions['OCC_HOUR'], errors='coerce').fillna(0).astype(int)

# 3. Create the Merge Key
df_collisions['merge_key'] = df_collisions.apply(
    lambda row: pd.Timestamp(row['date_clean']) + pd.Timedelta(hours=row['hour_clean']), 
    axis=1
)

# ==========================================
# PART 3: CLEAN, FILL & MERGE
# ==========================================

# 1. Get the "Patched" Weather Data (Pearson + City Centre)
# Note: Adjust years to match your actual dataset range
weather_data = get_combined_weather_data(2014, 2025)

# 2. Smart Time Filling (Resample & Forward Fill)
# Ensure every hour exists
weather_data = weather_data.asfreq('H')

print("Applying 4-hour limit forward fill...")
cols_to_fix = ['temperature', 'precipitation', 'visibility', 'weather_desc']

# Fill small gaps (up to 4 hours) using the previous hour's data
weather_data[cols_to_fix] = weather_data[cols_to_fix].ffill(limit=4)

# B) THE FIX: Force remaining missing Precipitation to 0.0
# If it's still missing after the patch, assume it wasn't raining.
weather_data['precipitation'] = weather_data['precipitation'].fillna(0.0)

# C) THE FIX: Backward Fill for the start of the file
# If the first few rows are missing temp/visibility, grab the first valid future value
weather_data[cols_to_fix] = weather_data[cols_to_fix].bfill()

# 3. Prepare for Merge
weather_data.reset_index(inplace=True)

# 4. Merge with Collisions
print(f"Merging {len(df_collisions)} collisions with weather data...")
df_final = pd.merge(
    df_collisions, 
    weather_data, 
    left_on='merge_key', 
    right_on='weather_datetime', 
    how='left'
)

# 6. Save
df_final.to_csv("Traffic_Collisions_With_Weather_Patched.csv", index=False)
print("\nSuccess! Saved 'Traffic_Collisions_With_Weather_Patched.csv'")

# ==========================================
# PART 4: STATISTICS & QUALITY CHECK
# ==========================================
print("\n=== MISSING DATA ANALYSIS ===")
total_rows = len(df_final)
missing_weather = df_final['temperature'].isnull().sum()

print(f"Total Collisions: {total_rows}")
print(f"Collisions with missing weather info: {missing_weather}")
print(f"Coverage: {((total_rows - missing_weather)/total_rows)*100:.2f}%")

print("\nSample Data:")
print(df_final[['merge_key', 'temperature', 'is_rain', 'is_snow']].head())