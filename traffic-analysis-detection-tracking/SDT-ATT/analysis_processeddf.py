import pandas as pd


df= pd.read_csv(r'C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\processed_combined_tracking_data.csv')

print(df.head())

print("Total number of frames: ", df['frame_number'].nunique())

