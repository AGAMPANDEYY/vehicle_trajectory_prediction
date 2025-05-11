import pandas as pd


df= pd.read_csv(r'D:\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\processed_combined_tracking_data (2).csv')

print(df.head())

print("Total number of frames: ", df['frame_number'].nunique())

