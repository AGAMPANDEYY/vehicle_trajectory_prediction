import pandas as pd 
import numpy as np 


df= pd.read_csv(r'C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\combined_tracking_data.csv')

#frame_number,timestamp,tracker_id,x,y,x1,y1,x2,y2,prediction_type,sequence_index

df_processed = df.copy()

df_processed= df_processed[df_processed['prediction_type'] == 'actual']

df_processed=df_processed.drop(columns=['prediction_type', 'sequence_index'])

df_processed['frame_number'] = df_processed['frame_number'].astype(int)
df_processed['timestamp']= df_processed['timestamp']
df_processed['tracker_id'] = df_processed['tracker_id'].astype(int)
df_processed['x'] = df_processed['x'].astype(float)
df_processed['y'] = df_processed['y'].astype(float)
df_processed['x1'] = df_processed['x1'].astype(float)
df_processed['y1'] = df_processed['y1'].astype(float)
df_processed['x2'] = df_processed['x2'].astype(float)
df_processed['y2'] = df_processed['y2'].astype(float)

df_processed.to_csv(r'C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\processed_combined_tracking_data.csv', index=False)