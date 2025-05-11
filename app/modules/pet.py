import pandas as pd
import numpy as np
import cv2
import json
from shapely.geometry import Polygon, Point

class PETPipeline:
    def __init__(self, tracking_path, zone_path, video_path):
        self.tracking_path = tracking_path
        self.zone_path = zone_path
        self.video_path = video_path
        self.df = None
        self.zone_df = None
        self.conflict_zones = []
        self.conflict_log = []
        self.pet_df = None
        self.width = None
        self.height = None
        self.fps = None

    def load_data(self):
        self.df = pd.read_csv(self.tracking_path)
        self.df.rename(columns={
            'timestamp': 'Time',
            'frame_number': 'Frame_ID',
            'vehicle_id': 'Vehicle_ID',
            'x': 'X_central',
            'y': 'Y_central',
            'x1': 'X1_boundary',
            'y1': 'Y1_boundary',
            'x2': 'X3_boundary',
            'y2': 'Y3_boundary'
        }, inplace=True)

        self.zone_df = pd.read_csv(self.zone_path)

    def build_conflict_zones(self):
        for _, row in self.zone_df.iterrows():
            shape = json.loads(row['region_shape_attributes'].replace("'", '"'))
            coords = list(zip(shape['all_points_x'], shape['all_points_y']))
            polygon = Polygon(coords)
            self.conflict_zones.append({
                'name': row['zone_id'],
                'polygon': polygon
            })

    def map_conflict_zones(self):
        def get_zone_id(row):
            pt = Point(row['X_central'], row['Y_central'])
            for zone in self.conflict_zones:
                if zone['polygon'].contains(pt):
                    return zone['name']
            return None

        self.df['Conflict_Zone_ID'] = self.df.apply(get_zone_id, axis=1)
        self.df['In_Conflict_Zone'] = self.df['Conflict_Zone_ID'].notnull()
        self.df['Conflict_Zone_ID'] = self.df['Conflict_Zone_ID'].fillna(-1)

    def detect_conflicts(self):
        entries = self.df[self.df['In_Conflict_Zone']]
        for vid in entries['Vehicle_ID'].unique():
            track = entries[entries['Vehicle_ID'] == vid].sort_values('Time')
            for zid in track['Conflict_Zone_ID'].unique():
                zone_track = track[track['Conflict_Zone_ID'] == zid]
                in_zone = False
                entry_time = None

                for i, (_, row) in enumerate(zone_track.iterrows()):
                    if not in_zone:
                        entry_time = row['Time']
                        in_zone = True
                    if in_zone and (i == len(zone_track) - 1 or zone_track.iloc[i+1]['Conflict_Zone_ID'] != zid):
                        exit_time = row['Time']
                        self.conflict_log.append({
                            'Vehicle_ID': vid,
                            'Conflict_Zone_ID': zid,
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'duration': exit_time - entry_time
                        })
                        in_zone = False

    def compute_pet(self):
        pet_data = []
        df_log = pd.DataFrame(self.conflict_log)
        if df_log.empty:
            self.pet_df = pd.DataFrame()
            return

        df_log = df_log.sort_values(['Conflict_Zone_ID', 'exit_time'])
        for zid in df_log['Conflict_Zone_ID'].unique():
            zone_df = df_log[df_log['Conflict_Zone_ID'] == zid].reset_index(drop=True)
            for i in range(1, len(zone_df)):
                prev = zone_df.iloc[i - 1]
                curr = zone_df.iloc[i]
                pet = curr['entry_time'] - prev['exit_time']
                pet_data.append({
                    'zone_id': zid,
                    'leading_vehicle': prev['Vehicle_ID'],
                    'following_vehicle': curr['Vehicle_ID'],
                    'pet': pet
                })

        self.pet_df = pd.DataFrame(pet_data)
        self.pet_df['risk_category'] = self.pet_df['pet'].apply(self.categorize_pet)

    def categorize_pet(self, pet):
        if pet <= 0:
            return 'Near-Miss'
        elif pet <= 1.5:
            return 'High Risk'
        elif pet <= 3.0:
            return 'Moderate Risk'
        elif pet <= 5.0:
            return 'Low Risk'
        else:
            return 'Safe'

    def prepare_video_metadata(self):
        cap = cv2.VideoCapture(self.video_path)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    def generate_heatmap(self):
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        weights = {
            'Near-Miss': 1.0**2,
            'High Risk': 0.8**2,
            'Moderate Risk': 0.6**2,
            'Low Risk': 0.4**2,
            'Safe': 0.2**2
        }

        for _, row in self.pet_df.iterrows():
            lead = self.df[self.df['Vehicle_ID'] == row['leading_vehicle']]
            follow = self.df[self.df['Vehicle_ID'] == row['following_vehicle']]
            common = np.intersect1d(lead['Frame_ID'], follow['Frame_ID'])

            for fid in common:
                l = lead[lead['Frame_ID'] == fid]
                f = follow[follow['Frame_ID'] == fid]
                if l.empty or f.empty:
                    continue

                for box in [l, f]:
                    x = int(box['X_central'].values[0])
                    y = int(box['Y_central'].values[0])
                    cv2.circle(heatmap, (x, y), 5, weights[row['risk_category']], -1)

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=10, sigmaY=10)
        heatmap = np.clip(heatmap, 0.05, 1.0)
        heatmap = (heatmap - 0.05) / (1.0 - 0.05)
        return heatmap

    def run(self):
        self.load_data()
        self.build_conflict_zones()
        self.map_conflict_zones()
        self.detect_conflicts()
        self.compute_pet()
        self.prepare_video_metadata()      
    def generate_heatmap_video(self, heatmap):
        cap = cv2.VideoCapture(self.video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output_with_heatmap.mp4", fourcc, fps, (width, height))

        normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET)

        # === Step 4: Overlay on video frames ===
        alpha = 0.6  # heatmap transparency

        def draw_heatmap_legend(frame, x=20, y=5, width=20, height=20):
            legend_colors = {
            'Near-Miss': (0, 0, 255),         # Red
            'High Risk': (0, 69, 255),        # Orange-Red
            'Moderate Risk': (0, 255, 255),   # Yellow
            'Low Risk': (255, 255, 0),        # Cyan
            'Safe': (255, 0, 0)               # Blue
            }

            gap = height + 5  # vertical spacing
            start_y = y

            for label, color in legend_colors.items():
                end_y = start_y + height
                cv2.rectangle(frame, (x, start_y), (x + width, end_y), color, -1)
                cv2.putText(frame, label, (x + width + 10, start_y + height // 2 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                start_y += gap

            return frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            if colored_heatmap.shape[:2] != frame.shape[:2]:
                colored_heatmap = cv2.resize(colored_heatmap, (frame.shape[1], frame.shape[0]))
            
            overlay = cv2.addWeighted(colored_heatmap, alpha, frame, 1 - alpha, 0)
            overlay = draw_heatmap_legend(overlay)  # Add legend after blending
            
            out.write(overlay)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
