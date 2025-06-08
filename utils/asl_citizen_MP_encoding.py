import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import mediapipe as mp

#Define Python Functions

# Functions receives video path as an input, and numpy array (frames, 107 landmarks, 3) as an output 
def extract_landmarks_from_video(video_path):
    important_face_landmarks = [
        33, 133, 159, 145, 153, 144,          # Right eye
        362, 263, 386, 374, 380, 373,         # Left eye
        70, 63, 105, 66, 107,                 # Right eyebrow
        295, 282, 320, 285, 318,              # Left eyebrow
        1, 168, 197, 4,                       # Nose bridge and tip
        78, 308, 13, 14, 81, 311              # Mouth and lips
        ]
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    all_landmarks = [] #Temporary list to store landmarks extracted from each frame of the video.
    with mp_holistic.Holistic(static_image_mode=False,  #Mediapipe configuration parameters
                              model_complexity=1, 
                              smooth_landmarks=True,
                              enable_segmentation=False,
                              min_detection_confidence=0.7,
                              min_tracking_confidence=0.7) as holistic:
        while cap.isOpened(): 
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            if results.pose_landmarks:
                pose = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark] #Selected all the pose landmarks
            else:
                pose = [(0, 0, 0)] * 33

            if results.face_landmarks:
                face_all = results.face_landmarks.landmark 
                face = [(face_all[i].x, face_all[i].y, face_all[i].z) for i in important_face_landmarks] #Selected subset of important face landmarks
            else:
                face = [(0, 0, 0)] * len(important_face_landmarks)

            if results.left_hand_landmarks:
                left_hand = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark] #Selected all hand landmarks
            else:
                left_hand = [(0, 0, 0)] * 21

            if results.right_hand_landmarks:
                right_hand = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]
            else:
                right_hand = [(0, 0, 0)] * 21

            combined = np.array(pose + left_hand + right_hand + face) #Array order: Index 0-32 pose, 33-53 left hand, 54-74 right hand, 75-end face landmarks
            all_landmarks.append(combined)
    cap.release()
    cv2.destroyAllWindows()
    all_landmarks = np.array(all_landmarks)
    return all_landmarks

# Receives an input of row from splits (train.csv, test.csv, and val.csv). 
# Output: landmarks processed from video (frames, 107 landmarks, 3), gloss, video file location, and participant id.
def process_video(row,video_dir): 
    participant_id = str(row['Participant ID'])
    video_file = str(row['Video file'])
    gloss = str(row['Gloss'])
    asl_lex_code = str(row['ASL-LEX Code'])

    video_path = os.path.join(video_dir, video_file)
    if not os.path.isfile(video_path):
        print(f'Warning: Video file {video_path} not found, skipping.')
        return None
    landmarks = extract_landmarks_from_video(video_path)
    return {
        'landmarks': landmarks,
        'gloss': gloss,
        'video_file': video_file,
        'participant_id': participant_id}

def process_asl_dataset(video_dir,processed_save_dir):
    """
    Processes all ASL Citizen video files into .npz landmark arrays.
    Args:
        video_dir (str): Path to the ASL Citizen video directory.
        processed_save_dir (str): Path to save the processed .npz files.
    """
    splits = ['train', 'test', 'val']
    asl_citizen_dir = os.path.dirname(video_dir)
    
    for split in splits:
        split_dir = os.path.join(processed_save_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        csv_path = os.path.join(asl_citizen_dir, 'splits', f'{split}.csv')
        df = pd.read_csv(csv_path)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {split}'):
            npz_filename = row['Video file'].replace('.mp4', '.npz')
            npz_path = os.path.join(split_dir, npz_filename)

            if os.path.exists(npz_path):
                print(f'Skipping {npz_filename}, already processed.')
                continue

            processed = process_video(row, video_dir)
            if processed is None:
                continue

            np.savez_compressed(
                npz_path,
                landmarks=np.array(processed['landmarks'][:, :, :2], dtype=np.float32),
                gloss=np.array(processed['gloss'])
            )
            del processed
