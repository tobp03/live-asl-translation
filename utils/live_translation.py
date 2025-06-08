#Import python modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI
import cv2
import os
import mediapipe as mp
import json
import threading
from dotenv import load_dotenv
from utils import preprocessing_split as preprocessing

## DEFINE FUNCTIONS
def extract_landmarks(results):
    """
    Extracts pose (filtered), left/right hands, and selected face landmarks from MediaPipe results,
    returning only the x and y coordinates.
    
    Parameters:
        results (mp_holistic.HolisticResults): The MediaPipe Holistic results object.
        important_face_landmarks (list): Indices of important face landmarks to extract.
    
    Returns:
        np.ndarray: Combined landmark array with shape (N, 2), where N is the total number of selected keypoints.
    """
    important_face_landmarks = [
        33, 133, 159, 145, 153, 144,          # Right eye
        362, 263, 386, 374, 380, 373,         # Left eye
        70, 63, 105, 66, 107,                 # Right eyebrow
        295, 282, 320, 285, 318,              # Left eyebrow
        1, 168, 197, 4,                       # Nose bridge and tip
        78, 308, 13, 14, 81, 311              # Mouth and lips
        ]
    
    # Indices to remove from pose landmarks
    indices_to_remove1 = np.arange(23, 33)
    indices_to_remove2 = np.arange(0, 11)
    pose_indices_to_remove = set(indices_to_remove1.tolist() + indices_to_remove2.tolist())
    pose_indices_to_keep = [i for i in range(33) if i not in pose_indices_to_remove]

    # Pose landmarks (x, y only)
    if results.pose_landmarks:
        pose = [(lm.x, lm.y) 
                for i, lm in enumerate(results.pose_landmarks.landmark) 
                if i in pose_indices_to_keep]
    else:
        pose = [(0, 0)] * len(pose_indices_to_keep)

    # Face landmarks (x, y only)
    if results.face_landmarks:
        face_all = results.face_landmarks.landmark
        face = [(face_all[i].x, face_all[i].y) for i in important_face_landmarks]
    else:
        face = [(0, 0)] * len(important_face_landmarks)

    # Left hand (x, y only)
    if results.left_hand_landmarks:
        left_hand = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
    else:
        left_hand = [(0, 0)] * 21

    # Right hand (x, y only)
    if results.right_hand_landmarks:
        right_hand = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
    else:
        right_hand = [(0, 0)] * 21

    # Combine all
    combined = np.array(pose + left_hand + right_hand + face)
    return combined


def preprocess(X):
    '''
    Applies normalization to various parts of the input landmark.
    '''
    neck = abs(X[0] + X[ 1])/2
    X[ :]=preprocessing.AnchorNorm(X[:],preprocessing.distance(X[0],X[ 1]),neck)
    #face norm
    X[ 54:]=preprocessing.AnchorNorm(X[ 54:],preprocessing.distance(X[ 0],X[ 1]),X[ 79])

    # #Left arm normalization, left shoulder as reference, scaled by left arm length
    left_arm = [2,4,6,8,10]
    X[ left_arm]=preprocessing.AnchorNorm(X[ left_arm],preprocessing.distance(X[ 0],X[ 2]),X[ 0])
    
    # #Right arm normalization, right shoulder as reference, scaled by left arm length
    right_arm =[3,5,7,9,11]
    X[ right_arm]=preprocessing.AnchorNorm(X[ right_arm],preprocessing.distance(X[ 1],X[ 3]),X[ 1])

    #Hand bounding box normalization
    X[ 12:33] = preprocessing.hand_normalize(X[ 12:33])
    X[ 33:54] = preprocessing.hand_normalize(X[ 33:54])
    return X


def get_client(env_path='.env'):
    load_dotenv(dotenv_path=env_path)
    client = OpenAI(api_key=os.getenv("API_KEY"))
    return client


# === Main function for live feed translation ====
def start_live_feed(
        model_path,
        encoder_path,
        client,
        threshold=1.2,
        webcam=0,
        complexity_setting=0
        ):

    ##################################################
    # == Define local functions for feed translation #
    ##################################################
    def movement_score(landmarks_frames,hand_weight=3):
        '''
        Calculate the average movement between consecutive frames.

        Parameters:
        landmark_frames (numpy array):  Sequence of landmarks with frames with shape (T,N,2), 
                                        where T is number of frames, N is the number of landmarks,
                                        and 2 corresponds to x and coordinates.

        hand_weight (integer)       :   An integer scaling factor for hands landmark (>0). A higher
                                        value increases the contribution of hand movement towards the final score

        Output:
        float: Average movement sccore across frames
        '''
        hand_indices = list(range(12, 33)) + list(range(33, 54))  # adjust based on your hand landmark indices
        diffs = []
        for i in range(len(landmarks_frames) - 1):
            diff = np.linalg.norm(landmarks_frames[i+1] - landmarks_frames[i], axis=1)  # shape (num_landmarks,)

            # Apply 2x weight to hand landmarks
            weighted_diff = diff.copy()
            weighted_diff[hand_indices] *= hand_weight

            total_diff = np.sum(weighted_diff)
            diffs.append(total_diff)

        avg_movement = np.mean(diffs)
        return avg_movement
    

    
    
    def update_ema(new_score,alpha=0.3):
        '''
        Updates the Exponential Moving Average score with a new value.

        Parameters:
        new_score (float)   : The latest score to incorporate into the EMA.
        alpha (float)       : The smoothing factor, where hqigher value gives more weight to the new score.

        Returns:
        float: updated EMA score
        '''
        nonlocal ema_score
        if ema_score is None:
            ema_score = new_score
        else:
            ema_score = alpha * new_score + (1 - alpha) * ema_score
        return ema_score

    def draw_last_predictions(frame, predictions, x=10, y=70, line_height=30, font_scale=0.7, thickness=2):
        '''
        Draws a semi-transparent rectangle overlay and list of prediction texts on the video frame.

        This function calculates the required width and height of a semi-transparent black rectangle
        to fit all the prediction texts, draws the rectangle, and overlays each prediction string.
        '''

        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 0
        height = line_height * len(predictions) + 20
        
        # Find max width for all texts
        for word in predictions:
            (w, _), _ = cv2.getTextSize(word, font, font_scale, thickness)
            if w > width:
                width = w
        # Draw semi-transparent rectangle as background
        cv2.rectangle(overlay, (x-10, y-20), (x + width + 10, y + height - 10), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw each prediction text
        for i, word in enumerate(predictions):
            text_y = y + i * line_height
            cv2.putText(frame, word, (x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def interpret_gloss_sequence(gloss_list,client,max_tokens=30):
        '''
        Interpret a list of glosses into a well-written english sentence with LLM (OpenAI GPT)

        Parameters:
        gloss_list (list of str)    : A list of glosses to be interpreted
        max_tokens (int)            : The maximum number of tokens to be generated
        '''
        nonlocal current_interpretation, interpreting

        gloss_string = " + ".join(gloss_list)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "developer", "content": 
                "You are a sign language interpreter. Convert input into well-formed English. Preserve whether the input is a question or a statement. If it ends with you, it is likely a question. Do not add your own commentary or questions."
    },
                {"role": "user", "content": gloss_string}
            ],
            stream=True,
            max_tokens=max_tokens
        )

        interpretation = ""
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                interpretation += content
                current_interpretation = interpretation  # live update
        interpreting = False

    def show_interpreted_text(frame, text):
        '''
        Displays interpreted text (LLM output) as an overlay at the bottom of the video frame
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 3
        color = (0, 255, 255)  # Yellow text
        bg_color = (50, 50, 50)  # Dark background

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        text_x = (frame.shape[1] - text_w) // 2
        text_y = frame.shape[0] - 40
        # Draw background rectangle
        cv2.rectangle(frame, (text_x - 20, text_y - text_h - 20),
                    (text_x + text_w + 20, text_y + 20),
                    bg_color, -1)

        # Draw interpreted text
        cv2.putText(frame, text, (text_x, text_y),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    def process_frame(frame,landmark_stored):
        '''
        Process a video frame to extract, preprocess, and store landmarks.

        Steps:
        - Flips the frame horizontally
        - Converts the image to RGB
        - Applied Mediapipe Holistic model to detect landmarks
        - Draws detected hand landmarks on the frame
        - Extracts and preprocess landmarks
        -Store the most recent 30 frames of landmarks
        '''
        frame = cv2.flip(frame,1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        numpy_data = extract_landmarks(results)
        data_processed = preprocess(numpy_data)
        landmark_stored.append(data_processed)
        landmark_stored = landmark_stored[-30:] 
        return landmark_stored,frame
    
    def handle_capture_logic(landmark_stored,
                             last_predictions,
                             encoder,
                             model,
                             counter,
                             threshold,
                             max_pred=10):
        
        status_text="Not Capturing"
        status_color=(0,0,255)
        if len(landmark_stored) > 5:
            score = movement_score(np.array(landmark_stored[-5:]))
            smoothed_score = update_ema(score)
            print(score, smoothed_score, counter)
            if smoothed_score > threshold:
                counter+=1
                status_text="Capturing"
                status_color = (0, 255, 0)  # Green in BGR     
            else:
                if counter>0:
                    landmark_array=np.array(landmark_stored)
                    padded_array=preprocessing.pad_video(landmark_array)
                    reshape_array=padded_array.reshape(padded_array.shape[0],-1)
                    model_input=  np.expand_dims(reshape_array, axis=0)
                    prediction=model.predict(model_input)
                    predicted_index=np.argmax(prediction)
                    prediction_result = encoder[str(predicted_index)]
                    counter=0
                    last_predictions.append(prediction_result)
                    if len(last_predictions)>max_pred:
                        last_predictions.pop(0)

        return last_predictions,status_text,status_color,counter

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

   
    last_predictions = []
    cap = cv2.VideoCapture(webcam)  # 0 is the default camera index
    landmark_stored = []
    counter=0
    ema_score = None
    current_interpretation = ""
    interpreting = False

    with open(encoder_path, 'r') as f:
        gloss_dict = json.load(f)
        encoder = {k: v for k, v in gloss_dict.items()}
    
    model = load_model(model_path)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexity_setting,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.9
) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue
            
            landmark_stored,frame=process_frame(frame,landmark_stored)

            last_predictions,status_text,status_color,counter=handle_capture_logic(landmark_stored,
                                                                        last_predictions,
                                                                        encoder,
                                                                        model,
                                                                        counter,
                                                                        threshold)
            # Draw a filled circle indicator top-left corner
            cv2.circle(frame, (40, 40), 20, status_color, -1)
            # Put status text next to the circle
            cv2.putText(frame, status_text, (70, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, status_color, 2, cv2.LINE_AA)

            # Draw last 10 predictions at top-left
            if last_predictions:
                draw_last_predictions(frame, last_predictions, x=10, y=70)
            # Draw LLM interpretation
            if current_interpretation:
                show_interpreted_text(frame, current_interpretation)    
            #If LLM is running, draw text
            if interpreting:
                cv2.putText(frame, "Interpreting...", 
                            (frame.shape[1] - 220, 30),  # Top-right corner, adjust X for alignment
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6,  # Font scale (small)
                            (255, 0, 0),  # Blue in BGR
                            1,  # Thin text
                            cv2.LINE_AA)
            cv2.imshow('MediaPipe Webcam Feed', frame)

            #Check for key inputs
            #Remove last prediction with "a" key
            if cv2.waitKey(1) & 0xFF == ord('a') and len(last_predictions)>0:
                last_predictions.pop()
            #Run LLM to interpret current predictions with "c"
            elif cv2.waitKey(1) & 0xFF == ord('c') and not interpreting:
                gloss_input = last_predictions.copy()
                last_predictions.clear()
                interpreting = True
                threading.Thread(target=interpret_gloss_sequence, args=(gloss_input,client), daemon=True).start()

            #Stop the live feed
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


# client = get_client('/home/toby/Documents/Codes/Python/Sign Transation/.env')

# start_live_feed(model_path='/home/toby/Documents/Codes/Python/Sign Transation/best_model200.keras',
#                 encoder_path='/home/toby/Documents/Codes/Python/Sign Transation/index_to_gloss_200.json',
#                 client=client)
