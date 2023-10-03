import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import math 
import traceback

import warnings
# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

analyzed_results = None 
xaxis = np.array([[1, 0, 0]])
yaxis = np.array([[0, 1, 0]])
zaxis = np.array([[0, 0, 1]])
uaxis = np.array([[1, 1, 1]])
def count_seconds(frame_number, frame_rate):
    return frame_number / frame_rate
def get_time_from_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}"
def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
def calculate_distance(pointX, pointY) -> float:
    '''
    Calculate a distance between 2 points
    '''

    x1, y1 = pointX
    x2, y2 = pointY

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)



    '''
    '''
def point_to_array(point) :
     return np.array([point.x, point.y, point.z, point.visibility])
def calculate_degree(pointX, pointY) : 
    return np.degrees(np.arccos(pointY @ pointX.T))
def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg
def check_perpendicular_limb(pointX, pointY, allowed_error = 15) :
     limb_xaxis_angle = np.degrees(np.arccos(pointX @ pointY.T))
     if abs(limb_xaxis_angle - 90) > allowed_error:
        return False
     else:
        return True
def calculate_distance3D(pointX, pointY) :
     return np.linalg.norm(pointY - pointX)
def extract_important_keypoints(results, IMPORTANT_LMS) -> list:
    '''
    Extract important keypoints from mediapipe pose detection
    '''
    landmarks = results.pose_landmarks.landmark

    data = []
    for lm in IMPORTANT_LMS:
        keypoint = landmarks[mp_pose.PoseLandmark[lm].value]
        data.append([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])
    
    return np.array(data).flatten().tolist()
def save_frame_as_image(frame, message: str = None):
    '''
    Save a frame as image to display the error
    '''
    now = datetime.datetime.now()

    if message:
        cv2.putText(frame, message, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
# Load model for counter
def squat(video_path) : 
    IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
    ]
    error_count = {
        "Foot_Tight": 0,
        "Foot_Wide": 0,
        "Knee_Tight": 0,
        "Knee_Wide": 0
    }

    headers = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
        analyzed_results = {
            "foot_placement": -1,
            "knee_placement": -1,
        }   
        landmarks = results.pose_landmarks.landmark

        # * Visibility check of important landmarks for foot placement analysis
        left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
        right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

        # If visibility of any keypoints is low cancel the analysis
        if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
            return analyzed_results
        
        # * Calculate shoulder width
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)

        # * Calculate 2-foot width
        left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        foot_width = calculate_distance(left_foot_index, right_foot_index)

        # * Calculate foot and shoulder ratio
        foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

        # * Analyze FOOT PLACEMENT
        min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
        if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 0
        elif foot_shoulder_ratio < min_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 1
        elif foot_shoulder_ratio > max_ratio_foot_shoulder:
            analyzed_results["foot_placement"] = 2
        
        # * Visibility check of important landmarks for knee placement analysis
        left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
        right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

        # If visibility of any keypoints is low cancel the analysis
        if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
            return analyzed_results

        # * Calculate 2 knee width
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        knee_width = calculate_distance(left_knee, right_knee)

        # * Calculate foot and shoulder ratio
        knee_foot_ratio = round(knee_width / foot_width, 1)

        # * Analyze KNEE placement
        up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("up")
        middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("middle")
        down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("down")

        if stage == "up":
            if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 0
            elif knee_foot_ratio < up_min_ratio_knee_foot:
                analyzed_results["knee_placement"] = 1
            elif knee_foot_ratio > up_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 2
        elif stage == "middle":
            if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 0
            elif knee_foot_ratio < middle_min_ratio_knee_foot:
                analyzed_results["knee_placement"] = 1
            elif knee_foot_ratio > middle_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 2
        elif stage == "down":
            if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 0
            elif knee_foot_ratio < down_min_ratio_knee_foot:
                analyzed_results["knee_placement"] = 1
            elif knee_foot_ratio > down_max_ratio_knee_foot:
                analyzed_results["knee_placement"] = 2
        
        return analyzed_results
    with open("./model/squat_model.pkl", "rb") as f:
        count_model = pickle.load(f)
    cap = cv2.VideoCapture(video_path)
    output_path = 'static\\videos\\output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# Counter vars
    counter = 0
    current_stage = ""
    PREDICTION_PROB_THRESHOLD = 0.7

    # Error vars
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.7, 1.0],
        "down": [0.7, 1.1],
    }
    last_error_time = "00:00:00"


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seconds = count_seconds(frame_number, frame_rate)
            time = get_time_from_seconds(seconds)
            # Reduce size of a frame
            image = rescale_frame(image, 100)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if not results.pose_landmarks:
                # out.write(image)
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Make detection
            try:
                # * Model prediction for SQUAT counter
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row])
                # Make prediction and its probability
                predicted_class = "up" if count_model.predict(X)[0] == 1 else "down"
                prediction_probabilities = count_model.predict_proba(X)[0]
                prediction_probability = round(prediction_probabilities[prediction_probabilities.argmax()], 2)
                # Evaluate model prediction
                if predicted_class == "down" and prediction_probability >= PREDICTION_PROB_THRESHOLD:
                    current_stage = "down"
                elif current_stage == "down" and predicted_class == "up" and prediction_probability >= PREDICTION_PROB_THRESHOLD: 
                    current_stage = "up"
                    counter += 1
                # Analyze squat pose
                analyzed_results = analyze_foot_knee_placement(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)
                foot_placement_evaluation = analyzed_results["foot_placement"]
                knee_placement_evaluation = analyzed_results["knee_placement"]
                # * Evaluate FOOT PLACEMENT error
                if foot_placement_evaluation == -1:
                    foot_placement = "UNK"
                elif foot_placement_evaluation == 0:
                    foot_placement = "Correct"
                elif foot_placement_evaluation == 1:
                    foot_placement = "Too tight"
                   
                elif foot_placement_evaluation == 2:
                    foot_placement = "Too wide"
                    
                
                # * Evaluate KNEE PLACEMENT error
                if knee_placement_evaluation == -1:
                    knee_placement = "UNK"
                elif knee_placement_evaluation == 0:
                    knee_placement = "Correct"
                elif knee_placement_evaluation == 1:
                    knee_placement = "Too tight"
                elif knee_placement_evaluation == 2:
                    knee_placement = "Too wide"
                    
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f'{str(counter)}, {predicted_class.split(" ")[0]}, {str(prediction_probability)}', (5, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Foot and Shoulder width ratio
                cv2.putText(image, "FOOT", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, foot_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display knee and Shoulder width ratio
                cv2.putText(image, "KNEE", (330, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, knee_placement, (325, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                if knee_placement_evaluation == 1:
                    if(time == last_error_time) :
                        continue
                    else:
                        last_error_time = time
                        error_count["Knee_Tight"] += 1
                        cv2.imwrite('./static/images/Knee_Tight_At_' + time +'.jpg', image)
                elif knee_placement_evaluation == 2:
                    if(time == last_error_time) :
                        continue
                    else:
                        last_error_time = time
                        error_count["Knee_Wide"] += 1
                        cv2.imwrite('./static/images/Knee_Wide_At_' + time +'.jpg', image)
                if foot_placement_evaluation == 1:
                    if(time == last_error_time) :
                        continue
                    else:
                        last_error_time = time
                        error_count["Foot_Tight"] += 1
                        cv2.imwrite('./static/images/Foot_Tight_At_' + time +'.jpg', image)
                elif foot_placement_evaluation == 2:
                    if(time == last_error_time) :
                        continue
                    else:
                        last_error_time = time
                        error_count["Foot_Wide"] += 1
                        cv2.imwrite('./static/images/Foot_Wide_At ' + time+'.jpg', image)
            except Exception as e:
                print(f"Error: {e}")
            out.write(image)
        out.release()

# Giảm data rate và tổng bitrate

    total_error_count = sum(error_count.values())
    return output_path, error_count, total_error_count
def deadlift(video_path) : 
    IMPORTANT_LMS = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]
    headers = ["label"] # Label column
    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    error_count = {
        "Torso_Curved": 0,
        "Foot_Wide": 0,
        "Foot_Tight": 0,
        "Grip_Wide": 0,
        "Grip_Tight": 0,
        "Knee_Tight": 0,
        "Knee_Wide": 0
    }
    def analyze_pose(results, visibility_threshold: int) : 
     analyzed_results = {
        "torso": -1,
        "stance":-1,
        "grip":-1,
        "depth":-1,
        "left_hand_straight":-1,
        "right_hand_straight":-1,
        "state": -1  
        }    
     landmarks = results.pose_landmarks.landmark
     
     left_shoulder = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
     right_shoulder = point_to_array(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
     left_hip = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
     right_hip = point_to_array( landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
     left_ankle = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
     right_ankle = point_to_array( landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
     left_wrist = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
     right_wrist = point_to_array( landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
     left_elbow = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])
     right_elbow = point_to_array( landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
     left_knee = point_to_array(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
     right_knee = point_to_array( landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
     if (left_shoulder[-1] < visibility_threshold or 
         right_shoulder[-1] < visibility_threshold or 
         left_hip[-1] < visibility_threshold or 
         right_hip[-1] < visibility_threshold or 
         left_ankle[-1] < visibility_threshold or
         right_ankle[-1] < visibility_threshold or
         left_wrist[-1] < visibility_threshold or
         right_wrist[-1] < visibility_threshold or
         left_elbow[-1] < visibility_threshold or
         right_elbow[-1] < visibility_threshold or
         left_knee[-1] < visibility_threshold or
         right_knee[-1] < visibility_threshold) :
        return analyzed_results
     #CHECK BODY STRAIGHT
     torso = np.array([(left_shoulder[:3] - left_hip[:3]) + (right_shoulder[:3] - right_hip[:3])])
     # torso = np.array(left_shoulder[:3] - left_hip[:3])
     # torso = np.array(right_shoulder[:3] - right_hip[:3])
     analyzed_results["torso"] = 0
     if not check_perpendicular_limb(pointX = torso, pointY = xaxis, allowed_error=5.):
          analyzed_results["torso"] = 1
     #CHECK STANCE
     hip_length = np.linalg.norm(left_hip[:3] - right_hip[:3])
     ankle_length = np.linalg.norm(left_ankle[:3] - right_ankle[:3])
     analyzed_results["stance"] = 0
     if ankle_length > hip_length*2.35 : 
          analyzed_results["stance"] = 1
     elif ankle_length < hip_length:
          analyzed_results["stance"] = 2
     #CHECK GRIP
     shoulder_width = np.linalg.norm(left_shoulder[:3] - right_shoulder[:3])
     grip_width = np.linalg.norm(left_wrist[:3] - right_wrist[:3])
     analyzed_results["grip"] = 0
     if grip_width > shoulder_width*1.85:
          analyzed_results["grip"] = 1
     elif grip_width < shoulder_width*1.1:
          analyzed_results["grip"] = 2
     #CHECK DEPTH
     hips = np.array([left_hip, right_hip])
     knees = np.array([left_knee, right_knee])
     h = ( calculate_distance3D(left_ankle[:3], left_knee[:3]) +  calculate_distance3D(right_ankle[:3], right_knee[:3]) ) /2
     margin = h * 0.25
     analyzed_results["depth"] = 0
     if(hips[:, 1] >= knees[:, 1] - margin).any() :
          analyzed_results["depth"] = 1
     #CHECK HAND STRAIGHT
     left_shoulder_to_wrist = left_wrist[:3] - left_shoulder[:3]
     right_shoulder_to_wrist = right_wrist[:3] - right_shoulder[:3]
     left_shoulder_to_elbow = left_elbow[:3] - left_shoulder[:3]
     right_shoulder_to_elbow = right_elbow[:3] - right_shoulder[:3]
     analyzed_results["left_hand_straight"] = 0
     analyzed_results["right_hand_straight"] = 0
     if(abs(calculate_degree(left_shoulder_to_wrist, left_shoulder_to_elbow)) > 15):
          analyzed_results["left_hand_straight"] = 1
     if(abs(calculate_degree(right_shoulder_to_wrist, right_shoulder_to_elbow)) > 15):
          analyzed_results["right_hand_straight"] = 1
     
     return analyzed_results
    cap = cv2.VideoCapture(video_path)
    output_path = 'static\\videos\\output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    # Generate all columns of the data frame

    # Counter vars
    VISIBILITY_THRESHOLD = 0.6
    last_error_time = "00:00:00"
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seconds = count_seconds(frame_number, frame_rate)
            time = get_time_from_seconds(seconds)
            # Reduce size of a frame
            image = rescale_frame(image, 100)

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            if not results.pose_landmarks:
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Make detection
            try:
                # * Model prediction for SQUAT counter
                # Extract keypoints from frame for the input

                # Evaluate model prediction

                # Analyze squat pose
                analyzed_results= analyze_pose(results=results, visibility_threshold=VISIBILITY_THRESHOLD)

                torso_evaluation = analyzed_results["torso"]
                legs_evaluation = analyzed_results["stance"]
                grip_evaluation = analyzed_results["grip"]
                depth_evaluation = analyzed_results["depth"]
                left_hand_straight = analyzed_results["left_hand_straight"]
                right_hand_straight = analyzed_results["right_hand_straight"]
                state_evaluation = analyzed_results["state"]
                # * Evaluate FOOT PLACEMENT error
                if torso_evaluation == -1:
                    torso_placement = "UNK"
                elif torso_evaluation == 0:
                    torso_placement = "Correct"
                elif torso_evaluation == 1:
                    torso_placement = "Curved"
                
                # * Evaluate KNEE PLACEMENT error
                if legs_evaluation == -1:
                    legs_placement = "UNK"
                elif legs_evaluation == 0:
                    legs_placement = "Correct"
                elif legs_evaluation == 1:
                    legs_placement = "Too wide"
                elif legs_evaluation == 2:
                    legs_placement = "Too tight"
                
                if grip_evaluation == -1:
                    grip_placement = "UNK"
                elif grip_evaluation == 0:
                    grip_placement = "Correct"
                elif grip_evaluation == 1:
                    grip_placement = "Too wide"
                elif grip_evaluation == 2:
                    grip_placement = "Too tight"
                
                # if left_hand_straight == -1 : 
                #     left_hand_placement = "UNK"
                # elif left_hand_straight == 0 : 
                #     left_hand_placement = "Correct"
                # elif left_hand_straight == 1 : 
                #     left_hand_placement = "Curved"
                    
                # if right_hand_straight == -1 : 
                #     right_hand_placement = "UNK"
                # elif right_hand_straight == 0 : 
                #     right_hand_placement = "Correct"
                # elif right_hand_straight == 1 : 
                #     right_hand_placement = "Curved"
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "TORSO", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, torso_placement, (5, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Foot and Shoulder width ratio
                cv2.putText(image, "LEGS", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, legs_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display knee and Shoulder width ratio
                cv2.putText(image, "GRIP", (330, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, grip_placement, (325, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(image, "LEFT_HAND", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, str(left_hand_straight), (455, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # cv2.putText(image, "RIGHT_HAND", (590, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(image,str(right_hand_straight), (585, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                if torso_evaluation == 1:
                    if(time == last_error_time) :
                        continue
                    else :
                        last_error_time = time
                        error_count["Torso_Curved"] += 1
                        cv2.imwrite('./static/images/Torso_Curved_At_' + time +'.jpg', image)
                
                # * Evaluate KNEE PLACEMENT error
                if legs_evaluation == 1:
                    if(time == last_error_time) :
                        continue
                    else :
                        last_error_time = time
                        error_count["Foot_Wide"] += 1
                        cv2.imwrite('./static/images/Foot_Wide_At_' + time +'.jpg', image)
                elif legs_evaluation == 2:
                    if(time == last_error_time) :
                        continue
                    else :
                        last_error_time = time
                        error_count["Foot_Tight"] += 1
                        cv2.imwrite('./static/images/Foot_Tight_At_' + time +'.jpg', image)
                
                if grip_evaluation == 1:
                    if(time == last_error_time) :
                        continue
                    else :
                        last_error_time = time
                        error_count["Grip_Wide"] += 1
                        cv2.imwrite('./static/images/Grip_Wide_At_' + time +'.jpg', image)
                elif grip_evaluation == 2:
                    if(time == last_error_time) :
                        continue
                    else :
                        last_error_time = time
                        error_count["Grip_Tight"] += 1
                        cv2.imwrite('./static/images/Grip_Tight_At_' + time +'.jpg', image)
            except Exception as e:
                print(f"Error: {e}")
            out.write(image)
        out.release()
    total_error_count = sum(error_count.values())
    return output_path, error_count, total_error_count
def bicep_curl(video_path) : 
    IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    ]
    error_count = {
        "Right_Loose": 0,
        "Left_Loose": 0,
        "Lean_Back": 0
    }
    class BicepPoseAnalysis:
        def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
            # Initialize thresholds
            self.stage_down_threshold = stage_down_threshold
            self.stage_up_threshold = stage_up_threshold
            self.peak_contraction_threshold = peak_contraction_threshold
            self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
            self.visibility_threshold = visibility_threshold

            self.side = side
            self.counter = 0
            self.stage = "down"
            self.is_visible = True
            self.detected_errors = {
                "LOOSE_UPPER_ARM": 0,
                "PEAK_CONTRACTION": 0,
                
            }

            # Params for loose upper arm error detection
            self.loose_upper_arm = False

            # Params for peak contraction error detection
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        
        def get_joints(self, landmarks) -> bool:
            '''
            Check for joints' visibility then get joints coordinate
            '''
            side = self.side.upper()

            # Check visibility
            joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

            is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
            self.is_visible = is_visible

            if not is_visible:
                return self.is_visible
            
            # Get joints' coordinates
            self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
            self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
            self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

            return self.is_visible
        
        def analyze_pose(self, landmarks, frame):
            '''
            - Bicep Counter
            - Errors Detection
            '''
            self.get_joints(landmarks)

            # Cancel calculation if visibility is poor
            if not self.is_visible:
                return (None, None)

            # * Calculate curl angle for counter
            bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
            if bicep_curl_angle > self.stage_down_threshold:
                self.stage = "down"
            elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
                self.stage = "up"
                self.counter += 1
            
            # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
            shoulder_projection = [ self.shoulder[0], 1 ] # Represent the projection of the shoulder to the X axis
            ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

            # * Evaluation for LOOSE UPPER ARM error
            if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
                # Limit the saved frame
                if not self.loose_upper_arm:
                    self.loose_upper_arm = True
                    # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                    self.detected_errors["LOOSE_UPPER_ARM"] += 1
            else:
                self.loose_upper_arm = False
            
            # * Evaluate PEAK CONTRACTION error
            if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
                # Save peaked contraction every rep
                self.peak_contraction_angle = bicep_curl_angle
                self.peak_contraction_frame = frame
                
            elif self.stage == "down":
                # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
                if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                    # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                    self.detected_errors["PEAK_CONTRACTION"] += 1
                
                # Reset params
                self.peak_contraction_angle = 1000
                self.peak_contraction_frame = None
            
            return (bicep_curl_angle, ground_upper_arm_angle)
    headers = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    with open("./model/bicep_input_scaler.pkl", "rb") as f:
        input_scaler = pickle.load(f)
    with open("./model/bicep_model.pkl", "rb") as f:
        sklearn_model = pickle.load(f)
    cap = cv2.VideoCapture(video_path)
    output_path = 'static\\videos\\output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# Counter vars
    VISIBILITY_THRESHOLD = 0.65

# Params for counter
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120

# Params to catch FULL RANGE OF MOTION error
    PEAK_CONTRACTION_THRESHOLD = 60

# LOOSE UPPER ARM error detection
    LOOSE_UPPER_ARM = False
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40

# STANDING POSTURE error detection
    POSTURE_ERROR_THRESHOLD = 0.7
    posture = "C"

# Init analysis class
    left_arm_analysis = BicepPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

    right_arm_analysis = BicepPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)
    last_error_time = "00:00:00"


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seconds = count_seconds(frame_number, frame_rate)
            time = get_time_from_seconds(seconds)
            # Reduce size of a frame
            image = rescale_frame(image, 100)
            video_dimensions = [image.shape[1], image.shape[0]]
            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if not results.pose_landmarks:
                # out.write(image)
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Make detection
            try:
                landmarks = results.pose_landmarks.landmark
            
                (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(landmarks=landmarks, frame=image)
                # * Model prediction for SQUAT counter
                # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row], columns=headers[1:])
                X = pd.DataFrame(input_scaler.transform(X))
                # Make prediction and its probability
                predicted_class = sklearn_model.predict(X)[0]
                prediction_probabilities = sklearn_model.predict_proba(X)[0]
                class_prediction_probability = round(prediction_probabilities[np.argmax(prediction_probabilities)], 2)
                # Evaluate model prediction
                if class_prediction_probability >= POSTURE_ERROR_THRESHOLD:
                    posture = predicted_class
                # Analyze squat pose
                # * Evaluate FOOT PLACEMENT error
                    
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (500, 40), (245, 117, 16), -1)

            # Display probability
                cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Left Counter
                cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # * Display error
            # Right arm error
                cv2.putText(image, "R_PC", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "R_LUA", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Left arm error
                cv2.putText(image, "L_PC", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.detected_errors["PEAK_CONTRACTION"]), (295, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "L_LUA", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"]), (375, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Lean back error
                cv2.putText(image, "LB", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"{predicted_class}", (440, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                print(posture)
                if left_arm_analysis.is_visible:
                    cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


            # Visualize RIGHT arm calculated angles
                if right_arm_analysis.is_visible:
                    cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                if posture != "C":
                    if(time == last_error_time) :
                        continue
                    else:
                        last_error_time = time
                        error_count["Lean_Back"] += 1
                        cv2.imwrite('./static/images/Lean_Back_At_' + time +'.jpg', image)
                if left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"] > error_count["Left_Loose"]:
                    if(time == last_error_time) :
                        continue
                    else:
                        error_count["Left_Loose"] = left_arm_analysis.detected_errors["LOOSE_UPPER_ARM"] 
                        last_error_time = time
                        cv2.imwrite('./static/images/Left_Hand_Loose_At_' + time +'.jpg', image)
                if right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"] > error_count["Right_Loose"]:
                    if(time == last_error_time) :
                        continue
                    else:
                        error_count["Right_Loose"] = right_arm_analysis.detected_errors["LOOSE_UPPER_ARM"] 
                        last_error_time = time
                        cv2.imwrite('./static/images/Right_Hand_Loose_At_' + time +'.jpg', image)
            except Exception as e:
                print(f"Error: {e}")
            out.write(image)
        out.release()
    total_error_count = sum(error_count.values())
    return output_path, error_count, total_error_count
def lunge(video_path) : 
# Determine important landmarks for lunge
    IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
    ]
    error_count = {
        "KOT" : 0
    }
    def analyze_knee_angle(
        mp_results, stage: str, angle_thresholds: list, draw_to_image: tuple = None
    ):
        """
        Calculate angle of each knee while performer at the DOWN position

        Return result explanation:
            error: True if at least 1 error
            right
                error: True if an error is on the right knee
                angle: Right knee angle
            left
                error: True if an error is on the left knee
                angle: Left knee angle
        """
        results = {
            "error": None,
            "right": {"error": None, "angle": None},
            "left": {"error": None, "angle": None},
        }

        landmarks = mp_results.pose_landmarks.landmark

        # Calculate right knee angle
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        ]
        right_knee = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
        ]
        right_ankle = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
        ]
        results["right"]["angle"] = calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate left knee angle
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        ]
        left_knee = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
        ]
        left_ankle = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        ]
        results["left"]["angle"] = calculate_angle(left_hip, left_knee, left_ankle)

        # Draw to image
        if draw_to_image is not None and stage != "down":
            (image, video_dimensions) = draw_to_image

            # Visualize angles
            cv2.putText(
                image,
                str(int(results["right"]["angle"])),
                tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(int(results["left"]["angle"])),
                tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if stage != "down":
            return results

        # Evaluation
        results["error"] = False

        if angle_thresholds[0] <= results["right"]["angle"] <= angle_thresholds[1]:
            results["right"]["error"] = False
        else:
            results["right"]["error"] = True
            results["error"] = True

        if angle_thresholds[0] <= results["left"]["angle"] <= angle_thresholds[1]:
            results["left"]["error"] = False
        else:
            results["left"]["error"] = True
            results["error"] = True

        # Draw to image
        if draw_to_image is not None:
            (image, video_dimensions) = draw_to_image

            right_color = (255, 255, 255) if not results["right"]["error"] else (0, 0, 255)
            left_color = (255, 255, 255) if not results["left"]["error"] else (0, 0, 255)

            right_font_scale = 0.5 if not results["right"]["error"] else 1
            left_font_scale = 0.5 if not results["left"]["error"] else 1

            right_thickness = 1 if not results["right"]["error"] else 2
            left_thickness = 1 if not results["left"]["error"] else 2

            # Visualize angles
            cv2.putText(
                image,
                str(int(results["right"]["angle"])),
                tuple(np.multiply(right_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                right_font_scale,
                right_color,
                right_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                str(int(results["left"]["angle"])),
                tuple(np.multiply(left_knee, video_dimensions).astype(int)),
                cv2.FONT_HERSHEY_COMPLEX,
                left_font_scale,
                left_color,
                left_thickness,
                cv2.LINE_AA,
            )

        return results
    headers = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    with open("./model/lunge_input_scaler.pkl", "rb") as f:
        input_scaler = pickle.load(f)
    with open("./model/sklearn/lunge_stage_SVC_model.pkl", "rb") as f:
        stage_sklearn_model = pickle.load(f)
    with open("./model/sklearn/lunge_err_LR_model.pkl", "rb") as f:
        err_sklearn_model = pickle.load(f)
    cap = cv2.VideoCapture(video_path)
    output_path = 'static\\videos\\output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# Counter vars
    current_stage = ""
    counter = 0

    prediction_probability_threshold = 0.8
    ANGLE_THRESHOLDS = [60, 135]

    knee_over_toe = False

    last_error_time = "00:00:00"


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seconds = count_seconds(frame_number, frame_rate)
            time = get_time_from_seconds(seconds)
            # Reduce size of a frame
            image = rescale_frame(image, 100)
            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if not results.pose_landmarks:
                # out.write(image)
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Make detection
            try:
            # Extract keypoints from frame for the input
                row = extract_important_keypoints(results)
                X = pd.DataFrame([row], columns=headers[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                stage_predicted_class = stage_sklearn_model.predict(X)[0]
                stage_prediction_probabilities = stage_sklearn_model.predict_proba(X)[0]
                stage_prediction_probability = round(stage_prediction_probabilities[stage_prediction_probabilities.argmax()], 2)

                # Evaluate model prediction
                if stage_predicted_class == "I" and stage_prediction_probability >= prediction_probability_threshold:
                    current_stage = "init"
                elif stage_predicted_class == "M" and stage_prediction_probability >= prediction_probability_threshold: 
                    current_stage = "mid"
                elif stage_predicted_class == "D" and stage_prediction_probability >= prediction_probability_threshold:
                    if current_stage in ["mid", "init"]:
                        counter += 1
                    
                    current_stage = "down"
                
                # Error detection
                # Knee angle
                analyze_knee_angle(mp_results=results, stage=current_stage, angle_thresholds=ANGLE_THRESHOLDS, draw_to_image=(image, video_dimensions))

                # Knee over toe
                err_predicted_class = err_prediction_probabilities = err_prediction_probability = None
                if current_stage == "down":
                    err_predicted_class = err_sklearn_model.predict(X)[0]
                    err_prediction_probabilities = err_sklearn_model.predict_proba(X)[0]
                    err_prediction_probability = round(err_prediction_probabilities[err_prediction_probabilities.argmax()], 2)
                    
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (800, 45), (245, 117, 16), -1)

                # Display stage prediction
                cv2.putText(image, "STAGE", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(stage_prediction_probability), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Display error prediction
                cv2.putText(image, "K_O_T", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(err_prediction_probability), (195, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(err_predicted_class), (245, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                print(err_predicted_class)
                # Display Counter
                cv2.putText(image, "COUNTER", (110, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (110, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if err_predicted_class != "C" :
                    if time == last_error_time:
                        continue
                    else :
                        cv2.imwrite('./static/images/Knee_Overs_Toe' + time +'.jpg', image)
                        error_count["KOT"] += 1
                        last_error_time = time
            except Exception as e:
                print(f"Error: {e}")
            out.write(image)
        out.release()
    total_error_count = sum(error_count.values())
    return output_path, error_count, total_error_count
def plank(video_path) : 
# Determine important landmarks for lunge
    IMPORTANT_LMS = [
        "NOSE",
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_ELBOW",
        "RIGHT_ELBOW",
        "LEFT_WRIST",
        "RIGHT_WRIST",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]
    error_count = {
        "High_Back": 0,
        "Low_Back": 0
    }
    headers = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]
    with open("./model/plank_LR_model.pkl", "rb") as f:
        sklearn_model = pickle.load(f)

    # Dump input scaler
    with open("./model/plank_input_scaler.pkl", "rb") as f2:
        input_scaler = pickle.load(f2)
    cap = cv2.VideoCapture(video_path)
    output_path = 'static\\videos\\output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    # out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# Counter vars
    current_stage = ""
    prediction_probability_threshold = 0.6

    last_error_time = "00:00:00"


    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            seconds = count_seconds(frame_number, frame_rate)
            time = get_time_from_seconds(seconds)
            # Reduce size of a frame
            image = rescale_frame(image, 100)
            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            if not results.pose_landmarks:
                # out.write(image)
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
            # Make detection
            try:
            # Extract keypoints from frame for the input
                row = extract_important_keypoints(results, IMPORTANT_LMS)
                X = pd.DataFrame([row], columns=headers[1:])
                X = pd.DataFrame(input_scaler.transform(X))

                # Make prediction and its probability
                predicted_class = sklearn_model.predict(X)[0]
                prediction_probability = sklearn_model.predict_proba(X)[0]
                # print(predicted_class, prediction_probability)
                print(prediction_probability)
                # Evaluate model prediction
                if predicted_class == 0 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold:
                    current_stage = "Correct"
                elif predicted_class == 1 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    current_stage = "Low back"
                elif predicted_class == 2 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    current_stage = "High back"
                else:
                    current_stage = "unk"
                
                # Visualization
                # Status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "CLASS", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, current_stage, (90, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display probability
                cv2.putText(image, "PROB", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(prediction_probability[np.argmax(prediction_probability)], 2)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if predicted_class == 1 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    if time == last_error_time :
                        continue
                    else :
                        last_error_time = time
                        error_count["Low_Back"] += 1
                        cv2.imwrite('./static/images/Low_Back_At_' + time +'.jpg', image)
                elif predicted_class == 2 and prediction_probability[prediction_probability.argmax()] >= prediction_probability_threshold: 
                    if time == last_error_time :
                        continue
                    else :
                        last_error_time = time
                        error_count["High_Back"] += 1
                        cv2.imwrite('./static/images/High_Back_At_' + time +'.jpg', image)
            except Exception as e:
                print(f"Error: {e}")
            out.write(image)
        out.release()
    total_error_count = sum(error_count.values())
    return output_path, error_count, total_error_count