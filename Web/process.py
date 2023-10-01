import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import math 
from static_remover import clear_folder
# Drawing helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

analyzed_results = None 

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


xaxis = np.array([[1, 0, 0]])
yaxis = np.array([[0, 1, 0]])
zaxis = np.array([[0, 0, 1]])
uaxis = np.array([[1, 1, 1]])
def point_to_array(point) :
     return np.array([point.x, point.y, point.z, point.visibility])
def calculate_degree(pointX, pointY) : 
     return np.degrees(np.arccos(pointY @ pointX.T))
def check_perpendicular_limb(pointX, pointY, allowed_error = 15) :
     limb_xaxis_angle = np.degrees(np.arccos(pointX @ pointY.T))
     print(limb_xaxis_angle)
     if abs(limb_xaxis_angle - 90) > allowed_error:
        return False
     else:
        return True
def calculate_distance3D(pointX, pointY) :
     return np.linalg.norm(pointY - pointX)

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
            print("Cannot see foot")
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
                # print(time)
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
    total_error_count = sum(error_count.values())
    return error_count, total_error_count
def deadlift(video_path) : 
    analyzed_results = {
     "torso": -1,
     "stance":-1,
     "grip":-1,
     "depth":-1,
     "left_hand_straight":-1,
     "right_hand_straight":-1,
     "state": -1  
     }    
    def analyze_pose(results, visibility_threshold: int) : 
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
     h = ( calculate_distance(left_ankle[:3], left_knee[:3]) +  calculate_distance(right_ankle[:3], right_knee[:3]) ) /2
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

    # Generate all columns of the data frame

    headers = ["label"] # Label column

    for lm in IMPORTANT_LMS:
        headers += [f"{lm.lower()}_x", f"{lm.lower()}_y", f"{lm.lower()}_z", f"{lm.lower()}_v"]

# Counter vars
    cap = cv2.VideoCapture(video_path)

    # Counter vars
    VISIBILITY_THRESHOLD = 0.6

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break
            
            # Reduce size of a frame
            image = rescale_frame(image, 50)

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
                cv2.rectangle(image, (0, 0), (900, 60), (245, 117, 16), -1)

                # Display class
                cv2.putText(image, "TORSO", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, torso_placement, (5, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Foot and Shoulder width ratio
                cv2.putText(image, "LEGS", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, legs_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                # Display knee and Shoulder width ratio
                cv2.putText(image, "GRIP", (330, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, grip_placement, (325, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "LEFT_HAND", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_hand_straight), (455, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(image, "RIGHT_HAND", (590, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image,str(right_hand_straight), (585, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")
            
            cv2.imshow("CV2", image)
            
            # Press Q to close cv2 window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # (Optional)Fix bugs cannot close windows in MacOS (https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv)
        for i in range (1, 5):
            cv2.waitKey(1)
  
  

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