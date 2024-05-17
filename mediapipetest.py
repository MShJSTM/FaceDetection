# import cv2
# import mediapipe as mp

# cap = cv2.VideoCapture(0)

# mp_face_mesh = mp.solutions.face_mesh

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
#     while cap.isOpened():
#         success, image = cap.read()
        
#         if not success:
#             print("Ignoring empty camera frame.")
#             break
        
#         results = face_mesh.process(image)
        
#         if results.multi_face_landmarks is not None:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     # rest of your code...
#                     connection_drawing_spec=mp_drawing_styles
#                     .get_default_face_mesh_tesselation_style()
#                 )
            
            
            
#             cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            
#             if cv2.waitKey(100) & 0xFF == ord('q'):
#                 break
        
#     cap.release()
#     cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp

# # Initialize webcam input
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe FaceMesh solution
# mp_face_mesh = mp.solutions.face_mesh

# # Drawing utilities and styles
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Define custom drawing specifications (color in BGR format)
# custom_tesselation_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green color

# with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         success, image = cap.read()
        
#         if not success:
#             print("Ignoring empty camera frame.")
#             break
        
#         # Convert the BGR image to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Process the image and detect face landmarks
#         results = face_mesh.process(image_rgb)
        
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Draw the face mesh with custom color
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=custom_tesselation_spec  # Use custom drawing spec
#                 )
        
#         # Flip the image horizontally for a selfie-view display
#         cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam input
cap = cv2.VideoCapture(0)

# Initialize MediaPipe FaceMesh solution
mp_face_mesh = mp.solutions.face_mesh

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks (example indices from MediaPipe documentation)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Function to calculate the EAR
def calculate_ear(eye_landmarks, landmarks):
    # Vertical distances
    A = np.linalg.norm(np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) - 
                       np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) - 
                       np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y]))
    
    # Horizontal distance
    C = np.linalg.norm(np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) - 
                       np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y]))
    
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold for eye closure
EAR_THRESHOLD = 0.25

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            break
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect face landmarks
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Calculate EAR for both eyes
                left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
                right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
                
                # Determine if eyes are open or closed
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    eye_status = "Eyes Closed"
                else:
                    eye_status = "Eyes Open"
                
                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None
                )
        
        # Flip the image horizontally for a selfie-view display
     # Flip the image horizontally for a selfie-view display
        flipped_image = cv2.flip(image, 1)
        
        # Set the text color (red in this example)
        if eye_status == "Eyes Closed":
            text_color = (0, 0, 255)
        else:
            text_color=(0,255,0)
        
        # Display eye status on the flipped image
        cv2.putText(flipped_image, eye_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        
        cv2.imshow('MediaPipe Face Mesh', flipped_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
