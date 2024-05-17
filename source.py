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
