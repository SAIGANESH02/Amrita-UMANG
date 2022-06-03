#!/usr/bin/python
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import json

def etl(path,wr_path):
    # Setup capture to video 
    cap = cv2.VideoCapture(path)

    pathp = path.split('_')
    s_id = int(pathp[1][-1])
    g_id = int(pathp[2][-1])
    if pathp[0][1] == 'H':
        env = "Home"
        pos = path[-1].split('.')[0]
    else:
        env = "Studio"
        pos = 'S'

    # Properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
                                                                                            
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        
    def center_crop(img, dim):
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
        return crop_img

    # Video Writer
    crop_path = wr_path+ r'/cropvid.mp4'
    video_writer = cv2.VideoWriter(crop_path , cv2.VideoWriter_fourcc('P','I','M','1'), fps, (320, 320), isColor=True) 
                                                                                            
    # Loop through each frame
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        # Read frame 
        ret, frame = cap.read()
        
        frame = center_crop(frame, (320,320))

        # Show image
        # cv2.imshow('Video Player', frame)
        
        # Write out frame 
        video_writer.write(frame)
        
        # Breaking out of the loop
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Close down everything
    cap.release()
    cv2.destroyAllWindows()
    # Release video writer
    video_writer.release()

    def mediapipe_detection(image, model):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connection

    def draw_styled_landmarks(image, results):
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 
    # Video Writer
    posepath = wr_path+ r'/posevid.mp4'
    video_writer = cv2.VideoWriter(posepath , cv2.VideoWriter_fourcc('P','I','M','1'), fps, (width, height), isColor=True)                         
    cap = cv2.VideoCapture(path)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            if(frame is not None): 
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Show to screen
                cv2.imshow('OpenCV Feed',image)

                video_writer.write(image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                cap.release()
                cv2.destroyAllWindows()
        draw_landmarks(frame, results)

    pose = []
    for res in results.pose_landmarks.landmark:
        test = np.array([res.x, res.y, res.z, res.visibility])
        pose.append(test)

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    def extract_keypoints(results):
        # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        pose_x = np.array([[res.x] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        pose_y = np.array([[res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        lh_x = np.array([[res.x] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        lh_y = np.array([[res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

        # rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        rh_x = np.array([[res.x] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        rh_y = np.array([[res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return pose_x,pose_y,lh_x,lh_y,rh_x,rh_y

    pose_x,pose_y,lh_x,lh_y,rh_x,rh_y = extract_keypoints(results)

    # Data to be written
    dictionary = {
        "pose_x" : pose_x.tolist(),
        "pose_y" : pose_y.tolist(),
        
        "hand1_x" : rh_x.tolist(),
        "hand1_y" : rh_y.tolist(),

        "hand2_x" : lh_x.tolist(),
        "hand2_y" : lh_y.tolist()   
    }

    with open("params.json", "w") as outfile:
        json.dump(dictionary, outfile)

    print(path,env,s_id,g_id,pos,n_frames,fps,height,width,crop_path,posepath)

x = sys.argv[1]
y = sys.argv[2]
etl(x,y)
