import os
import streamlit as st
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import cv2

# Streamlit app layout
st.set_page_config(page_title="Attendance System", page_icon="✅")

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    if img is not None and len(img) > 0:  # Check if img is not empty
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    # Ensure that faces is a 2D array
    faces = np.array(faces)
    faces = faces.reshape(-1, 50 * 50 * 3)  # Assuming the image size is 50x50 with 3 channels (BGR)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

# Streamlit app layout
st.title("Attendance System")

# Sidebar navigation
st.sidebar.header("Navigation")
selected_page = st.sidebar.radio("Go to:", ["Home", "Take Attendance", "Add New User"])

if selected_page == "Home":
    names, rolls, times, l = extract_attendance()
    st.write(f"Date: {datetoday2}")
    st.write("Total Registered Users:", totalreg())
    st.write("Attendance:")
    if l > 0:
        st.table(pd.DataFrame({"Name": names, "Roll": rolls, "Time": times}))
    else:
        st.write("No attendance recorded yet.")

elif selected_page == "Take Attendance":
    st.write("Press 'q' to quit the camera feed.")
    st.write("Attendance:")
    
    # Using cv2.VideoCapture to capture video frames
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        # Display the resulting frame
        st.image(frame, caption="Processed Image", channels="BGR", use_column_width=True)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

elif selected_page == "Add New User":
    st.title("Add New User")
    newusername = st.text_input("Enter new user name:")
    newuserid = st.text_input("Enter new user ID:")
    if st.button("Start Adding User"):
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        # Use OpenCV to capture camera frames
        cap = cv2.VideoCapture(0)
        capture_count = 0
        while capture_count < 50:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                resized_face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                user_image = resized_face.ravel()
                if len(user_image) > 0:
                    faces.append(user_image)

                capture_count += 1
                st.write(f"Images Captured: {capture_count}/50")

            st.image(frame, channels="BGR", caption="Adding User", use_container_width=True)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

        if len(faces) > 0:
            # Ensure that faces is a 2D array
            faces = np.array(faces)
            faces = faces.reshape(-1, 50 * 50 * 3)  # Assuming the image size is 50x50 with 3 channels (BGR)

            st.success('Training Model...')
            train_model()
            names, rolls, times, l = extract_attendance()
            if totalreg() > 0:
                st.success("User added successfully!")
                st.button("Go to Attendance", key="go_to_attendance")
            else:
                st.error("Failed to add user. Please try again.")
        else:
            st.error("No face detected. Please try again.")