import streamlit as st
import pandas as pd
import cv2
import os
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Set page configuration
st.set_page_config(page_title="Attendance Management", page_icon="✅")

# Database
database_path = "database.csv"
if not os.path.exists(database_path):
    columns = ["ID", "Name", "Time"]
    database = pd.DataFrame(columns=columns)
    database.to_csv(database_path, index=False)
else:
    database = pd.read_csv(database_path)

# Load face recognition model
knn_model_path = "face_recognition_model.pkl"
if os.path.exists(knn_model_path):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.zeros((1, 2500)), ["Unknown"])
else:
    st.warning("Face recognition model not found. Please add new students to train the model.")

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def identify_face(facearray):
    return knn.predict(facearray)

def add_attendance(student_id, student_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    database.loc[len(database)] = [student_id, student_name, current_time]
    database.to_csv(database_path, index=False)
    st.success(f"Attendance recorded for {student_name} (ID: {student_id}) at {current_time}")

def add_new_student(student_id, student_name):
    st.info("Please look into the camera to capture your face.")
    picture = st.camera_input("Take a picture", key="add_student_camera")
    
    if picture:
        np_image = np.array(picture)
        face = extract_faces(np_image)
        
        if len(face) == 1:
            resized_face = cv2.resize(np_image[face[0][1]:face[0][1]+face[0][3], face[0][0]:face[0][0]+face[0][2]], (50, 50))
            face_array = resized_face.ravel().reshape(1, -1)
            
            # Update face recognition model
            knn_model_path = "face_recognition_model.pkl"
            if os.path.exists(knn_model_path):
                knn.fit(np.vstack([knn.kneighbors_graph()["nonzero"][0], face_array]),
                        np.append(knn.predict(knn.kneighbors_graph()["nonzero"][0]), student_name))
            else:
                knn.fit(face_array, [student_name])

            # Save the captured face
            face_folder = f"faces/{student_name}_{student_id}"
            os.makedirs(face_folder, exist_ok=True)
            cv2.imwrite(f"{face_folder}/{student_name}_{student_id}.jpg", resized_face)

            # Update database and save
            add_attendance(student_id, student_name)
        else:
            st.warning("No face detected. Please try again.")

# Streamlit app layout
st.title("Attendance Management System")

# Sidebar navigation
selected_page = st.sidebar.radio("Go to:", ["View Database", "Take Attendance", "Add New Student"])

if selected_page == "View Database":
    st.subheader("Database")
    st.table(database)

elif selected_page == "Take Attendance":
    st.subheader("Take Attendance")
    st.write("Press 'q' to quit the camera feed.")
    
    picture = st.camera_input("Take a picture", key="take_attendance_camera")

    if picture:
        np_image = np.array(picture)
        faces = extract_faces(np_image)

        for (x, y, w, h) in faces:
            cv2.rectangle(np_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = cv2.resize(np_image[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person, "Unknown")
            cv2.putText(np_image, f'{identified_person} (Unknown)', (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        st.image(np_image, channels="BGR", caption="Processed Image", use_column_width=True)

elif selected_page == "Add New Student":
    st.subheader("Add New Student")
    new_student_id = st.text_input("Enter new student ID:")
    new_student_name = st.text_input("Enter new student name:")

    if st.button("Add Student"):
        add_new_student(new_student_id, new_student_name)
