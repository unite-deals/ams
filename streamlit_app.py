import streamlit as st
import cv2
import os
import numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

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
    if img != []:
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
    faces = np.array(faces)
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

# Streamlit app
def main():
    st.title("Attendance Management System")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Take Attendance", "Add Student"])

    if page == "Home":
        home_page()
    elif page == "Take Attendance":
        take_attendance_page()
    elif page == "Add Student":
        add_student_page()

def home_page():
    names, rolls, times, l = extract_attendance()
    st.write(f"## Today's Attendance ({datetoday2})")
    st.table(pd.DataFrame({"Name": names, "Roll": rolls, "Time": times}))

    st.write(f"Total Registered Students: {totalreg()}")

def take_attendance_page():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        st.warning("There is no trained model in the static folder. Please add a new face to continue.")
        return

    st.write("## Taking Attendance")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    while ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        # Display the resulting frame
        st.image(frame, channels="BGR")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

def add_student_page():
    st.write("## Add New Student")

    newusername = st.text_input("Enter Student Name:")
    newuserid = st.text_input("Enter Student ID:")

    if st.button("Add Student"):
        userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        cap = cv2.VideoCapture(0)
        i, j = 0, 0
        while j < 500:
            i, frame = cap.read()
            faces = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(faces, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                if j % 10 == 0:
                    name = newusername + '_' + str(i) + '.jpg'
                    cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            st.image(frame, channels="BGR")
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        st.write('Training Model')
        train_model()

        st.success("Student added successfully!")

if __name__ == "__main__":
    main()
