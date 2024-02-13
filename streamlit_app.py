import streamlit as st
import cv2
import os
import numpy as np
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
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

def set_timezone():
    current_local_time = datetime.now()
    timezone_js = """
        <script>
            var timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            var data = { timezone: timezone };
            fetch("/set_timezone", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });
        </script>
    """
    st.markdown(timezone_js, unsafe_allow_html=True)

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
    return model.predict(facearray.reshape(1, -1))

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    # Check if there are users with images
    if not userlist:
        print("No users found for training.")
        return

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)

    # Check if faces array is not empty
    if faces.size == 0:
        print("No faces found for training.")
        return

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
    df_home = pd.DataFrame({"Name": names, "Roll": rolls, "Time": times})
    st.table(df_home)

    st.write(f"Total Registered Students: {totalreg()}")

def take_attendance_page():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        st.warning("There is no trained model in the static folder. Please add a new face to continue.")
        return

    st.write("## Taking Attendance")

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()

        # convert image from opened file to np.array
        image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_array_copy = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(image_array_copy, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image_array_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(image_array_copy[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(image_array_copy, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        # Display the resulting frame
        st.image(image_array_copy, channels="BGR", use_column_width=True)

def add_student_page():
    st.title("Capture Images 10 various poses")
    newusername = st.text_input('Enter new username:')
    newuserid = st.text_input('Enter new user ID:')
    userimagefolder = 'static/faces/' + newusername + '_' +'ID:'+ str(newuserid)

    # Check if the user folder already exists
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i = 0

    # Capture 10 images for training using Streamlit's camera input
    for i in range(10):
        img_file_buffer = st.camera_input(f"Take picture {i + 1}", key=f"image_{i}")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            faces = face_detector.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(image_array, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(image_array, f'Images Captured: {i + 1}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 20), 2, cv2.LINE_AA)

                # Save the captured image
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(os.path.join(userimagefolder, name), image_array[y:y + h, x:x + w])

    st.success("Images Captured. Training the model...")

    # Train the model after capturing images
    train_model()

    # Display success message or other relevant information
    st.success("Training complete.")
if __name__ == "__main__":
    main()