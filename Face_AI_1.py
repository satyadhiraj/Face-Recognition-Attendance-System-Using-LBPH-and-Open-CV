import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request

app = Flask(__name__)

# Constants
START_TIME = "12:00"
END_TIME = "13:00"

# Function to register a face
def register_face(name, student_id, student_class):
    # Create directory if it doesn't exist
    if not os.path.exists("faces"):
        os.makedirs("faces")

    # Initialize OpenCV's face recognizer
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(f"faces/{name}_{student_id}.jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Register Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"{name}'s face registered successfully!")

    # Save registration information to a CSV file
    registration_data = pd.DataFrame({'Name': [name], 'ID': [student_id], 'Class': [student_class]})
    registration_data.to_csv('registration_data.csv', mode='a', header=not os.path.exists('registration_data.csv'), index=False)

    video_capture.release()
    cv2.destroyAllWindows()

# Function to train face recognizer
def train_face_recognizer():
    # Load registration data
    registration_data = pd.read_csv('registration_data.csv')

    # Initialize OpenCV's face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Initialize OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    ids = []

    # Load face images and corresponding IDs
    for index, row in registration_data.iterrows():
        face_id = row['ID']
        face_path = f'faces/{row["Name"]}_{row["ID"]}.jpg'
        if os.path.exists(face_path):
            face_image = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
            faces.append(face_image)
            ids.append(int(face_id))  # Convert ID to int for compatibility

    # Train the face recognizer
    face_recognizer.train(faces, np.array(ids))

    # Save the trained model
    face_recognizer.save('face_trainer.yml')

# Function to take attendance
def take_attendance():
    start_time = datetime.strptime(START_TIME, "%H:%M")
    end_time = datetime.strptime(END_TIME, "%H:%M")

    # Load registration data
    registration_data = pd.read_csv('registration_data.csv')

    # Initialize face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trainer.yml')

    # Initialize OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    # Define attendance data
    current_date = datetime.now().strftime("%Y-%m-%d")
    attendance_file = 'attendance.csv'
    attendance_data = pd.DataFrame(columns=['Name', current_date])

    if os.path.exists(attendance_file):
        attendance_data = pd.read_csv(attendance_file)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            # Recognize faces
            id_, confidence = face_recognizer.predict(roi_gray)
            if confidence >= 45:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = registration_data.loc[registration_data['ID'] == id_]['Name'].values[0]
                cv2.putText(frame, f'{name} - {confidence}%', (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Mark attendance in Excel sheet if within the time range
                current_time = datetime.now().strftime("%H:%M")
                if start_time.time() <= datetime.strptime(current_time, "%H:%M").time() <= end_time.time():
                    if name in attendance_data['Name'].values:
                        attendance_data.loc[attendance_data['Name'] == name, current_date] = 'Present'
                    else:
                        new_row = pd.DataFrame({'Name': [name], current_date: ['Absent']})
                        attendance_data = pd.concat([attendance_data, new_row], ignore_index=True)
                else:
                    if name not in attendance_data['Name'].values:
                        new_row = pd.DataFrame({'Name': [name], current_date: ['Absent']})
                        attendance_data = pd.concat([attendance_data, new_row], ignore_index=True)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Take Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Save attendance to a CSV file
    attendance_data.to_csv(attendance_file, index=False)

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register_face", methods=["POST"])
def do_register_face():
    name = request.form["name"]
    student_id = request.form["id"]
    student_class = request.form["class"]
    register_face(name, student_id, student_class)
    return render_template("index.html", message="Face registered successfully!")

@app.route("/train_face_recognizer")
def do_train_face_recognizer():
    train_face_recognizer()
    return render_template("index.html", message="Face recognizer trained successfully!")

@app.route("/take_attendance")
def do_take_attendance():
    take_attendance()
    return render_template("index.html", message="Attendance taken successfully!")

if __name__ == "__main__":
    app.run(debug=True)
