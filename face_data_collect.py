import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_data = []
dataset_path = 'data'
skip = 0
file_name = input('Enter name of person: ')

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # f[2]*f[3] means area Width x Height
    faces = sorted(faces, key=lambda f: f[2]*f[3],  reverse=True)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract (crop the required part of face) also known as region of initerest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data), end='\r')

    cv2.imshow('video', frame)
    #cv2.imshow('captured', face_section)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# convert face_data or face list into a numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save this data intp file system
np.save(os.path.join(dataset_path, file_name), face_data)
print('success')
cap.release()
cv2.destroyAllWindows()
