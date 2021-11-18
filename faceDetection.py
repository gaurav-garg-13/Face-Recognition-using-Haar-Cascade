import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    # returns coordinates of face (x,y) and width ad height
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video Frame', frame)

    # when user input q stop the stream
    # we use & 0xFF to convert the 32 bit integer returned from cv2.waitkey() to a 8 bit value
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
