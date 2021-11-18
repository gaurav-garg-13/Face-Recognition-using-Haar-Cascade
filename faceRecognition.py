import numpy as np
import cv2
import os


def dist(x1, x2):
    # Eucledian distance
    return np.sqrt(sum((x1-x2)**2))


def knn(train, test, k=7):
    dis = []

    for i in range(train.shape[0]):
        # get vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        d = dist(test, ix)
        dis.append([d, iy])

    # sort based on distance and get top k
    dk = sorted(dis, key=lambda x: x[0])[:k]
    # retreive the labels
    labels = np.array(dk)[:, -1]

    # get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # find ,ax frequemcy and corresponding labels
    index = np.argmax(output[1])
    return output[0][index]

###################################################


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
skip = 0
face_data = []  # this is X_train
labels = []  # this is Y_train
dataset_path = 'data/'

class_id = 0  # labels for given file
names = {}  # mapping between id -name

# Data preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx.split('.')[0]
        print('loaded', fx)

        data_item = np.load(os.path.join(dataset_path+fx))
        face_data.append(data_item)

        # create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

print(face_dataset.shape, face_labels.shape)
train_set = np.concatenate((face_dataset, face_labels), axis=1)
print(train_set.shape)

# reading video stream
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        offset = 10
        face_section = frame[y-offset:y+w+offset, x-offset:x+h+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # predicted label
        out = knn(train_set, face_section.flatten())

        # display label and face with a box
        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Faces', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
