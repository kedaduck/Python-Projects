import cv2
import face_recognition

# 1 read face data
# 1.1 read picture
yuan = cv2.imread('dataset/yuan01.jpg')
me = cv2.imread('dataset/me01.png')
# 1.2 encoding for picture
yuan_encoding = face_recognition.face_encodings(yuan)[0]
zhao_encoding = face_recognition.face_encodings(me)[0]

# 1.3 read encoding list for some pictures
know_face_encodings = [yuan_encoding, zhao_encoding]
know_face_names = ['yuan', 'me']

# 2 capture picture of video
vc = cv2.VideoCapture(0)
while True:
    ret, img = vc.read();
    if not ret:
        print('no picture')
        break
    # 1. find location of picture
    locations = face_recognition.face_locations(img)
    # 2. encoding for picture
    face_encodings = face_recognition.face_encodings(img, locations)
    for (top, right, bottom, left), face_encoding in zip(locations,face_encodings):
        # 3 find location of face which in video
        matchs = face_recognition.compare_faces(know_face_encodings, face_encoding)
        name = 'unknown'
        for match, know_name in zip(matchs, know_face_names):
            if match:
                name = know_name
                break
        # 4 is compare with picture
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(img, name, (left, top-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) != -1:
        vc.release()
        cv2.destroyAllWindows()
        break
