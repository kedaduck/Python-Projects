import cv2
import face_recognition


# 1 read picture
obama = cv2.imread('dataset/me2.jpg')
img = cv2.imread('dataset/me3.jpg')

# 1.2  get location face
locations = face_recognition.face_locations(img)

#
obama_face_encoding = face_recognition.face_encodings(obama)[0]
know_face_encoding = [obama_face_encoding]
know_face_name = ['me']

unknow_encoding = face_recognition.face_encodings(img, locations)

for (top, right, bottom, left), face_encoding in zip(locations, unknow_encoding):
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    matchs = face_recognition.compare_faces(know_face_encoding, face_encoding)
    name = 'unknow'
    for match, know_name in zip(matchs, know_face_name):
        if match:
            name = know_name
            break
    #
    cv2.putText(img, name, (left, top-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)



# 1.3 use rect tag face


# 2 show
cv2.imshow('obama', img)

# 3 wait keyborad
cv2.waitKey(0)



# 4  destory windows
cv2.destroyAllWindows()

