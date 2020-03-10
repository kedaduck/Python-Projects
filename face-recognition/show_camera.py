import cv2

# 1 import
# 2 create video capture entry
vc = cv2.VideoCapture(0)

# 3 capture picture of video continue
while True:
    ret, img = vc.read()
    if not ret:
        print('no capture video')
        break

    # 4 show picture
    cv2.imshow('me', img)
    # 5 wait keyborad
    if cv2.waitKey(1) != -1:
        # 6 close camera and windows
        vc.release()
        cv2.destroyAllWindows()
        break
