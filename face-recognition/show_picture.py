# opencv import package cv2
import cv2


# 1 read picture
img = cv2.imread('dataset/obama.jpg')


# 2 show
cv2.imshow('obama', img)

# 3 wait keyborad
cv2.waitKey(0)



# 4  destory windows
cv2.destroyAllWindows()

