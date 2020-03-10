# opencv import package cv2
import cv2


# 1 read picture
img = cv2.imread('dataset/me.png')

# write rect in picture
cv2.rectangle(img, (100, 100), (200, 250), (0, 0, 255), 2)
cv2.putText(img, 'me', (100, 98), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
# 2 show
cv2.imshow('obama', img)

# 3 wait keyborad
cv2.waitKey(0)



# 4  destory windows
cv2.destroyAllWindows()

