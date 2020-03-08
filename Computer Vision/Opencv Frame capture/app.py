import capture_frame
import cv2

#The first parameter is the name with which the output image will be saved
a, output = capture_frame.capture('My Image.jpg')
cv2.imshow(output, a)
cv2.waitKey(2550)
cv2.destroyAllWindows()

print("Program ran successfully!")