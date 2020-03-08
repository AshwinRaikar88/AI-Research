import capture_frame
import cv2

#The first parameter is the name with which the output image will be saved
#The second parameter is the time in seconds for delay (here 1sec = 31, So for 4sec = 120)
a, output = capture_frame.capture_delay('My Image.jpg', 120)
cv2.imshow(output, a)
cv2.waitKey(2550)
cv2.destroyAllWindows()

print("Program ran successfully!")
