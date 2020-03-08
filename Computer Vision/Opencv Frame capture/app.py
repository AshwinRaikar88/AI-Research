import capture_frame
import cv2

a, output = capture_frame.capture_delay('My Image.jpg', 120)
cv2.imshow(output, a)
cv2.waitKey(2550)
cv2.destroyAllWindows()

print("Program ran successfully!")