import cv2


def capture(output_img='result.jpg'):
    """
    To capture images from webcam and store them in the output directory
    press s to save
    press q to quit

    :param output_img: The file name of the output image (by default the image will be saved as 'result.jpg')
    :return: returns the img_new required to read and display images in cv2 and the location of the output directory
    """
    key = cv2.waitKey(1)
    out = './output/' + output_img
    webcam = cv2.VideoCapture(0)
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename=out, img=frame)
            webcam.release()
            img_new = cv2.imread(out, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("./output/saved_img.jpg", img_new)
            cv2.waitKey(4550)
            cv2.destroyAllWindows()
            print("Image saved!")

            return img_new, out

        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.\nProgram ended.")
            cv2.destroyAllWindows()
            break

def capture_delay(output_img='result.jpg', secs=156):
    """
    To capture images from webcam and store them in the output directory after a specific time interval

    :param secs: Time in seconds after which the image will be captured (here 1 sec = 31 value, so for 5 secs = 156 )
    :param output_img: The file name of the output image (by default the image will be saved as 'result.jpg')
    :return: returns the img_new required to read and display images in cv2 and the location of the output directory
    """
    out = './output/' + output_img
    webcam = cv2.VideoCapture(0)
    a = 0
    while True:
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        if a == secs:
            cv2.imwrite(filename=out, img=frame)
            webcam.release()
            img_new = cv2.imread(out, cv2.IMREAD_ANYCOLOR)
            cv2.imshow("./output/saved_img.jpg", img_new)
            cv2.waitKey(1550)
            cv2.destroyAllWindows()
            print("Image saved!")
            return img_new, out

        else:
            a += 1
            continue