# Using OpenCV Python to capture images from a webcam
You can use Opencv python to capture a frame from the webcam and save it into the system.

## Installation

### Requirements

  * Python 3.6+
  * OpenCV 3.4+ 
  ```bash
  pip install opencv-python
  ```
  
## Usage

### Default usage
Run app.py on the terminal
```bash
python3 app.py
```
* To capture an image press 's'
* If you want to quit, just press 'q'

The image will be saved in output directory.

### Capture image after a specified time delay
Run app_delay.py on the terminal
```bash
python3 app_delay.py
```

The image will be captured after a time as specified in the code
 * The first parameter is the name with which the output image will be saved
 * The second parameter is the time in seconds for delay (here 1sec = 31, So for 4sec = 120)
```python
a, output = capture_frame.capture_delay('My Image.jpg', 120)
...
```

The image will be saved in output directory.

## Further improvements
*  New funtionalities will be added
*  This program was successfully tested on windows 
*  Feel free to modify and test on other OS
