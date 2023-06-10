from flask import Flask, render_template, request
import cv2

app = Flask(__name__)

# Define global variables for the video capture and writer
video_capture = None
video_writer = None

# Define a function to start the video capture
def start_video_capture():
    global video_capture, video_writer
    video_capture = cv2.VideoCapture(0)
    video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (640, 480))

# Define a function to stop the video capture
def stop_video_capture():
    global video_capture, video_writer
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    if video_writer is not None:
        video_writer.release()
        video_writer = None

# Define a route to display the video recording page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to start the video capture
@app.route('/start_capture', methods=['POST'])
def start_capture():
    start_video_capture()
    return 'Video capture started'

# Define a route to stop the video capture
@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    stop_video_capture()
    return 'Video capture stopped'

if __name__ == '__main__':
    app.run(debug=True)
