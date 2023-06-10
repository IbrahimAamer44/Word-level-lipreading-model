# This Docker file creates an image to run Flask Demonstration App of lip-reading

From python:3.9.13

RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /docker-lipsol-flask

ADD . /docker-lipsol-flask

# Install the Python dependencies
RUN pip install -r requirements.txt

#These commands install the cv2 dependencies that are normally present on
#the local machine, but due missing in Docker container causing the issue.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Setting the entry point command to run the flask application
CMD ["python","FlaskApp/app.py"]