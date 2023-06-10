From python:3.9.13

RUN pip install --upgrade pip

WORKDIR /docker-lipsol-flask

ADD . /docker-lipsol-flask

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#These commands install the cv2 dependencies that are normally present on
#the local machine, but due missing in Docker container causing the issue.

CMD ["python","app.py"]