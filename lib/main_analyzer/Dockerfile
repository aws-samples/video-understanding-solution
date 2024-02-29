FROM python:3.12
#RUN apk update
#RUN apk upgrade
#RUN apk add --no-cache ffmpeg libsm6 libxext6
#RUN yum update && yum install ffmpeg libsm6 libxext6  -y

#FROM python:3.12.2-alpine3.19
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN addgroup --system vusgroup && adduser --system vususer --ingroup vusgroup --home /vus
USER vususer
WORKDIR /lib/main_analyzer
ADD . /lib/main_analyzer
RUN python3.12 -m pip install -r ./requirements.txt
CMD python3.12 index.py
