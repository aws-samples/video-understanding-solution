FROM --platform=linux/amd64 python:3.12
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN addgroup --system vusgroup && adduser --system vususer --ingroup vusgroup --home /vus
USER vususer
WORKDIR /lib/main_analyzer
ADD . /lib/main_analyzer
RUN python3.12 -m pip install -r ./requirements.txt
CMD python3.12 index.py