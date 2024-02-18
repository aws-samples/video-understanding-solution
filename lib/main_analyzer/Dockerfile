FROM python:3.12.2-alpine3.19
WORKDIR /lib/main_analyzer
ADD . /lib/main_analyzer
RUN addgroup -S vusgroup && adduser -S vususer -G vusgroup
USER vususer
RUN python3.12 -m pip install -r ./requirements.txt
CMD python3.12 index.py