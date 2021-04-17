FROM python:3.8-slim

#FROM nickgryg/alpine-pandas:3.9.1

# FROM ubuntu

WORKDIR /birdsongs

COPY requirements.txt .

RUN apt-get update
RUN apt-get -y install libsndfile1 ffmpeg libavcodec-extra

# RUN apk add alsa-lib-dev

# RUN pip install python-dateutil pytz>=2011k
# RUN pip --no-deps pandas
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

WORKDIR ./app
CMD ["python", "./app.py"]
# CMD ["gunicorn" "app.app:server"]
