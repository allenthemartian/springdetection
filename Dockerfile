FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="team-erc"

ENV WORKERS_PER_CORE=4 
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

# RUN mkdir /yolov8-fastapi

# COPY requirements.txt /yolov8-fastapi

# COPY . /yolov8-fastapi

# WORKDIR /yolov8-fastapi

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]