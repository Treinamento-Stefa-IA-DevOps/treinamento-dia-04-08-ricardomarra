#FROM python:3.8-slim
FROM centos/python-36-centos7

COPY . /app
COPY model /model
COPY requirements /requirements

WORKDIR /app

RUN pip install -r requirements

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "80", "--reload"]
