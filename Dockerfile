
FROM python:3.8


WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "server.py"]

