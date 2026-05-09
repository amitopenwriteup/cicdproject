FROM python:3.9-alpine

WORKDIR /flask_app

COPY app/tests/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pytest

EXPOSE 8080

COPY app/ .

COPY tests/ app/tests/

CMD [ "python", "app.py" ]
