FROM python:alpine3.7
RUN mkdir /app
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.5.0 tensorflow-gpu==2.5.0
EXPOSE 5000
CMD python ./app.py