FROM python:alpine3.19

# Set the working directory
WORKDIR .

# Copy the application code into the container
COPY . .

RUN pip install -r requirements.txt

CMD ["gunicorn"  , "-b", "0.0.0.0:80", "app:app"]