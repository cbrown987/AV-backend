FROM python:alpine3.19

# Set the working directory
WORKDIR .

# Copy the application code into the container
COPY . .

RUN pip install -r requirements.txt

# Set the default command to run when the container starts
CMD ["python", "app.py"]
