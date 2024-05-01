FROM pytorch/pytorch:latest

# Install additional dependencies
RUN pip install -r requirements.txt

# Set the working directory
WORKDIR .

# Copy the application code into the container
COPY . .

# Set the default command to run when the container starts
CMD ["python", "app.py"]