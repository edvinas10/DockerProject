# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the app directory contents into the container at /app
COPY app /app

# Copy the data directory into the container at /data
COPY data /data

# Run the main.py script
CMD ["python", "main.py"]