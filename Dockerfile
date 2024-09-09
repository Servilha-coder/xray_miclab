# Use an official Python runtime as a parent image
FROM python:3.9-slim


RUN apt-get update && apt-get install -y git build-essential

RUN mkdir /code
COPY . /code/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
#CMD ["python", "main.py"]
