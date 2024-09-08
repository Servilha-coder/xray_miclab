# Base image with Orthanc and plugins
FROM jodogne/orthanc-plugins

# Update and install Python3 and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Expose the necessary port for Orthanc
EXPOSE 8042

# Command to start Orthanc
CMD ["orthanc", "/etc/orthanc/"]
