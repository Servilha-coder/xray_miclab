# Base image with Orthanc and plugins
FROM jodogne/orthanc-plugins

# Update and install Python3, pip, and virtualenv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the project files into the container
COPY . /app

# Install Python dependencies in the virtual environment
RUN pip install -r requirements.txt

# Expose the necessary port for Orthanc
EXPOSE 8042

# Command to start Orthanc
CMD ["orthanc", "/etc/orthanc/"]
