ARG BUILD_FROM
FROM $BUILD_FROM

# Install required packages
RUN \
  apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-requests \
    py3-scipy

# Set working directory
WORKDIR /usr/src/app

# Copy requirements first for better caching
COPY rootfs/usr/bin/requirements.txt /usr/src/app/
RUN pip3 install --no-cache-dir -r /usr/src/app/requirements.txt

# Copy files to the right locations
COPY rootfs/ /

# Copy and make run.sh executable
COPY run.sh /
RUN chmod a+x /run.sh

# Create shared directory
RUN mkdir -p /share

# Command to run when the container starts
CMD [ "/run.sh" ]