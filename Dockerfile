ARG BUILD_FROM
FROM $BUILD_FROM

# Install packages required to build and run the add-on
RUN \
  apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    python3-dev \
    jpeg-dev \
    zlib-dev \
    libffi-dev \
    cargo

# Copy the entire contents of the local rootfs directory
# to the root of the container's filesystem.
COPY blueprint_engine/rootfs/ /

# Install Python dependencies using the correct, absolute path
# to the requirements.txt file inside the container.
RUN pip3 install --no-cache-dir --break-system-packages -r /usr/bin/requirements.txt

# Copy and make run.sh executable
COPY run.sh /
RUN chmod a+x /run.sh

# The /share directory is mounted by Home Assistant, but creating
# it here ensures it exists.
RUN mkdir -p /share

# Command to run when the container starts
CMD [ "/run.sh" ]