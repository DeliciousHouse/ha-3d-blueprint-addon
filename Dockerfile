ARG BUILD_FROM
FROM ${BUILD_FROM}

# Set a default language environment
ENV LANG C.UTF-8

# Copy the entire 'rootfs' from your 'blueprint_engine' folder
# into the root of the Docker image. This correctly places your scripts.
COPY blueprint_engine/rootfs/ /

# Install Python, pip, and necessary build tools.
# Then, install all Python packages from your requirements.txt file.
# This is more efficient than installing numpy separately.
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    python3-dev && \
    pip3 install --no-cache-dir -r /usr/bin/requirements.txt

# The CMD line is removed as it's not needed. The Home Assistant
# Supervisor executes the run.sh from your repository root automatically.