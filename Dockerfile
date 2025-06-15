ARG BUILD_FROM
FROM $BUILD_FROM

# Install requirements for add-on
RUN \
  apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy

# Copy data for add-on
COPY run.sh /
RUN chmod a+x /run.sh

# Copy Python requirements and install
COPY blueprint_engine/rootfs/usr/bin/requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy the engine script
COPY blueprint_engine/rootfs/usr/bin/engine.py /usr/bin/

CMD [ "/run.sh" ]
