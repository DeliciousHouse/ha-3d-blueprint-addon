FROM homeassistant/base-python:3.12

# Set up our working directory
WORKDIR /usr/src/app

# Copy the requirements file and install dependencies
COPY blueprint_engine/rootfs/usr/bin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our Python engine code into the container
COPY blueprint_engine/rootfs/usr/bin/engine.py .

# Expose the internal API port
EXPOSE 8124

CMD [ "python3", "-u", "./engine.py" ]
