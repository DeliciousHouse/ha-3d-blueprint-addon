#!/usr/bin/with-contenv bashio
# ==============================================================================
# Home Assistant Community Add-on: Blueprint Engine
#
# This script is executed when the add-on is started.
# ==============================================================================

bashio::log.info "Starting the 3D Blueprint Engine..."

# Export standard config
export LOG_LEVEL=$(bashio::config 'log_level')
bashio::log.info "Log level is set to: ${LOG_LEVEL}"

# Export InfluxDB connection details as environment variables
export INFLUXDB_URL=$(bashio::config 'influxdb_url')
export INFLUXDB_TOKEN=$(bashio::config 'influxdb_token')
export INFLUXDB_ORG=$(bashio::config 'influxdb_org')
export INFLUXDB_BUCKET=$(bashio::config 'influxdb_bucket')

# Check if the variables are set
if [[ -z "$INFLUXDB_URL" || -z "$INFLUXDB_TOKEN" ]]; then
    bashio::log.fatal "InfluxDB connection details are not set. Please configure the add-on."
    exit 1
fi

bashio::log.info "InfluxDB URL set to: ${INFLUXDB_URL}"
bashio::log.info "InfluxDB Org set to: ${INFLUXDB_ORG}"
bashio::log.info "InfluxDB Bucket set to: ${INFLUXDB_BUCKET}"

# Start the Python application
python3 -u /usr/bin/engine.py

bashio::log.info "3D Blueprint Engine has stopped."
