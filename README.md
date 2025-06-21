# HA 3D Blueprint Add-on

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Home Assistant add-on that provides the processing engine for the HA&nbsp;3D&nbsp;Blueprint&nbsp;system. This add-on performs the heavy mathematical calculations required for Radio Tomographic Imaging (RTI) to generate a 2D/3D model of your home's layout using Bluetooth RSSI values.

**This project is under active development and is considered experimental.**

---

## Overview

The Blueprint Engine is the computational backend for the HA&nbsp;3D&nbsp;Blueprint&nbsp;system. It:

- Collects and processes RSSI data from your Bluetooth sensors
- Performs Radio Tomographic Imaging calculations
- Generates SVG blueprints of your home layout
- Provides a REST API for the Home Assistant integration to communicate with

This add-on works in conjunction with the [HA 3D Blueprint Integration](https://github.com/delicioushouse/ha-3d-blueprint-integration) to provide a complete home mapping solution.

## Prerequisites

Before installing this add-on, ensure you have:

1. **InfluxDB Add-on:** The official [InfluxDB Home Assistant Add-on](https://github.com/home-assistant/addons/blob/master/influxdb/DOCS.md) installed and configured
2. **Bluetooth Sensors:** At least 4-5 stationary Bluetooth-enabled devices (e.g., ESPresense proxies)
3. **Mobile Beacon:** A Bluetooth beacon representing your mobile device

## Installation

1. Navigate to **Settings > Add-ons > Add-on Store** in Home Assistant
2. Click the three-dots menu and select **"Repositories"**
3. Add this repository URL:
   ```
   https://github.com/delicioushouse/ha-3d-blueprint-addon
   ```
4. Install the "Blueprint Engine" add-on
5. Configure the InfluxDB connection details in the Configuration tab
6. Start the add-on

## Configuration

The add-on requires the following configuration parameters:

```yaml
log_level: info
influxdb_url: "http://a0d7b954-influxdb:8086"
influxdb_token: "your-influxdb-token"
influxdb_org: "your-org-name"
influxdb_bucket: "blueprint-data"
```

### Configuration Parameters

- **log_level**: Set the logging level (trace, debug, info, notice, warning, error, fatal)
- **influxdb_url**: URL of your InfluxDB instance
- **influxdb_token**: Authentication token for InfluxDB
- **influxdb_org**: Organization name in InfluxDB
- **influxdb_bucket**: Bucket name for storing RSSI data

## API Endpoints

The add-on exposes the following REST API endpoints on port 8124:

- `POST /configure`: Configure the tomography model with sensor information
- `POST /tag_location`: Process location tagging events for building the blueprint

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/delicioushouse/ha-3d-blueprint-addon/issues) page.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
