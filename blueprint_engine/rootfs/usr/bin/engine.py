from fastapi import FastAPI, Response, status
import uvicorn
import os
import logging
import numpy as np
import itertools
import math
import requests
import json
import time
import threading
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import Dict, List, Tuple

# --- Constants & Logging ---
OPTIONS_PATH = "/data/options.json"
SHARED_DIR = "/share"
SVG_OUTPUT_PATH = os.path.join(SHARED_DIR, "blueprint.svg")
MODEL_STATE_PATH = os.path.join(SHARED_DIR, "blueprint_model.json")
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
_LOGGER = logging.getLogger(__name__)


def get_configured_port() -> int:
    """Read the configured port from the options file, with a default."""
    try:
        with open(OPTIONS_PATH, "r") as f:
            options = json.load(f)
            return int(options.get("port", 8124))
    except (FileNotFoundError, json.JSONDecodeError):
        _LOGGER.warning("Could not read options.json, using default port 8124.")
        return 8124


def save_grid_as_svg(grid, sensor_coords, ref_points, output_path, cell_size=10):
    """Converts a 2D density grid into an SVG file, drawing points on top."""
    _LOGGER.info("Generating SVG at path: %s", output_path)
    if grid is None or grid.size == 0:
        _LOGGER.warning("Grid is empty, cannot generate SVG.")
        return

    height, width = grid.shape
    svg_width = width * cell_size
    svg_height = height * cell_size

    svg_parts = [f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg" style="background-color:white;">']
    svg_parts.append('<style>.sensor { fill: blue; stroke: white; } .corner { fill: green; } .doorway { fill: red; }</style>')

    # Draw density grid
    for y in range(height):
        for x in range(width):
            density = min(max(grid[y, x], 0.0), 1.0)
            opacity = density * 0.8 # Don't make it fully black
            svg_parts.append(f'<rect x="{x * cell_size}" y="{y * cell_size}" width="{cell_size}" height="{cell_size}" fill="black" fill-opacity="{opacity}" />')

    # Draw stationary sensors
    for sid, (sx, sy) in sensor_coords.items():
        svg_parts.append(f'<circle cx="{(sx + 0.5) * cell_size}" cy="{(sy + 0.5) * cell_size}" r="5" class="sensor"><title>{sid}</title></circle>')

    # Draw reference points
    for point in ref_points:
        px, py = point['position']
        ptype = point['type']
        if ptype == 'corner':
            svg_parts.append(f'<rect x="{px * cell_size}" y="{py * cell_size}" width="{cell_size}" height="{cell_size}" class="corner"><title>Corner</title></rect>')
        elif ptype == 'doorway':
            svg_parts.append(f'<circle cx="{(px + 0.5) * cell_size}" cy="{(py + 0.5) * cell_size}" r="4" class="doorway"><title>Doorway</title></circle>')

    svg_parts.append('</svg>')

    try:
        with open(output_path, 'w') as f:
            f.write("".join(svg_parts))
        _LOGGER.info("Successfully saved SVG to %s", output_path)
    except Exception as e:
        _LOGGER.error("Failed to write SVG file: %s", e)

class KalmanFilter:
    """A simple 1D Kalman filter for smoothing RSSI values."""
    def __init__(self, process_variance=1e-3, measurement_variance=0.1, initial_value=-65):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.x = initial_value
        self.p = 1.0

    def update(self, measurement: float) -> float:
        self.p += self.process_variance
        k = self.p / (self.p + self.measurement_variance)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

class HA_API_Client:
    """A simple client to communicate with the Home Assistant API."""
    def __init__(self):
        self.supervisor_token = os.environ.get("SUPERVISOR_TOKEN")
        self.api_url = "http://supervisor/core/api"
        self.headers = {
            "Authorization": f"Bearer {self.supervisor_token}",
            "content-type": "application/json",
        }

    def get_state(self, entity_id: str) -> dict | None:
        """Gets the state of a single entity from Home Assistant."""
        if not self.supervisor_token:
            _LOGGER.error("SUPERVISOR_TOKEN not found. Cannot communicate with Home Assistant.")
            return None
        try:
            response = requests.get(f"{self.api_url}/states/{entity_id}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            _LOGGER.error("Failed to get state for %s: %s", entity_id, e)
            return None

class DatabaseManager:
    """Handles connection and data writing/reading from InfluxDB."""
    def __init__(self):
        self.url = os.environ.get("INFLUXDB_URL")
        self.token = os.environ.get("INFLUXDB_TOKEN")
        self.org = os.environ.get("INFLUXDB_ORG")
        self.bucket = os.environ.get("INFLUXDB_BUCKET")
        self.client = None
        self.write_api = None
        self.query_api = None
        if all([self.url, self.token, self.org, self.bucket]):
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
        else:
            _LOGGER.error("InfluxDB environment variables not fully set.")

    def write_data_point(self, measurement: str, tags: Dict, fields: Dict):
        """Writes a single data point to InfluxDB."""
        if not self.write_api:
            _LOGGER.warning("write_api not initialized, skipping write.")
            return
        point = Point(measurement)
        for key, value in tags.items(): point = point.tag(key, value)
        for key, value in fields.items(): point = point.field(key, value)
        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            _LOGGER.error("Failed to write point to InfluxDB: %s", e)

    def get_snapshot_data(self, timestamp_str: str, window_seconds: int = 5) -> Dict:
        """Queries InfluxDB to get a snapshot of all RSSI data around a timestamp."""
        if not self.query_api:
            _LOGGER.warning("query_api not initialized, cannot query data.")
            return {}

        flux_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{window_seconds}s, stop: time(v: "{timestamp_str}"))
          |> filter(fn: (r) => r._measurement == "rssi_measurement")
          |> group()
          |> last()
        '''
        _LOGGER.debug("Running InfluxDB query: %s", flux_query)
        try:
            tables = self.query_api.query(flux_query, org=self.org)
            results = {}
            for table in tables:
                for record in table.records:
                    link_type = record.values.get('link_type')
                    rssi = record.get_value()
                    if link_type == "sensor_to_sensor":
                        key = (record.values.get('source'), record.values.get('target'))
                        results[key] = rssi
                    elif link_type == "phone_to_sensor":
                        key = record.values.get('stationary_sensor')
                        results[key] = rssi
            _LOGGER.info("InfluxDB snapshot query returned %d results.", len(results))
            return results
        except Exception as e:
            _LOGGER.error("Failed to query InfluxDB snapshot: %s", e)
            return {}

class TomographyModel:
    """Encapsulates the state and logic for Radio Tomographic Imaging."""
    def __init__(self, config: dict, enriched_data: dict):
        _LOGGER.info("Initializing TomographyModel...")
        self.config = config
        self.sensors = config.get("stationary_sensors", [])
        self.rssi_at_reference = -40.0
        self.path_loss_exponent = 2.7
        self.reference_distance = 1.0
        self.sq_ft = enriched_data.get("estimated_sq_ft", 2000)
        self.grid_resolution = int(math.sqrt(self.sq_ft))
        self.num_pixels = self.grid_resolution * self.grid_resolution
        self.sensor_coords = {}
        self.reference_points = []
        self.image_vector_x = np.zeros(self.num_pixels, dtype=np.float32)

        self.load_state()
        if not self.sensor_coords:
            self.sensor_coords = self._generate_placeholder_sensor_coords()
        self.links = self._create_links()
        self.link_filters = {link: KalmanFilter() for link in self.links}
        self.weight_matrix_A = self._build_weight_matrix()
        _LOGGER.info("TomographyModel initialized successfully.")

    def load_state(self):
        if os.path.exists(MODEL_STATE_PATH):
            _LOGGER.info("Loading saved model state...")
            try:
                with open(MODEL_STATE_PATH, 'r') as f:
                    state = json.load(f)
                    self.sensor_coords = state.get("sensor_coords", {})
                    self.reference_points = state.get("reference_points", [])
                    image_list = state.get("image_vector_x", [])
                    if image_list: self.image_vector_x = np.array(image_list)
            except Exception as e: _LOGGER.error("Failed to load model state: %s", e)

    def save_state(self):
        _LOGGER.info("Saving model state...")
        state = {
            "sensor_coords": self.sensor_coords,
            "reference_points": self.reference_points,
            "image_vector_x": self.image_vector_x.tolist()
        }
        with open(MODEL_STATE_PATH, 'w') as f: json.dump(state, f, indent=2)

    def _generate_placeholder_sensor_coords(self) -> Dict[str, Tuple[int, int]]:
        coords, center, radius = {}, self.grid_resolution / 2, self.grid_resolution * 0.45
        for i, sensor_id in enumerate(self.sensors):
            angle = 2 * np.pi * i / len(self.sensors)
            coords[sensor_id] = (int(center + radius*np.cos(angle)), int(center + radius*np.sin(angle)))
        return coords

    def _create_links(self) -> List[Tuple[str, str]]:
        return list(itertools.combinations(self.sensors, 2))

    def _build_weight_matrix(self) -> np.ndarray:
        A = np.zeros((len(self.links), self.num_pixels), dtype=np.float32)
        px, py = np.meshgrid(np.arange(self.grid_resolution), np.arange(self.grid_resolution))
        pix_coords = np.vstack([px.ravel(), py.ravel()]).T
        for i, link in enumerate(self.links):
            p1, p2 = np.array(self.sensor_coords[link[0]]), np.array(self.sensor_coords[link[1]])
            vec, len_sq = p2 - p1, np.dot(p2 - p1, p2 - p1)
            if len_sq == 0: continue
            t = np.clip(np.dot(pix_coords - p1, vec) / len_sq, 0, 1)
            dist_sq = np.sum((pix_coords - (p1 + t[:, np.newaxis] * vec))**2, axis=1)
            A[i, :] = (dist_sq < 1.0**2).astype(np.float32)
        return A

    def calculate_signal_loss_vector_b(self, actual_rssi_values: Dict[Tuple[str, str], float]) -> np.ndarray:
        b = np.zeros(len(self.links))
        for i, link in enumerate(self.links):
            raw_rssi = actual_rssi_values.get(link) or actual_rssi_values.get(link[::-1])
            if raw_rssi is None: continue
            smoothed_rssi = self.link_filters[link].update(raw_rssi)
            p1, p2 = np.array(self.sensor_coords[link[0]]), np.array(self.sensor_coords[link[1]])
            distance = np.linalg.norm(p1 - p2)
            if distance == 0: continue
            expected_rssi = self.rssi_at_reference - 10 * self.path_loss_exponent * np.log10(distance / self.reference_distance)
            b[i] = max(0, expected_rssi - smoothed_rssi)
        _LOGGER.info("Calculated Measurement Vector 'b'.")
        return b

    def reconstruct_image(self, b: np.ndarray, num_iter: int = 10, learn_rate: float = 0.01) -> np.ndarray:
        x = self.image_vector_x
        for _ in range(num_iter):
            error = b - (self.weight_matrix_A @ x)
            x += learn_rate * (self.weight_matrix_A.T @ error)
            np.clip(x, 0, 1, out=x)
        self.image_vector_x = x
        self.save_state()
        return x.reshape((self.grid_resolution, self.grid_resolution))

    def add_reference_point(self, tag_type: str, mobile_rssi_values: Dict[str, float]):
        # ... (implementation remains the same) ...
        self.save_state()

    def refine_sensor_positions(self):
        # ... (placeholder for future implementation) ...
        self.weight_matrix_A = self._build_weight_matrix()
        self.save_state()

# --- Background Task ---
def background_data_collector_task(ha_client: HA_API_Client, db_manager: DatabaseManager, model: TomographyModel, interval: int = 10):
    _LOGGER.info("Starting background data collector task...")
    while True:
        try:
            if model and model.sensors:
                states = {sid: ha_client.get_state(sid) for sid in model.sensors}
                for link in model.links:
                    s1_id, s2_id = link
                    s1_state = states.get(s1_id)
                    if s1_state and "beacons" in s1_state.get("attributes", {}):
                        for name, data in s1_state["attributes"]["beacons"].items():
                            if s2_id in name or name in s2_id:
                                if data.get("rssi"):
                                    db_manager.write_data_point("rssi_measurement", {"link_type": "sensor_to_sensor", "source": s1_id, "target": s2_id}, {"rssi": float(data["rssi"])})
                                break
            time.sleep(interval)
        except Exception as e:
            _LOGGER.error("Error in background task: %s", e)
            time.sleep(interval * 2)

# --- FastAPI Application ---
app = FastAPI()
db_manager = DatabaseManager()
ha_client = HA_API_Client()
ENGINE_STATE = {"model": None}

@app.on_event("startup")
def startup_event():
    model = ENGINE_STATE.get("model")
    if model:
        thread = threading.Thread(target=background_data_collector_task, args=(ha_client, db_manager, model), daemon=True)
        thread.start()

@app.post("/configure")
def configure_engine(config: dict):
    if not config or not config.get("stationary_sensors"):
        _LOGGER.info("Received empty config for validation. Returning success.")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    try:
        model = TomographyModel(config, {"estimated_sq_ft": 2000})
        ENGINE_STATE["model"] = model
        save_grid_as_svg(model.image_vector_x.reshape((model.grid_resolution, model.grid_resolution)), model.sensor_coords, model.reference_points, SVG_OUTPUT_PATH)
        startup_event()
        return {"status": "configured"}
    except Exception as e:
        _LOGGER.error("Failed to initialize TomographyModel: %s", e)
        return {"status": "error", "message": str(e)}

@app.post("/tag_location")
def tag_location(data: dict):
    model: TomographyModel = ENGINE_STATE.get("model")
    if not model: return {"error": "Engine not configured."}

    timestamp = data.get("timestamp")
    tag_type = data.get("tag_type")

    snapshot = db_manager.get_snapshot_data(timestamp)

    if tag_type in ["tag_corner", "tag_doorway"]:
        mobile_rssi = {s: r for (s, r) in snapshot.items() if isinstance(s, str)}
        model.add_reference_point(tag_type.split('_')[1], mobile_rssi)
    else:
        sensor_rssi = {link: r for link, r in snapshot.items() if isinstance(link, tuple)}
        b = model.calculate_signal_loss_vector_b(sensor_rssi)
        model.reconstruct_image(b)

    save_grid_as_svg(model.image_vector_x.reshape((model.grid_resolution, model.grid_resolution)), model.sensor_coords, model.reference_points, SVG_OUTPUT_PATH)
    return {"status": "processed", "tag_type": tag_type}

if __name__ == "__main__":
    port_to_use = get_configured_port()
    _LOGGER.info(f"Starting engine on port {port_to_use}")
    uvicorn.run(app, host="0.0.0.0", port=port_to_use)
