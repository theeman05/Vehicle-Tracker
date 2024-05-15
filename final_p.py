import os

import open3d as o3d
import numpy as np
import csv

CSV_OUTPUT_DIR = "perception_results/frame_{0}.csv"
CSV_OUTPUT_FORMAT = ["vehicle_id", "position_x", "position_y", "position_z", "mvec_x", "mvec_y", "mvec_z", "bbox_x_min",
                     "bbox_x_max", "bbox_y_min", "bbox_y_max", "bbox_z_min", "bbox_z_max"]

FOCUS_DIST_THRESHOLD = 70
MAX_FOCUS_HEIGHT = 3
START_DISPLACEMENT_THRESHOLD = 1.75
CLOSEST_THRESHOLD = 1.8

MAX_VEHICLE_POINTS = 125
MIN_VEHICLE_SQUARE = .35
MAX_VEHICLE_SQUARE = 6
MAX_VEHICLE_HEIGHT = 1.5
POSITIONS_TO_MONITOR = 5
VEHICLE_SIZE_PROPORTION = 1.75

INFER_TICK_DIFF = 3

HORIZONTAL_VECTOR = np.array([1, 1, 0])
ORIGIN = np.array([0, 0, 0])

VEHICLE_COLORS = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
]


class Vehicle:
    current_tick = 0

    def __init__(self, vehicle_id, position, bounding_box):
        self.vehicle_id = vehicle_id
        self.position = position
        self.bounding_box = bounding_box
        self.start_pos = position
        self.positions = []
        self.average_velocity = ORIGIN
        self.largest_bbox = bounding_box
        self.last_update_tick = Vehicle.current_tick

    def update(self, position, bounding_box):
        """
        Update a vehicle's position and bounding box.
        :param position: New position of the vehicle.
        :param bounding_box: New BBox of the vehicle.
        """
        self.position = position
        self.last_update_tick = Vehicle.current_tick

        # Compute largest bbox if all current bounds are greater than previous.
        if np.all(
                self.bounding_box.max_bound - self.bounding_box.min_bound > self.largest_bbox.max_bound - self.largest_bbox.min_bound):
            self.largest_bbox = bounding_box

        self.bounding_box = bounding_box
        self.positions.append(position * HORIZONTAL_VECTOR)  # Append position without the vertical component

        # Compute average velocity of the last POSITIONS_TO_MONITOR frames
        if len(self.positions) > 2:
            self.average_velocity = np.mean(np.diff(self.positions, axis=0), axis=0)
            if len(self.positions) > POSITIONS_TO_MONITOR:
                self.positions = self.positions[-POSITIONS_TO_MONITOR:]


def are_bounds_within_size(bounds):
    """
    Check if the bounds are within the MAX and MIN VEHICLE SQUARE.
    :param bounds: Bounds to check.
    :return: True if bounds are within. False o/w
    """
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    # Calculate dimensions excluding height (y dimension)
    dimensions = np.array([max_bound[0] - min_bound[0], max_bound[1] - min_bound[1]])
    # Check if both dimensions are within the specified size range
    return np.all((MIN_VEHICLE_SQUARE <= dimensions) & (dimensions <= MAX_VEHICLE_SQUARE)) and max_bound[2] - min_bound[
        2] < MAX_VEHICLE_HEIGHT


class VehicleTracker:
    def __init__(self):
        self.vehicles = {}
        self.verified_vehicles = {}
        self.next_vehicle_id = 0

    def update(self, position, bounding_box):
        """
        Update the current vehicle set given a position and bounding box.
        If a vehicle is not located, it is added and next_vehicle_id is incremented.
        :param position: Position of the vehicle.
        :param bounding_box: Bounding box of the vehicle.
        """
        closest_vehicle_id = self.get_closest_vehicle(position)
        if closest_vehicle_id is not None:
            vehicle = self.vehicles[closest_vehicle_id]

            # BBox must maintain a certain size proportion
            if np.all((bounding_box.max_bound - bounding_box.min_bound) / (
                    vehicle.largest_bbox.max_bound - vehicle.largest_bbox.min_bound) < VEHICLE_SIZE_PROPORTION):

                # Update vehicle and bounding box
                bounding_box.color = vehicle.bounding_box.color
                vehicle.update(position, bounding_box)

                # Ensure we have enough colors, the vehicle travelled enough distance, and is within the valid size
                if (len(self.verified_vehicles) < len(VEHICLE_COLORS) and
                        np.linalg.norm((vehicle.start_pos - vehicle.position) * HORIZONTAL_VECTOR) > START_DISPLACEMENT_THRESHOLD
                        and are_bounds_within_size(vehicle.bounding_box) and closest_vehicle_id not in self.verified_vehicles):
                    vehicle.bounding_box.color = VEHICLE_COLORS[len(self.verified_vehicles)]
                    self.verified_vehicles[closest_vehicle_id] = vehicle
        else:
            # New vehicle, create and add
            self.vehicles[self.next_vehicle_id] = Vehicle(self.next_vehicle_id, position, bounding_box)
            self.next_vehicle_id += 1

    def get_closest_vehicle(self, position):
        """
        Attempts to get the closest vehicle to the position as long as it's within the CLOSEST THRESHOLD.
        :param position: Position to compare other vehicle points to.
        :return: Closest vehicle ID or None.
        """
        closest_vehicle_id = None
        closest_distance = np.inf
        for vehicle_id, vehicle in self.vehicles.items():
            distance = np.linalg.norm((vehicle.position - position) * HORIZONTAL_VECTOR)
            if distance < closest_distance:
                closest_vehicle_id = vehicle_id
                closest_distance = distance

        if closest_distance < CLOSEST_THRESHOLD:
            return closest_vehicle_id
        elif closest_distance < CLOSEST_THRESHOLD * 4:
            vehicle = self.verified_vehicles.get(closest_vehicle_id)
            if vehicle and Vehicle.current_tick - vehicle.last_update_tick > INFER_TICK_DIFF:
                # print("Inferred", closest_vehicle_id)
                return closest_vehicle_id


def preprocess_pcd(pcd):
    """
    Preprocesses the pcd by removing points too far away, and flat planes.
    :param pcd: Pcd to process.
    :return: preprocessed pcd.
    """
    point_arr = np.array(pcd.points)
    # Compute distances from each point to the reference point
    distances = np.linalg.norm(point_arr - ORIGIN, axis=1)

    # Grab point heights
    heights = point_arr[:, 2]

    # Use a mask to extract the filtered point cloud
    pcd = pcd.select_by_index(np.where((distances <= FOCUS_DIST_THRESHOLD) & (heights <= MAX_FOCUS_HEIGHT))[0])

    # Remove the road
    _, inliers = pcd.segment_plane(distance_threshold=.2, ransac_n=10, num_iterations=1000)
    pcd = pcd.select_by_index(inliers, invert=True)

    return pcd


def scan_vehicles(processed_pcd, vehicle_tracker):
    """
    Scan for vehicles in the processed pcd and update the vehicle tracker with valid clusters.
    :param processed_pcd: Pcd to scan through.
    :param vehicle_tracker: Vehicle tracker to update.
    """
    # Convert the point cloud to a numpy array
    points = np.asarray(processed_pcd.points)
    labels = np.array(processed_pcd.cluster_dbscan(eps=1.9, min_points=2))

    # Compute a bounding box for each cluster
    for i in range(labels.max() + 1):
        cluster_points = points[labels == i]
        if len(cluster_points) < MAX_VEHICLE_POINTS:
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=cluster_points.min(axis=0),
                                                                   max_bound=cluster_points.max(axis=0))
            vehicle_tracker.update(np.mean(cluster_points, axis=0), bounding_box)


def color_points_within_bounding_box(pcd, vis, bounding_box):
    """
    Colors all points within a given bounding box, and includes the bounding box, and the relative pcd.
    :param pcd: Pcd to grab points from
    :param vis: Visualizer to add points to
    :param bounding_box: Bounding box of the object.
    """
    inside_pcd = pcd.select_by_index(bounding_box.get_point_indices_within_bounding_box(pcd.points))
    for i in range(len(inside_pcd.points)):
        inside_pcd.colors.append(bounding_box.color)
        # If we have 1 point, visualize the BBox
        if i == 0:
            vis.add_geometry(bounding_box)
    vis.add_geometry(inside_pcd)


def vehicle_to_csv_data(vehicle):
    """
    Converts a vehicle into a list, so it can be written to a csv.
    :param vehicle: Vehicle to convert
    :return: List of data to be written.
    """
    min_bound = vehicle.bounding_box.min_bound
    max_bound = vehicle.bounding_box.max_bound

    return [vehicle.vehicle_id, vehicle.position[0], vehicle.position[1], vehicle.position[2], vehicle.average_velocity[0],
            vehicle.average_velocity[1], vehicle.average_velocity[2], min_bound[0],
            max_bound[0], min_bound[1], max_bound[1], min_bound[2], max_bound[2]]


def main():
    pcd_folder = 'dataset/PointClouds'  # Folder where new PCD files are saved
    vehicle_tracker = VehicleTracker()
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    placeholder_vehicle = Vehicle(-1, ORIGIN, o3d.geometry.AxisAlignedBoundingBox(min_bound=ORIGIN, max_bound=ORIGIN))

    for i in range(len(os.listdir(pcd_folder))):
        Vehicle.current_tick = i
        vis.clear_geometries()
        pcd = o3d.io.read_point_cloud(f"{pcd_folder}/{i}.pcd")
        pcd = preprocess_pcd(pcd)
        scan_vehicles(pcd, vehicle_tracker)

        with open(CSV_OUTPUT_DIR.format(i), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(CSV_OUTPUT_FORMAT)

            # Write actual vehicles
            for vehicle in vehicle_tracker.verified_vehicles.values():
                csv_writer.writerow(vehicle_to_csv_data(vehicle))
                color_points_within_bounding_box(pcd, vis, vehicle.bounding_box)

            # Write placeholder vehicles
            for _ in range(len(VEHICLE_COLORS) - len(vehicle_tracker.verified_vehicles)):
                csv_writer.writerow(vehicle_to_csv_data(placeholder_vehicle))

            # Update the visualization window
            vis.poll_events()
            vis.update_renderer()


if __name__ == '__main__':
    main()
