import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import cv2
import numpy as np
import time

class ArucoFollower(Node):
    def __init__(self, target_id):
        super().__init__('aruco_follower')
        self.cmd_vel_pub = self.create_publisher(Twist, '/roombatron/cmd_vel', 10)

        self.camera_matrix = np.array([
            [2490, 0, 640],
            [0, 2490, 360],
            [0,    0,   1]
        ], dtype=float)
        self.dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=float)
        self.marker_length = 0.1  # Marker size in meters

        self.angular_error = None
        self.detected_id = None
        self.target_id = target_id  # The ArUco ID to track
        self.visit_count = {30: 0, 10: 0, 20: 0}  # Track visits for specific markers


        # LiDAR obstacle detection variables
        self.lidar_data = None
        self.lidar_obstacle_detected = False
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

    def lidar_callback(self, msg):
        if self.target_id == 1:  # Only process LiDAR data for target_id 1
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[250:290]  # Check only angles 250-290 degrees

            # Define a threshold distance for obstacles
            threshold_distance = 0.25  # meters
            self.lidar_obstacle_detected = np.any(valid_ranges < threshold_distance)

            if self.lidar_obstacle_detected:
                self.get_logger().warn("Obstacle detected within 250-290 degrees!")
                self.stop_robot()

    def capture_aruco(self):
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)

        if not cam.isOpened():
            self.get_logger().error("Unable to access the camera")
            return False

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)  # Process incoming LiDAR data

                ret, frame = cam.read()
                if not ret:
                    self.get_logger().error("No frame detected. Exiting...")
                    return False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, rejected = detector.detectMarkers(gray)
                self.get_logger().info(f"Detected markers: {ids}")

                if ids is not None:
                    for i in range(len(ids)):
                        if ids[i] == self.target_id:
                            obj_points = np.array([
                                [-self.marker_length / 2, self.marker_length / 2, 0],
                                [self.marker_length / 2, self.marker_length / 2, 0],
                                [self.marker_length / 2, -self.marker_length / 2, 0],
                                [-self.marker_length / 2, -self.marker_length / 2, 0]
                            ], dtype=np.float32)
                            img_points = corners[i][0].astype(np.float32)

                            ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.camera_matrix, self.dist_coeffs)
                            if ret:
                                marker_center = np.mean(corners[i][0], axis=0)
                                frame_center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
                                horizontal_pixel_distance = marker_center[0] - frame_center[0]
                                alpha = (horizontal_pixel_distance / frame.shape[1]) * 60  # degrees
                                delta = np.sqrt(tvec[0, 0]**2 + tvec[1, 0]**2)  # Euclidean distance from camera to marker

                                self.detected_id = ids[i]
                                self.angular_error = alpha
                                self.get_logger().info(f"Detected Target Marker ID: {self.detected_id}, Alpha: {alpha:.2f}, Distance: {delta:.2f} m")
                                stop_distance = 0.15

                                #Fail Safe Movement
                                if (self.target_id == 20 and self.detected_id == 10):
                                    self.get_logger().info("Target reached.")
                                    self.stop_robot()
                                    return True

                                if (self.target_id == 1 and self.detected_id == 30):
                                    self.get_logger().info("ArUco 1 is not in view at the moment. Moving to ArUco 10.")
                                    self.target_id = 10
                                    continue

                                if self.target_id == 30:
                                    self.visit_count[30] = 0 
                                    if self.detected_id == 10:
                                        self.visit_count[30] += 1
                                    if self.visit_count[30] == 2:
                                        self.get_logger().info("Robot was placed too close to ArUco 30. Starting ArUco 10.")
                                        self.stop_robot()
                                        return True 

                                if self.target_id == 10:
                                    self.visit_count[10] = 0
                                    if self.detected_id == 30:
                                        self.visit_count[10] += 1
                                    if self.visit_count[10] == 2:
                                        self.get_logger().info("Robot was placed too close to ArUco 10. Starting ArUco 30.")
                                        self.stop_robot()
                                        return True

                                if self.target_id == 1:
                                    if ((alpha < 3 or alpha > -3) and delta < 0.5):
                                        stop_distance = 0
                                        self.move_forward_continuously()

                                self.publish_command(delta, stop_distance)

                                if delta <= stop_distance:  # Stop condition
                                    self.get_logger().info("Target reached.")
                                    self.stop_robot()
                                    return True

                                break  # Stop searching once the target is found
                        else:
                            self.rotate_to_find_target()  # Rotate if the target ID is not found
                else:
                    self.get_logger().warn("No ArUco markers detected.")
                    self.rotate_to_find_target()  # Continue rotating if no markers are detected
        finally:
            cam.release()
    def publish_command(self, delta, stop_distance):
        if self.detected_id == self.target_id:
            if abs(self.angular_error) <= 1.5:
                if delta > stop_distance:
                    self.get_logger().info(f"Aligned with Marker ID {self.target_id}. Driving forward.")
                    self.move_forward_continuously()
                else:
                    self.get_logger().info("Stopping.")
                    self.stop_robot()
            else:
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = -0.01 * self.angular_error  # Proportional control
                self.cmd_vel_pub.publish(twist)
                self.get_logger().info(f"Adjusting angular error: {self.angular_error:.2f} degrees.")

    def rotate_to_find_target(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.2  # Rotational speed
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Rotating to find target...")

    def move_forward_continuously(self):
        twist = Twist()
        twist.linear.x = 0.2
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)

    # Track ArUco markers in sequence: 30 -> 10 -> 30 -> 10 -> 30
    for target_id in [30, 10, 30, 10, 30, 20, 1, 1]:
        node = ArucoFollower(target_id=target_id)
        try:
            success = node.capture_aruco()
            if success:
                node.get_logger().info(f"Finished tracking ArUco ID {target_id}.")
            else:
                node.get_logger().warn(f"Failed to track ArUco ID {target_id}. Moving to next.")
        finally:
            node.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()

