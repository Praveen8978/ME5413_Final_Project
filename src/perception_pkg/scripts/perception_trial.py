#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import tf
import easyocr
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Int32MultiArray
from tf.transformations import euler_from_quaternion
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visual:
    def __init__(self, rate=10):
        rospy.loginfo(f" Running rate: {rate}")
        self.detect_mode = "number"
        self.bridge = CvBridge()
        self.rate = rate
        
        # Store all detected number positions for clustering
        self.all_detected_positions = []
        self.all_detected_numbers = []
        self.detection_timestamps = []  # To track when detections were made
        self.cube_clusters = {}  # Dictionary to store cube clusters and their associated numbers
        
        # Detection parameters
        self.distance_threshold = 1.5  # Increased from 0.8 to handle larger separation
        self.clustering_eps = 1.2  # Increased clustering distance threshold to capture same cube detections
        self.min_samples = 1  # Reduced to 1 to allow singleton clusters initially
        self.cluster_update_frequency = 1  # Update clusters after each new detection
        self.detection_count = 0
        self.last_clustering_time = rospy.Time.now()
        self.clustering_cooldown = rospy.Duration(2.0)  # Wait at least 2 seconds between clustering operations
        
        # For confidence in detection
        self.confidence_threshold = 0.99
        
        # Initialize image and other data
        self.img_curr = None
        self.img_curr_gray = None
        self.num_detect_result = [0] * 10  # Final count of each number (0-9)
        
        # Camera configuration
        self.camera_info = rospy.wait_for_message("/front/camera_info", CameraInfo)
        self.intrinsic = np.array(self.camera_info.K).reshape(3, 3)
        self.projection = np.array(self.camera_info.P).reshape(3, 4)
        self.distortion = np.array(self.camera_info.D)
        self.img_frame = self.camera_info.header.frame_id
        
        # OCR configuration
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)
        
        # Publishers
        self.target_pose_pub = rospy.Publisher("/percep/pose", PoseStamped, queue_size=1)
        self.number_database = rospy.Publisher("/percep/numberData", Int32MultiArray, queue_size=1)
        
        # Subscribers
        self.tf_sub = tf.TransformListener()
        self.scansub = rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
        self.img_sub = rospy.Subscriber("/front/image_raw", Image, self.img_callback)
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        
        # Debug visualization flag
        self.enable_debug_visualization = False
        
        rospy.loginfo("Visual node initialized with improved cube detection")

    def odom_callback(self, msg):
        self.curr_odom = msg

    def img_callback(self, msg: Image):
        self.img_curr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def scan_callback(self, msg: LaserScan):
        self.scan_curr = msg.ranges
        self.scan_params_curr = [msg.angle_min, msg.angle_max, msg.angle_increment]
    
    def visualize_clusters(self):
        """
        Debug function to visualize the clusters
        """
        if not self.enable_debug_visualization or len(self.all_detected_positions) < 2:
            return
            
        try:
            positions_array = np.array(self.all_detected_positions)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract x, y coordinates
            x = positions_array[:, 0]
            y = positions_array[:, 1]
            z = np.zeros_like(x)  # Use zero for z since we're working in 2D
            
            # Plot the detected points
            scatter = ax.scatter(x, y, z, c=self.all_detected_numbers, cmap='tab10', 
                                 s=100, alpha=0.8, edgecolors='k')
            
            # Add labels for the points showing the detected number
            for i, (x_val, y_val) in enumerate(zip(x, y)):
                ax.text(x_val, y_val, 0.1, str(self.all_detected_numbers[i]), 
                        color='red', fontsize=12)
            
            # Plot lines between points that are likely to be on the same cube
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    if dist < self.clustering_eps:
                        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'k--', alpha=0.3)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Z')
            ax.set_title('Detected Numbers and Clustering')
            
            plt.colorbar(scatter, label='Detected Number')
            plt.savefig('/tmp/cluster_visualization.png')
            plt.close()
            rospy.loginfo("Saved cluster visualization to /tmp/cluster_visualization.png")
        except Exception as e:
            rospy.logwarn(f"Visualization error: {e}")
    
    def check_duplicate_detection(self, x, y, number):
        """
        Check if a number at a position is likely to be a duplicate detection
        """
        for i, (pos_x, pos_y) in enumerate(self.all_detected_positions):
            # Check if same number and close position
            dist = np.sqrt((pos_x - x)**2 + (pos_y - y)**2)
            if dist < self.distance_threshold and self.all_detected_numbers[i] == number:
                rospy.loginfo(f"Duplicate detection filtered: Number {number} at ({x:.2f}, {y:.2f}), " +
                             f"similar to existing detection at ({pos_x:.2f}, {pos_y:.2f}), distance: {dist:.2f}m")
                return True
        return False
    
    def perform_clustering(self):
        """
        Cluster the detected numbers to identify individual cubes
        """
        if len(self.all_detected_positions) < 1:
            return
            
        current_time = rospy.Time.now()
        if (current_time - self.last_clustering_time) < self.clustering_cooldown:
            return
            
        self.last_clustering_time = current_time
        
        # Convert positions to numpy array for clustering
        positions_array = np.array(self.all_detected_positions)
        
        # Use DBSCAN for spatial clustering of detected numbers
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples).fit(positions_array)
        
        # Get labels for each point (position)
        labels = clustering.labels_
        
        # Group numbers by cluster
        cube_clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points get their own clusters
                # Create a new singleton cluster
                max_label = max(labels) if len(labels) > 0 and max(labels) >= 0 else -1
                new_label = max_label + 1
                if new_label not in cube_clusters:
                    cube_clusters[new_label] = []
                cube_clusters[new_label].append((self.all_detected_numbers[i], i))
            else:
                if label not in cube_clusters:
                    cube_clusters[label] = []
                cube_clusters[label].append((self.all_detected_numbers[i], i))
        
        # Log the clusters
        for cluster_id, number_idx_pairs in cube_clusters.items():
            numbers = [pair[0] for pair in number_idx_pairs]
            positions = [self.all_detected_positions[pair[1]] for pair in number_idx_pairs]
            
            rospy.loginfo(f"Cube {cluster_id}: Contains numbers {numbers}")
            rospy.loginfo(f"Cube {cluster_id}: At positions {positions}")
        
        # Update number counts based on unique numbers per cube
        self.num_detect_result = [0] * 10
        
        for cluster_id, number_idx_pairs in cube_clusters.items():
            numbers = [pair[0] for pair in number_idx_pairs]
            unique_numbers = set(numbers)
            
            for number in unique_numbers:
                self.num_detect_result[number] += 1
        
        rospy.loginfo(f"Updated number counts: {self.num_detect_result}")
        
        # Publish updated number count
        number_msg = Int32MultiArray()
        number_msg.data = self.num_detect_result
        self.number_database.publish(number_msg)
        
        # Optional visualization
        self.visualize_clusters()

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.img_curr is None:
                rate.sleep()
                continue

            result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
            
            detection_in_current_frame = False
            
            for detection in result:
                # Validate detection
                if len(detection[1]) > 1 or not detection[1].isdigit():  # Not a single digit
                    continue
                if detection[2] < self.confidence_threshold:
                    continue

                # Calculate center of bounding box
                center = [(x + y) / 2 for x, y in zip(detection[0][0], detection[0][2])]
                center_x = int(center[0])
                center_y = int(center[1])

                # Calculate direction in camera frame
                direction = np.array([[center[0]], [center[1]], [1]])
                direction = np.dot(np.linalg.inv(self.intrinsic), direction)

                # Create pose in camera frame
                p_in_cam = PoseStamped()
                p_in_cam.header.frame_id = self.img_frame
                p_in_cam.header.stamp = rospy.Time.now()
                p_in_cam.pose.position.x = direction[0].item()
                p_in_cam.pose.position.y = direction[1].item()
                p_in_cam.pose.position.z = direction[2].item()
                p_in_cam.pose.orientation.w = 1

                # Transform to odom frame
                try:
                    self.tf_sub.waitForTransform("odom", self.img_frame, p_in_cam.header.stamp, rospy.Duration(1))
                    transformed_pose = self.tf_sub.transformPose("odom", p_in_cam)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logwarn(f"Transform error: {e}")
                    continue

                x = transformed_pose.pose.position.x
                y = transformed_pose.pose.position.y

                # Process the detected number
                if 0 <= int(detection[1]) <= 9:
                    number = int(detection[1])
                    
                    # Check if this is a duplicate detection
                    if self.check_duplicate_detection(x, y, number):
                        continue
                    
                    # Log the new detection
                    rospy.loginfo(f"Detected number {number} at position ({x:.2f}, {y:.2f})")
                    
                    # Add to our detection database
                    self.all_detected_positions.append((x, y))
                    self.all_detected_numbers.append(number)
                    self.detection_timestamps.append(rospy.Time.now())
                    self.detection_count += 1
                    detection_in_current_frame = True
                    
                    # Publish the pose for visualization or navigation
                    self.target_pose_pub.publish(transformed_pose)
            
            # Perform clustering if we had a new detection
            if detection_in_current_frame:
                self.perform_clustering()
                self.detection_count = 0
            
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("visual")
    v = Visual(rate=30)
    try:
        v.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down visual node.")