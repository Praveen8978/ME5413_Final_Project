#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import tf
#import tf2_ros
import easyocr
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Int32MultiArray
from tf.transformations import euler_from_quaternion


class Visual:
    def __init__(self, rate=10):
        rospy.loginfo(f" Running rate: {rate}")
        self.detect_mode = "number"
        self.bridge = CvBridge()
        self.rate = rate
        self.detected_numbers_positions = {}
        self.distance_threshold = 0.8  #TODO: Adjust this if doesn't work properly

        self.img_curr = None
        self.img_curr_gray = None
        self.num_detect_result = [0] * 10 # Using this as count for number of times a number is seen
        self.camera_info = rospy.wait_for_message("/front/camera_info", CameraInfo)
        self.intrinsic = np.array(self.camera_info.K).reshape(3, 3)
        self.projection = np.array(self.camera_info.P).reshape(3, 4)
        self.distortion = np.array(self.camera_info.D)
        self.img_frame = self.camera_info.header.frame_id
        self.ocr_detector = easyocr.Reader(["en"], gpu=True)

        #Publishers
        self.target_pose_pub = rospy.Publisher("/percep/pose", PoseStamped, queue_size=1)
        self.number_database = rospy.Publisher("/percep/numberData", Int32MultiArray, queue_size=1)

        #Subscribers
        self.tf_sub = tf.TransformListener()
        self.scansub = rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
        self.img_sub = rospy.Subscriber("/front/image_raw", Image, self.img_callback)
        self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback) 

        rospy.loginfo("visual node initialized")

    def odom_callback(self, msg):
        self.curr_odom = msg

    def img_callback(self, msg: Image):
        self.img_curr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def scan_callback(self, msg: LaserScan):
        self.scan_curr = msg.ranges
        self.scan_params_curr = [msg.angle_min, msg.angle_max, msg.angle_increment]

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.img_curr is None:
                continue

            result = self.ocr_detector.readtext(self.img_curr, batch_size=2, allowlist="0123456789")
            img_show = self.img_curr.copy()

            for detection in result:
                    # detection[0]: the bounding box of the detected text
                    # detection[1]: the detected text
                    # detection[2]: the confidence of the detected text
                if len(detection[1]) > 1:  # not a single digit
                    continue
                if detection[2] < 0.99:
                    continue

                center = [(x + y) / 2 for x, y in zip(detection[0][0], detection[0][2])]
                center_x = int(center[0])
                center_y = int(center[1])

                diag_vec = np.array(detection[0][2]) - np.array(detection[0][0])
                diag_len = np.linalg.norm(diag_vec)

                direction = np.array([[center[0]], [center[1]], [1]])
                direction = np.dot(np.linalg.inv(self.intrinsic), direction)

                p_in_cam = PoseStamped()
                p_in_cam.header.frame_id = self.img_frame
                p_in_cam.pose.position.x = direction[0].item()
                p_in_cam.pose.position.y = direction[1].item()
                p_in_cam.pose.position.z = direction[2].item()
                p_in_cam.pose.orientation.w = 1

                self.tf_sub.waitForTransform("odom", self.img_frame, rospy.Time.now(), rospy.Duration(1))
                transformed_pose = self.tf_sub.transformPose("odom", p_in_cam)

                x = transformed_pose.pose.position.x
                y = transformed_pose.pose.position.y

                if detection[1].isdigit() and 0 <= int(detection[1]) <= 9:
                    number = int(detection[1])
                
                if number not in self.detected_numbers_positions:
                    self.detected_numbers_positions[number] = []
                
                already_seen = False
                for pos in self.detected_numbers_positions[number]:
                    dist = np.linalg.norm([pos[0] - x, pos[1] - y])
                    if dist < self.distance_threshold:
                        already_seen = True
                        break

                if already_seen:
                    rospy.loginfo(f"Number at ({x:.2f}, {y:.2f}) already detected. Skipping.")
                    continue
                
                self.detected_numbers_positions[number].append((x, y))
                rospy.loginfo(f"New number detected at ({x:.2f}, {y:.2f}), processing...")

                self.num_detect_result[number] += 1
                rospy.loginfo(f"Detected number {number}, count: {self.num_detect_result[number]}")

                number_msg = Int32MultiArray()
                number_msg.data = self.num_detect_result
                self.number_database.publish(number_msg)
                rospy.loginfo(" Published the updated number_database")
                
                self.target_pose_pub.publish(transformed_pose)
                rate.sleep()

            
if __name__ == "__main__":
    rospy.init_node("visual")
    v = Visual(rate=30)
    try:
        v.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down visual node.")








