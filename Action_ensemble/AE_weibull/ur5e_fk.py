"""
This forward computation is based on the base_link
the result is the end-effector's pose, not the wrist_3_link
"""
import numpy as np

from math import cos as cos
from math import sin as sin
from math import sqrt as sqrt
from geometry_msgs.msg import Pose, Quaternion
import tf.transformations as tf
from scipy.spatial.transform import Rotation

global mat

# UR5e dh parameters
# This can be find here:
# https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics/
# or in the ur_e_description package
d1 =  0.163
a2 = -0.425
a3 = -0.392
d4 =  0.127
d5 =  0.1
d6 =  0.1

a = [0, -0.425, -0.392, 0, 0, 0]
d = [0.163, 0, 0, 0.127, 0.1, 0.1]
alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

def matrix2ros(np_pose):
    """Transform pose from 4*4 matrix format to ROS Pose format.

    Args:
        np_pose: A pose in 4*4 matrix format

    Returns:
        ros pose: vector(1, 3), quaternion (1, 4)
    """

    # ROS pose
    ros_pose = Pose()

    # ROS position
    ros_pose.position.x = np_pose[0, 3]
    ros_pose.position.y = np_pose[1, 3]
    ros_pose.position.z = np_pose[2, 3]

    # ROS orientation
    np_q = tf.quaternion_from_matrix(np_pose)
    ros_pose.orientation.x = np_q[0]
    ros_pose.orientation.y = np_q[1]
    ros_pose.orientation.z = np_q[2]
    ros_pose.orientation.w = np_q[3]

    return ros_pose

def euler2ros(ur_pose):
    """Transform pose from Euler format to ROS Pose format.

    Args:
        ur_pose: A pose in Euler format [px, py, pz, rx, ry, rz]
        (type: list)

    Returns:
        ros pose: vector(1, 3), quaternion (1, 4)
    """

    # ROS pose
    ros_pose = Pose()

    # ROS position
    ros_pose.position.x = ur_pose[0]
    ros_pose.position.y = ur_pose[1]
    ros_pose.position.z = ur_pose[2]

    # Ros orientation
    angle = sqrt(ur_pose[3] ** 2 + ur_pose[4] ** 2 + ur_pose[5] ** 2)
    direction = [i / angle for i in ur_pose[3:6]]
    np_T = tf.rotation_matrix(angle, direction)
    np_q = tf.quaternion_from_matrix(np_T)
    ros_pose.orientation.x = np_q[0]
    ros_pose.orientation.y = np_q[1]
    ros_pose.orientation.z = np_q[2]
    ros_pose.orientation.w = np_q[3]

    return ros_pose


def ros2matrix(ros_pose):
    """Transform pose from ROS Pose format to 4*4 matrix format.

    Args:
        ros_pose: A pose in ROS Pose format (type: Pose)

    Returns:
        4*4 matrix np.array.
    """

    # orientation
    np_pose = tf.quaternion_matrix([ros_pose.orientation.x, ros_pose.orientation.y, \
                                    ros_pose.orientation.z, ros_pose.orientation.w])

    # position
    np_pose[0][3] = ros_pose.position.x
    np_pose[1][3] = ros_pose.position.y
    np_pose[2][3] = ros_pose.position.z

    return np_pose

def ros2euler(ros_pose):
    """Transform pose from ROS Pose format to euler format.

    Args:
        ros_pose: A pose in ROS Pose format (type: Pose)

    Returns:
        1*6 vector.
    """
    quaternion = (
        ros_pose.orientation.x,
        ros_pose.orientation.y,
        ros_pose.orientation.z,
        ros_pose.orientation.w)
    euler = tf.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return np.array([ros_pose.position.x, ros_pose.position.y, ros_pose.position.z, roll, pitch, yaw])

def forward(q):
    s = [sin(q[0]), sin(q[1]), sin(q[2]), sin(q[3]), sin(q[4]), sin(q[5])]
    c = [cos(q[0]), cos(q[1]), cos(q[2]), cos(q[3]), cos(q[4]), cos(q[5])]

    q23 = q[1]+q[2]
    q234 = q[1]+q[2]+q[3]

    s23 = sin(q23)
    c23 = cos(q23)
    s234 = sin(q234)
    c234 = cos(q234)
    T = np.matrix(np.identity(4))
    T[0, 0] = c234*c[0]*s[4] - c[4]*s[0]
    T[0, 1] = c[5]*(s[0]*s[4] + c234*c[0]*c[4]) - s234*c[0]*s[5]
    T[0, 2] = -s[5]*(s[0]*s[4] + c234*c[0]*c[4]) - s234*c[0]*c[5]
    T[0, 3] = d6*c234*c[0]*s[4] - a3*c23*c[0] - a2*c[0]*c[1] - d6*c[4]*s[0] - d5*s234*c[0] - d4*s[0]
    T[1, 0] = c[0]*c[4] + c234*s[0]*s[4]
    T[1, 1] = -c[5]*(c[0]*s[4] - c234*c[4]*s[0]) - s234*s[0]*s[5]
    T[1, 2] = s[5]*(c[0]*s[4] - c234*c[4]*s[0]) - s234*c[5]*s[0]
    T[1, 3] = d6*(c[0]*c[4] + c234*s[0]*s[4]) + d4*c[0] - a3*c23*s[0] - a2*c[1]*s[0] - d5*s234*s[0]
    T[2, 0] = -s234*s[4]
    T[2, 1] = -c234*s[5] - s234*c[4]*c[5]
    T[2, 2] = s234*c[4]*s[5] - c234*c[5]
    T[2, 3] = d1 + a3*s23 + a2*s[1] - d5*(c23*c[3] - s23*s[3]) - d6*s[4]*(c23*s[3] + s23*c[3])
    T[3, 0] = 0.0
    T[3, 1] = 0.0
    T[3, 2] = 0.0
    T[3, 3] = 1.0
    return T

def HTM(i, theta):
    """Calculate the HTM between two links.
    Args:
        i: A target index of joint value.
        theta: A list of joint value solution. (unit: radian)
    Returns:
        An HTM of Link l w.r.t. Link l-1, where l = i + 1.
    """
    T_i = np.matrix(np.identity(4))

    T_i[0, 0] = cos(theta[i])
    T_i[0, 1] = -sin(theta[i])*cos(alpha[i])
    T_i[0, 2] = sin(theta[i])*sin(alpha[i])
    T_i[0, 3] = a[i]*cos(theta[i])
    T_i[1, 0] = sin(theta[i])
    T_i[1, 1] = cos(theta[i])*cos(alpha[i])
    T_i[1, 2] = -cos(theta[i])*sin(alpha[i])
    T_i[1, 3] = a[i]*sin(theta[i])
    T_i[2, 0] = 0
    T_i[2, 1] = sin(alpha[i])
    T_i[2, 2] = cos(alpha[i])
    T_i[2, 3] = d[i]
    T_i[3, 0] = 0
    T_i[3, 1] = 0
    T_i[3, 2] = 0
    T_i[3, 3] = 1

    return T_i

def fwd_kin(theta):
    # The raw D-H parameters specify a transform from the 0th link to the 6th link
    # To work with the raw D-H kinematics, we need to offset the transforms
    # Transform from the base link to 0th link [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # Transform from 6th link to end-effector [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    # The return result is from the base_link to end_effector
    T_06 = np.matrix(np.identity(4))
    # from base_link to 0th link offset transform
    T_06 *= np.matrix([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    positions_list = []
    orientations_list = []

    for i in range(6):
        T_06 *= HTM(i, theta)
        position = T_06[:3, 3]
        orientation = T_06[:3, :3]

        # Append the position and orientation to the respective lists
        positions_list.append(position.copy())  # Store a copy to avoid storing references
        orientations_list.append(orientation.copy())  # Store a copy to avoid storing references
        
    quaternions_list = [Rotation.from_matrix(orientation).as_quat() for orientation in orientations_list]

    positions_list = [position.ravel().tolist() for position in positions_list]
    positions_list = [item for sublist in positions_list for item in sublist]

    # from 6th link to end-effector offset transform
    T_06 *= np.matrix([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    
    # transfer the format into ros msg type
    ros_T = matrix2ros(T_06)
    # transfer the format into x, y, z, euler angles type
    ur_T = ros2euler(ros_T)

    return positions_list, quaternions_list, ros_T, ur_T  

