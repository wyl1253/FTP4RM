# Yongliang Wang
# July 2022
# PyBullet UR5e_robotiq140 Environment 
import random
import time
from turtle import width, window_width
import numpy as np
import sys
from gym import spaces
import gym
import scipy

import os
import math 
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools

from scipy.spatial.transform import Rotation

from ur5e_fk import *



# ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_robotiq140.urdf"
ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"

PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
# TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
TABLE_URDF_PATH = "./ur_e_description/urdf/objects/table.urdf"
SPHERE_URDF_PATH = "./ur_e_description/urdf/objects/sphere.urdf"  # boxes for target
BLOCK_URDF_PATH = "./ur_e_description/urdf/objects/block.urdf"  # boxes for target

# CUBE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "cube_small.urdf")

# x,y,z distance
def goal_distance(goal_a, goal_b):
    goal_a = np.array(goal_a)
    goal_b = np.array(goal_b)
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# x,y distance
def goal_distance2d(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

# def quaternion_angle_diff(quat1, quat2):
#     """Calculate the angle (radians) difference between two quaternions"""
#     q1 = np.array(quat1)
#     q2 = np.array(quat2)
#     dot_product = np.dot(q1, q2)
#     # Ensure dot product is in range [-1, 1] due to potential floating point errors
#     dot_product = np.clip(dot_product, -1.0, 1.0)
#     return 2 * np.arccos(dot_product)

def quaternion_angle_diff(quat1, quat2):
    """Calculate the shortest angle (radians) difference between two quaternions"""
    def shortest_angular_difference(theta1, theta2):
        theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
        theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
        delta_theta = abs(theta2 - theta1)
        if delta_theta > np.pi:
            delta_theta = 2 * np.pi - delta_theta
        return delta_theta

    q1 = np.array(quat1)
    q2 = np.array(quat2)
    dot_product = np.dot(q1, q2)
    # Ensure dot product is in range [-1, 1] due to potential floating point errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the shortest angular difference
    angle_diff = 2 * np.arccos(dot_product)

    return shortest_angular_difference(0, angle_diff)

# def quaternion_angle_diff(quat1, quat2):
#     euler_angles1, euler_angles2 = pybullet.getEulerFromQuaternion(quat1), pybullet.getEulerFromQuaternion(quat2)
#     def shortest_angular_difference(theta1, theta2):
#         theta1 = math.atan2(math.sin(theta1), math.cos(theta1))
#         theta2 = math.atan2(math.sin(theta2), math.cos(theta2))
#         delta_theta = abs(theta2 - theta1)
#         if delta_theta > math.pi:
#             delta_theta = 2 * math.pi - delta_theta
#         return delta_theta

#     roll_diff = shortest_angular_difference(euler_angles1[0], euler_angles2[0])
#     pitch_diff = shortest_angular_difference(euler_angles1[1], euler_angles2[1])
#     yaw_diff = shortest_angular_difference(euler_angles1[2], euler_angles2[2])

#     sum_difference = roll_diff + pitch_diff + yaw_diff
#     return sum_difference

# def generate_points(position, quaternion, d = 0.1):
#     x, y, z = position
#     # Create new points along the orthogonal axes
#     point_x = (x+d, y, z, quaternion)
#     point_y = (x, y+d, z, quaternion)
#     point_z = (x, y, z+d, quaternion)
#     return point_x, point_y, point_z

def generate_points(position, quaternion, distance = 0.001):


    rpy_rad = pybullet.getEulerFromQuaternion(quaternion)

    # Calculate the rotation matrices for roll, pitch, and yaw
    roll_matrix = np.array([[1, 0, 0],
                            [0, math.cos(rpy_rad[0]), -math.sin(rpy_rad[0])],
                            [0, math.sin(rpy_rad[0]), math.cos(rpy_rad[0])]])

    pitch_matrix = np.array([[math.cos(rpy_rad[1]), 0, math.sin(rpy_rad[1])],
                             [0, 1, 0],
                             [-math.sin(rpy_rad[1]), 0, math.cos(rpy_rad[1])]])

    yaw_matrix = np.array([[math.cos(rpy_rad[2]), -math.sin(rpy_rad[2]), 0],
                           [math.sin(rpy_rad[2]), math.cos(rpy_rad[2]), 0],
                           [0, 0, 1]])

    # Calculate the points along the roll, pitch, and yaw orientations
    points_roll = position + distance * roll_matrix[:, 0]
    points_pitch = position + distance * pitch_matrix[:, 1]
    points_yaw = position + distance * yaw_matrix[:, 2]

    # pybullet.loadURDF(BLOCK_URDF_PATH, points_roll)
    # pybullet.loadURDF(BLOCK_URDF_PATH, points_pitch)
    # pybullet.loadURDF(BLOCK_URDF_PATH, points_yaw)

    return points_roll, points_pitch, points_yaw



class ur5eGymEnv(gym.Env):
    def __init__(self,
                 camera_attached=False,
                 # useIK=True,
                 actionRepeat=100,
                 renders=False,
                 maxSteps=100,
                 # numControlledJoints=3, # XYZ, we use IK here!
                 simulatedGripper=False,
                 randObjPos=False,
                 task=0, # here target number
                 learning_param=0):

        self.lp = learning_param
        self.renders = renders
        self.actionRepeat = actionRepeat

        self.goal_roll_line_id = None
        self.goal_pitch_line_id = None
        self.goal_yaw_line_id = None

        self.tool_roll_line_id = None
        self.tool_pitch_line_id = None
        self.tool_yaw_line_id = None

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)
        # pybullet.setGravity(0,0,0)

        pybullet.setRealTimeSimulation(False)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME,1)
        pybullet.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        
        # setup robot arm:
        self.end_effector_index = 7
        self.plane = pybullet.loadURDF(PLANE_URDF_PATH)
        self.table = pybullet.loadURDF(TABLE_URDF_PATH, [0, 0.75, 0.01], [0, 0, 0, 1])

        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.POSITION_CONTROL, targetPosition=0, positionGain=0.1, velocityGain=0.1, force=info.maxForce)
                # pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
        # explicitly deal with mimic joints
        def controlGripper(robotID, parent, children, mul, **kwargs):
            controlMode = kwargs.pop("controlMode")
            if controlMode==pybullet.POSITION_CONTROL:
                pose = kwargs.pop("targetPosition")
                # move parent joint
                pybullet.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose, 
                                        force=parent.maxForce, maxVelocity=parent.maxVelocity) 
                # move child joints
                for name in children:
                    child = children[name]
                    childPose = pose * mul[child.name]
                    pybullet.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose, 
                                            force=child.maxForce, maxVelocity=child.maxVelocity) 
            else:
                raise NotImplementedError("controlGripper does not support \"{}\" control mode".format(controlMode))
            # check if there 
            if len(kwargs) is not 0:
                raise KeyError("No keys {} in controlGripper".format(", ".join(kwargs.keys())))
        # self.mimicParentName = "robotiq_140_joint_finger"
        # self.mimicChildren = {"robotiq_arg2f_base_to_robotiq_140_left_outer_knuckle":      1,
        #                 "robotiq_arg2f_base_to_robotiq_140_left_inner_knuckle":       1,
        #                 "robotiq_arg2f_base_to_robotiq_140_right_inner_knuckle": 1,
        #                 "robotiq_140_left_outer_finger_to_inner":    1,
        #                 "robotiq_140_right_outer_finger_to_inner":   1}
        # self.parent = self.joints[self.mimicParentName] 
        # self.children = AttrDict((j, self.joints[j]) for j in self.joints if j in self.mimicChildren.keys())
        # self.controlRobotiqC2 = functools.partial(controlGripper, self.ur5, self.parent, self.children, self.mimicChildren)

        # object:
        self.initial_obj_pos = [0.5, 0.4, 0.5] # initial object pos
        self.obj = pybullet.loadURDF(BLOCK_URDF_PATH, self.initial_obj_pos)

        # obstacles
        self.initial_obs1_pos = [0.5, 0.3, 0.1] # initial object pos
        self.initial_obs2_pos = [0.5, 0.3, 0.2] # initial object pos
        self.initial_obs3_pos = [0.5, 0.3, 0.3] # initial object pos

        self.obs1 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs1_pos)
        self.obs2 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs2_pos)
        self.obs3 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs3_pos)


        self.name = 'ur5eGymEnv'
        self.simulatedGripper = simulatedGripper
        self.action_dim = 6
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)

        self.task = task
        self.learning_param = learning_param
     
        self._action_bound = 3.14 # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset()
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')

    # def set_joint_angles(self, joint_angles):
    #     poses = []
    #     indexes = []
    #     forces = []
    #     # gripper_opening_length = 0 # close gripper
    #     gripper_opening_length = 0.085 # open gripper
    #     gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  

    #     for i, name in enumerate(self.control_joints):    
    #         joint = self.joints[name]
    #         if i != 6:
    #             poses.append(joint_angles[i])
    #             indexes.append(joint.id)
    #             forces.append(joint.maxForce)

    #         if name==self.mimicParentName:
    #             self.controlRobotiqC2(controlMode=pybullet.POSITION_CONTROL, targetPosition=gripper_opening_angle)
    #         else:
    #             pass

    #     pybullet.setJointMotorControlArray(
    #         self.ur5, indexes,
    #         pybullet.POSITION_CONTROL,
    #         targetPositions=joint_angles,
    #         targetVelocities=[0]*len(poses),
    #         positionGains=[0.05]*len(poses),
    #         forces=forces
    #     )

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )



    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints

    def get_joint_velocities(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joint_velocities = [i[1] for i in j]
        return joint_velocities

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        # print(len(collisions))
        if len(collisions) > 1:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    # def calculate_ik(self, position, orientation):
    #     quaternion = pybullet.getQuaternionFromEuler(orientation)
    #     # quaternion = (0,1,0,1)
    #     lower_limits = [-math.pi]*6
    #     upper_limits = [math.pi]*6
    #     joint_ranges = [2*math.pi]*6
    #     # rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]
    #     rest_poses = [(1.57, -1.57, 1.80, -1.57, -1.57, 0.00)] # rest pose of our ur5 robot

    #     joint_angles = pybullet.calculateInverseKinematics(
    #         self.ur5, self.end_effector_index, position, quaternion, 
    #         jointDamping=[0.01]*6, upperLimits=upper_limits, 
    #         lowerLimits=lower_limits, jointRanges=joint_ranges, 
    #         restPoses=rest_poses
    #     )
    #     return joint_angles
       
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)


    def get_link_lines(self):
        base_link = pybullet.getLinkState(self.ur5, 0, computeForwardKinematics=True)
        shp_link = pybullet.getLinkState(self.ur5, 1, computeForwardKinematics=True)
        shl_link = pybullet.getLinkState(self.ur5, 2, computeForwardKinematics=True)
        elb_link = pybullet.getLinkState(self.ur5, 3, computeForwardKinematics=True)
        wr1_link = pybullet.getLinkState(self.ur5, 4, computeForwardKinematics=True)
        wr2_link = pybullet.getLinkState(self.ur5, 5, computeForwardKinematics=True)
        wr3_link = pybullet.getLinkState(self.ur5, 6, computeForwardKinematics=True)
        ee_link = pybullet.getLinkState(self.ur5, 7, computeForwardKinematics=True)

        link_set = [base_link[4], shp_link[4], shl_link[4], elb_link[4], wr1_link[4], wr2_link[4], wr3_link[4], ee_link[0]]
        
        return link_set, shl_link[5], elb_link[5], ee_link[5]

    # def line_to_point_distance(self, p, q, r):
    #     """Calculate a distance between point r and a line crossing p and q."""
    #     def foo(t: float):
    #         # x is point on line, depends on t
    #         x = t * (p-q) + q
    #         # we return a distance, which also depends on t
    #         return self.get_euclidean_dist(x, r)
    #     # which t minimizes distance?
    #     t0 = scipy.optimize.minimize(foo, 0.1).x[0]
    #     return foo(t0)


    def line_to_point_distance(self, p, q, r):
        """Calculate the distance between point r and a line formed by points p and q."""
        v = q - p
        w = r - p

        # Calculate the projection of w onto v
        projection = np.dot(w, v) / np.dot(v, v)

        if 0 <= projection <= 1:
            closest_point = p + projection * v
        else:
            # If the projection is outside the line segment, choose the closest endpoint
            dist_p = np.linalg.norm(r - p)
            dist_q = np.linalg.norm(r - q)
            closest_point = p if dist_p < dist_q else q

        distance = np.linalg.norm(r - closest_point)
        return distance


    def cal_inter_point(self, a_position, a_quat, length):

        # Convert the quaternion orientation of point a to a rotation matrix
        rotation_matrix = Rotation.from_quat(a_quat).as_matrix()

        # Calculate the direction vector from point a to point b
        direction_vector = np.dot(rotation_matrix, [0, 0, length]) #0.424
        b_position = np.array(a_position) + direction_vector

        return b_position


    # def cal_lines2obs(self, links_positions, links_quaternions, obstacle_position):

    #     l1 = np.array(links_positions[0])
    #     l2 = np.array(links_positions[1])
    #     l3 = np.array(links_positions[2])
    #     l4 = np.array(links_positions[3])
    #     l5 = np.array(links_positions[4])
    #     l6 = np.array(links_positions[5])

    #     l1_5 = self.cal_inter_point(l1, links_quaternions[0], 0.14)
    #     l2_5 = self.cal_inter_point(l2, links_quaternions[1], 0.14)
    #     # l7 = self.cal_inter_point(l6, links_quaternions[5], 0.23)

    #     # pybullet.addUserDebugLine(l1, l1_5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l1_5, l2_5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l2_5, l2, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l2, l3, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l3, l4, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l4, l5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l5, l6, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
    #     # pybullet.addUserDebugLine(l6, l7, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)

    #     op = np.array(obstacle_position)

    #     d1 = self.line_to_point_distance(l1,l1_5,op)
    #     d2 = self.line_to_point_distance(l1_5,l2_5,op)
    #     d3 = self.line_to_point_distance(l2_5,l2,op)
    #     d4 = self.line_to_point_distance(l2,l3,op)
    #     d5 = self.line_to_point_distance(l3,l4,op)
    #     d6 = self.line_to_point_distance(l4,l5,op)
    #     d7 = self.line_to_point_distance(l5,l6,op)
    #     # d8 = self.line_to_point_distance(l6,l7,op)

    #     d = np.min([d1, d2, d3, d4, d5, d6, d7])

    #     # print(d)

    #     return d


    def cal_lines2obs(self, obstacle_position):
        line_points, a_quat, b_quat, c_quat = self.get_link_lines() # 7 points
        l1 = np.array(line_points[2])
        l2 = np.array(line_points[3])
        l3 = np.array(line_points[4])
        l4 = np.array(line_points[5])
        l5 = np.array(line_points[6])
        l6 = np.array(line_points[7])


        op = np.array(obstacle_position)

        l1_5 = self.cal_inter_point(l1, a_quat, 0.42)
        l2_5 = self.cal_inter_point(l2, b_quat, 0.38)
        # l7 = self.cal_inter_point(l6, c_quat, 0.23)


        # pybullet.addUserDebugLine(l1, l1_5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l1_5, l2, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l2, l2_5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l2_5, l3, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l3, l4, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l4, l5, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l5, l6, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)
        # pybullet.addUserDebugLine(l6, l7, lineColorRGB=[0.5,0.6,0.4], lineWidth=5)

        # pybullet.resetBasePositionAndOrientation(self.obs1, [l5[0], l5[1], l5[2]+0.09], [0.,0.,0.,1.0]) # reset object pos

        d1 = self.line_to_point_distance(l1,l1_5,op)
        d2 = self.line_to_point_distance(l1_5,l2,op)
        d3 = self.line_to_point_distance(l2,l2_5,op)
        d4 = self.line_to_point_distance(l2_5,l3,op)
        d5 = self.line_to_point_distance(l3,l4,op)
        d6 = self.line_to_point_distance(l4,l5,op)
        d7 = self.line_to_point_distance(l5,l6,op)
        # d8 = self.line_to_point_distance(l6,l7,op)


        d = np.min([d1, d2, d3, d4, d5, d6, d7])

        return d
   
    def get_euclidean_dist(self, p_in, p_pout):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        distance = np.linalg.norm(p_in-p_pout)

        return distance

    def xyztorus(self):
        circle = False
        while circle == False:
            a = 2*(random.random()-0.5)
            b = random.random()
            c = random.random()
            if (a**2 + b**2 + c**2 <= 0.5625) and (a**2+b**2>=0.09) and (c >= 0.05):
                circle = True
            else :
                circle = False
        return a, b, c

    def rpy(self, x, y, z):

        # tuples = [(0, 1.57, 1.57), (1.57, 0, 3.14), (0, 3.14, -1.57)]
        tuples = [(0, 1.57, 1.57)]

        selected_tuple = random.choice(tuples)
        r, p, y = selected_tuple

        return r, p, y

    def oxyztorus(self, x, y, z):
        circle = False
        while circle == False:
            a = 2 * (random.random() - 0.5)
            b = random.random() 
            c = random.random()
            if ((a-x)**2+(b-y)**2+(c-z)**2 >= 0.0225) and (a**2 + b**2 + c**2 <= 0.5625) and (a**2 + b**2 >= 0.04):
                circle = True
            else :
                circle = False
        return a, b, c

    def xytorus(self):
        circle = False
        while circle == False:
            a = 2*(random.random()-0.5)
            b = random.random()
            # c = random.random() 
            if (a**2+b**2 >= 0.16) and (a**2+b**2 <= 0.9025):
                circle = True
            else :
                circle = False
        return a, b

    def z_work(self, x, y):
        circle = False
        while circle == False:
            c = random.random() 
            if (x**2+y**2+c**2 >= 0.01) and (x**2+y**2+c**2 <= 0.9025):
                circle = True
            else :
                circle = False
        return c

    def pos_neg(self):
        if random.random() < 0.5:
            return 1
        else:
            return -1

    def update_goal_axes_lines(self, pos, orn):

        # If the lines have been drawn before, remove them.
        if self.goal_roll_line_id is not None:
            pybullet.removeUserDebugItem(self.goal_roll_line_id)
            pybullet.removeUserDebugItem(self.goal_pitch_line_id)
            pybullet.removeUserDebugItem(self.goal_yaw_line_id)

        euler = pybullet.getEulerFromQuaternion(orn)
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape((3, 3))
        roll_axis, pitch_axis, yaw_axis = rot_matrix.T

        line_width = 10  # Set the line width to 10 (bold line)
        line_length = 0.08

        # Draw the lines and store their IDs.
        self.goal_roll_line_id = pybullet.addUserDebugLine(pos, pos + line_length * roll_axis, [1, 0, 0], lineWidth=line_width) # Roll in red.
        self.goal_pitch_line_id = pybullet.addUserDebugLine(pos, pos + line_length * pitch_axis, [0, 1, 0], lineWidth=line_width) # Pitch in green.
        self.goal_yaw_line_id = pybullet.addUserDebugLine(pos, pos + line_length * yaw_axis, [0, 0, 1], lineWidth=line_width) # Yaw in blue.

    def update_tool_axes_lines(self, pos, orn):

        # If the lines have been drawn before, remove them.
        if self.tool_roll_line_id is not None:
            pybullet.removeUserDebugItem(self.tool_roll_line_id)
            pybullet.removeUserDebugItem(self.tool_pitch_line_id)
            pybullet.removeUserDebugItem(self.tool_yaw_line_id)

        euler = pybullet.getEulerFromQuaternion(orn)
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape((3, 3))
        roll_axis, pitch_axis, yaw_axis = rot_matrix.T

        line_width = 10  # Set the line width to 3 (bold line)
        line_length = 0.08

        # Draw the lines and store their IDs.
        self.tool_roll_line_id = pybullet.addUserDebugLine(pos, pos + line_length * roll_axis, [1, 0, 0], lineWidth=line_width) # Roll in red.
        self.tool_pitch_line_id = pybullet.addUserDebugLine(pos, pos + line_length * pitch_axis, [0, 1, 0], lineWidth=line_width) # Pitch in green.
        self.tool_yaw_line_id = pybullet.addUserDebugLine(pos, pos + line_length * yaw_axis, [0, 0, 1], lineWidth=line_width) # Yaw in blue.

    def reset(self):
        self.stepCounter = 0
        self.terminated = False
        self.ur5_or = [0.0, 1/2*math.pi, 0.0]

        x,y,z = self.xyztorus()
        o_r,o_p,o_y = self.rpy(x,y,z)
        # print(r,p,y1)
        # x,y,z,r,p,y1 = self.xyzrpy()

        x1,y1,z1 = self.oxyztorus(x, y, z)
        x2,y2,z2 = self.oxyztorus(x, y, z)
        x3,y3,z3 = self.oxyztorus(x, y, z)


        # print(x1,y1,z1,"", x2,y2,z2, "", x3,y3,z3)

        # pybullet.addUserDebugText('X', self.obj_pos, [0,1,0], 1) # display goal
        if self.randObjPos:
           self.initial_obj_pos = [x,y,z]
           self.initial_obj_ori = pybullet.getQuaternionFromEuler([o_r,o_p,o_y])

           self.initial_obs1_pos = [x1,y1,z1]
           self.initial_obs2_pos = [x2,y2,z2]
           self.initial_obs3_pos = [x3,y3,z3]

        #    self.initial_obj_pos = [-0.03942809486685994,0.5782546623048851,0.5928282974056718]
        #    pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, [0.,0.,0.,1.0]) # reset object pos
           pybullet.resetBasePositionAndOrientation(self.obj, self.initial_obj_pos, self.initial_obj_ori) # reset object pos

           pybullet.resetBasePositionAndOrientation(self.obs1, self.initial_obs1_pos, [0.,0.,0.,1.0]) # reset object pos
           pybullet.resetBasePositionAndOrientation(self.obs2, self.initial_obs2_pos, [0.,0.,0.,1.0]) # reset object pos
           pybullet.resetBasePositionAndOrientation(self.obs3, self.initial_obs3_pos, [0.,0.,0.,1.0]) # reset object pos

        if self.renders:
            # Display Roll, Pitch and Yaw axes. Do not display when it is training
            pos, orn = pybullet.getBasePositionAndOrientation(self.obj)
            self.update_goal_axes_lines(pos, orn)

        # reset robot simulation and position:
        joint_angles = (1.57, -1.57, 1.57, -1.57, -1.57, 0.00) # pi/2 = 1.5707
        self.set_joint_angles(joint_angles)
        # step simualator:
        for i in range(200):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)

        # get obs and return:
        self.getExtendedObservation()

        return self.observation
    
    
    def step(self, action, lp):

        self.lp = lp

        action = np.array(action)
        arm_action = action[0:self.action_dim].astype(float) # j1-j6 - range: [-1,1]


        inti_joint_angles = (1.57, -1.57, 1.57, -1.57, -1.57, 0.00) # pi/2 = 1.5707

        joint_angles = arm_action + inti_joint_angles

        # move joints
        self.set_joint_angles(joint_angles)

        # # step simualator:
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)

        self.getExtendedObservation()
        reward = self.compute_reward(self.obj_tool_pos, self.obj_tool_ori, joint_angles, None)
        done = self.my_task_done()

        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True

        self.stepCounter += 1

        return self.observation, reward, done, info

    # observations are: arm position, arm acceleration, ...
    def getExtendedObservation(self):
        # sensor values:
        js = self.get_joint_angles()
        jv = self.get_joint_velocities()

        obs1 = pybullet.getBasePositionAndOrientation(self.obs1)
        obs2 = pybullet.getBasePositionAndOrientation(self.obs2)
        obs3 = pybullet.getBasePositionAndOrientation(self.obs3)


        # links_positions, links_quaternions, ros_T, tool_pose = fwd_kin(js)

        obstacle_dis1 = self.cal_lines2obs(obs1[0])
        obstacle_dis2 = self.cal_lines2obs(obs2[0])
        obstacle_dis3 = self.cal_lines2obs(obs3[0])

        # obstacle_dis1 = self.cal_lines2obs(links_positions, links_quaternions, obs1[0])
        # obstacle_dis2 = self.cal_lines2obs(links_positions, links_quaternions, obs2[0])
        # obstacle_dis3 = self.cal_lines2obs(links_positions, links_quaternions, obs3[0])


        tool_pos = self.get_current_pose()[0] # XYZ
        tool_ori = self.get_current_pose()[1] # Quaternion

        # print(pybullet.getEulerFromQuaternion(tool_ori))
        if self.renders:
            # Do not display when it is training
            self.update_tool_axes_lines(tool_pos, tool_ori)

        self.obj_pos, self.obj_ori = pybullet.getBasePositionAndOrientation(self.obj)

        self.obj_tool_pos = np.array(np.concatenate((self.obj_pos, tool_pos)))
        self.obj_tool_ori = np.array(np.concatenate((self.obj_ori, tool_ori)))

        error_x = tool_pos[0] - self.obj_pos[0]
        error_y = tool_pos[1] - self.obj_pos[1]
        error_z = tool_pos[2] - self.obj_pos[2]

        error_o1 = tool_ori[0] - self.obj_ori[0]
        error_o2 = tool_ori[1] - self.obj_ori[1]
        error_o3 = tool_ori[2] - self.obj_ori[2]
        error_o4 = tool_ori[3] - self.obj_ori[3]

        error_xyz = [error_x, error_y, error_z]
        error_o = [error_o1, error_o2, error_o3, error_o4]


        orient_error = quaternion_angle_diff(self.obj_tool_ori[-4:], self.obj_tool_ori[:4])


        tool_p1, tool_p2, tool_p3 = generate_points(self.obj_tool_pos[-3:], self.obj_tool_ori[-4:])
        goal_p1, goal_p2, goal_p3 = generate_points(self.obj_tool_pos[:3], self.obj_tool_ori[:4])

        p1_dist = goal_distance(tool_p1[:3], goal_p1[:3])
        p2_dist = goal_distance(tool_p2[:3], goal_p2[:3])
        p3_dist = goal_distance(tool_p3[:3], goal_p3[:3])

        obs_dis = np.array([obstacle_dis1, obstacle_dis2, obstacle_dis3])

        # print("obs_dis", obs_dis)

        error = np.array([p1_dist + p2_dist + p3_dist, orient_error])

        # print(error)

        self.observation = np.array(np.concatenate((js, tool_pos, tool_ori, self.obj_pos, self.obj_ori, error, obs_dis)))


    def my_task_done(self):
        # NOTE: need to call compute_reward before this to check termination!
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c

    def compute_reward(self, obj_tool_pos, obj_tool_ori, joint_angles, info):
        reward = 0

        # obs1 = pybullet.getBasePositionAndOrientation(self.obs1)
        # obs2 = pybullet.getBasePositionAndOrientation(self.obs2)
        # obs3 = pybullet.getBasePositionAndOrientation(self.obs3)

        # links_positions, links_quaternions, ros_T, tool_pose = fwd_kin(joint_angles)

        # obstacle_dis1 = self.cal_lines2obs(obs1[0])
        # obstacle_dis2 = self.cal_lines2obs(obs2[0])
        # obstacle_dis3 = self.cal_lines2obs(obs3[0])

        # obstacle_dis1 = self.cal_lines2obs(links_positions, links_quaternions, obs1[0])
        # obstacle_dis2 = self.cal_lines2obs(links_positions, links_quaternions, obs2[0])
        # obstacle_dis3 = self.cal_lines2obs(links_positions, links_quaternions, obs3[0])

        obstacle_dis1, obstacle_dis2, obstacle_dis3 = self.observation[-3:]
        # print(obstacle_dis1, obstacle_dis2, obstacle_dis3)

        distance_threshold = 0.08

        fa1 = max(0, 1 - obstacle_dis1/distance_threshold)
        fa2 = max(0, 1 - obstacle_dis2/distance_threshold)
        fa3 = max(0, 1 - obstacle_dis3/distance_threshold)

        fa = fa1+fa2+fa3

        # print(obstacle_dis1, obstacle_dis2, obstacle_dis3)


        # grip_pos = obj_tool_pos[-3:]
        # goal_pos = obj_tool_pos[:3]

        # grip_ori = obj_tool_ori[-4:]
        # goal_ori = obj_tool_ori[:4]

        # self.target_dist = goal_distance(grip_pos, goal_pos)
        # self.orient_error = quaternion_angle_diff(grip_ori, goal_ori)

        # tool_p1, tool_p2, tool_p3 = generate_points(obj_tool_pos[-3:], obj_tool_ori[-4:])
        # goal_p1, goal_p2, goal_p3 = generate_points(obj_tool_pos[:3], obj_tool_ori[:4])

        # p1_dist = goal_distance(tool_p1[:3], goal_p1[:3])
        # p2_dist = goal_distance(tool_p2[:3], goal_p2[:3])
        # p3_dist = goal_distance(tool_p3[:3], goal_p3[:3])


        # # errors_coef = 0.1*(1 - ((p1_dist + p2_dist + p3_dist) / (0.01 + (p1_dist + p2_dist + p3_dist))))
        # # errors = (p1_dist + p2_dist + p3_dist) + errors_coef*(self.orient_error/6.28)
        # error = p1_dist + p2_dist + p3_dist

        # errors = max(self.target_dist, 0.25*self.orient_error)

        error1 = self.observation[-5]
        error2 = self.observation[-4]

        # print("error1:%d", error1)
        # print("error2:%d", error2)


        errors = 0.5*(error1) + 0.25*(error2)

        # print(self.orient_error, error, errors)

        # check approach velocity:
        # tv = self.tool.getVelocity()
        # approach_velocity = np.sum(tv)

        # reward += -self.target_dist * 10

        # if self.target_dist < 0.1:
        #    reward += 0.5*np.exp(-300*pow(self.target_dist, 2))+5
        # else :
        #    reward += -np.log(pow(self.target_dist, 2))

        # print(self.target_dist)

        # reward += np.exp(-20*(pow(self.errorx, 2)+pow(self.errory, 2)+pow(self.errorz, 2)))
        # reward += 0.5*(np.exp(-50*(pow(self.target_dist, 2))))+0.5*(-0.1*np.log(pow(self.target_dist, 2)))
        # reward += np.exp(-30*pow(self.target_dist, 2))


        reward += -0.001*pow(errors, 2) - np.log(pow(errors, 2)+0.0001) - 0.1*fa


        # reward += -np.log(pow(self.target_dist, 2) + 5*self.target_dist)

        # reward += -np.log(pow(self.target_dist, 2)+0.5*self.target_dist+np.sqrt(self.target_dist))


        # reward += -np.log((10*(i_episode/max_episodes)+1)*pow(self.target_dist, 2))

        # task 0: reach object:
        if error1/3 < self.lp:# and approach_velocity < 0.05:
            self.terminated = True
            # print('Successful!')

        # penalize if it tries to go lower than desk / platform collision:
        # if grip_trans[1] < self.desired_goal[1]-0.08: # lower than position of object!
            # reward[i] += -1
            # print('Penalty: lower than desk!')

        # check collisions:
        if self.check_collisions(): 
            reward += -1
            # print('Collision!')

        return reward
