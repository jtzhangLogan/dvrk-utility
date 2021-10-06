# Author: Jintan Zhang
# Date: 09/24/2021

import sys
import dvrk
import PyKDL
import numpy as np
import time as t
import rospy
import collections
from geometry_msgs.msg import WrenchStamped


class FsRegister:
    def __init__(self, arm, iters, force, size):
        # create arm
        self.p = dvrk.psm(arm)

        # parameters
        self.order_name = ['x', 'y', 'z']
        self.order = [0, 1, 2]
        self.force = force
        self.iters = iters
        self.size = size

        # rostopic subscriber
        self.fs_data = collections.deque(maxlen=size)
        rospy.Subscriber('/measured_cf', WrenchStamped, self.fs_cb)

        # test force
        self.jaw_cf = np.zeros((1, 1))

        # local ram
        self.register_out = []
        self.rot = []

    def validate(self, axis, body_cf, Rot_M):
        # collect bias
        bias = self.reset_data()

        # send test force
        self.p.body.servo_cf(body_cf)
        t.sleep(t_wait)

        # collect force sensor feedback
        test_gt = self.reset_data()

        # restore
        self.p.body.servo_cf(np.array([0, 0, 0, 0, 0, 0]))
        t.sleep(t_wait)

        # check
        est = Rot_M.dot(body_cf[0:3].T) + bias
        print("axis {}: \n GT = {}, EST = {}\n".format(axis, test_gt, est))

    def fs_cb(self, msg):
        self.fs_data.append([msg.wrench.force.x] + [msg.wrench.force.y] + [msg.wrench.force.z])

    def reset_data(self):
        self.fs_data.clear()
        while len(self.fs_data) < self.size:
            continue
        return np.mean(self.fs_data, axis=0)

    def register(self):
        # send zero force, user find contact point
        raw_input("Press any key to disable PID")
        body_cf = np.zeros((1, 6))
        self.p.jaw.servo_jf(body_cf)

        # contact point found, close grip
        raw_input("Press any key to close gripper")
        self.jaw_cf[0] = -0.3
        self.p.jaw.servo_jf(self.jaw_cf)

        # main loop
        r = rospy.Rate(100)

        raw_input("Press any key to start registration... (P.S.: might be a good idea to rebias sensor)")

        while not rospy.is_shutdown():

            for idx, j in enumerate(self.order):
                # exert force in x axis
                print("exert {} force...".format(self.order_name[idx]))

                fb = []
                ref = []
                for i in range(iters):
                    print("Attempt {} ...".format(i))
                    # collect reference
                    ref.append(self.reset_data())

                    # apply force
                    if self.order_name[j] == 'x':
                        body_cf = np.array([self.force, 0, 0, 0, 0, 0])
                    elif self.order_name[j] == 'y':
                        body_cf = np.array([0, self.force, 0, 0, 0, 0])
                    elif self.order_name[j] == 'z':
                        body_cf = np.array([0, 0, self.force, 0, 0, 0])

                    self.p.body.servo_cf(body_cf)
                    t.sleep(t_wait)

                    # collect feedback
                    fb.append(self.reset_data())

                    # restore
                    self.p.body.servo_cf(np.array([0, 0, 0, 0, 0, 0]))
                    t.sleep(t_wait)

                # compute mean
                x_mean = np.mean(np.array(fb)[:, 0] - np.array(ref)[:, 0])
                y_mean = np.mean(np.array(fb)[:, 1] - np.array(ref)[:, 1])
                z_mean = np.mean(np.array(fb)[:, 2] - np.array(ref)[:, 2])
                print("registered {} vector = {}".format(self.order_name[j], (x_mean, y_mean, z_mean)))
                self.register_out.append(np.array([x_mean, y_mean, z_mean]))

            if len(self.register_out) == len(self.order_name):
                # --------------------------------------------------
                # check if vectors are perpendicular to each other
                # --------------------------------------------------
                print("sanity check...")
                vec_x = self.register_out[0]
                vec_y = self.register_out[1]
                vec_z = self.register_out[2]
                xy_deg = np.arccos(vec_x.dot(vec_y.T) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))) * 180 / np.pi
                yz_deg = np.arccos(vec_y.dot(vec_z.T) / (np.linalg.norm(vec_y) * np.linalg.norm(vec_z))) * 180 / np.pi
                xz_deg = np.arccos(vec_x.dot(vec_z.T) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_z))) * 180 / np.pi
                print("xy = {}, yz = {}, xz = {}".format(xy_deg, yz_deg, xz_deg))

                # --------------------------------------------------
                # Get transformation
                # --------------------------------------------------
                # get un-normalized quaternion
                q_orig = PyKDL.Rotation(vec_x[0], vec_y[0], vec_z[0],
                                        vec_x[1], vec_y[1], vec_z[1],
                                        vec_x[2], vec_y[2], vec_z[2]).GetQuaternion()

                # get normalize quaternion
                q_orig = q_orig / (np.linalg.norm(q_orig))

                # get normalized rotational matrix
                r_norm = PyKDL.Rotation.Quaternion(q_orig[0], q_orig[1], q_orig[2], q_orig[3])
                Rot_M = np.array([[r_norm.UnitX().x(), r_norm.UnitY().x(), r_norm.UnitZ().x()],
                                  [r_norm.UnitX().y(), r_norm.UnitY().y(), r_norm.UnitZ().y()],
                                  [r_norm.UnitX().z(), r_norm.UnitY().z(), r_norm.UnitZ().z()]])

                # --------------------------------------------------
                # check correctness of calculated rotation matrix
                # --------------------------------------------------
                # check x
                body_cf = np.array([self.force, 0, 0, 0, 0, 0])
                self.validate("PSM-X", body_cf, Rot_M)

                # check y
                body_cf = np.array([0, self.force, 0, 0, 0, 0])
                self.validate("PSM-Y", body_cf, Rot_M)

                # check z
                body_cf = np.array([0, 0, self.force, 0, 0, 0])
                self.validate("PSM-Z", body_cf, Rot_M)

                # check xyz
                body_cf = np.array([self.force, self.force, self.force, 0, 0, 0])
                self.validate("PSM-XYZ", body_cf, Rot_M)

                # --------------------------------------------------
                # wait for user feedback
                # --------------------------------------------------
                print("The rotation matrix is \n{}".format(Rot_M))
                decision = raw_input("Press r repeat, any other key to exit...\n")

                if decision != 'r':
                    sys.exit(0)

            r.sleep()


if __name__ == "__main__":
    arm = 'PSM2'
    iters = 2
    force = 4
    size = 25
    t_wait = 2

    fsRegister = FsRegister(arm, iters, force, size)
    fsRegister.register()
