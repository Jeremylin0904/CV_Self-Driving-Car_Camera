#!/usr/bin/env python

from __builtin__ import True # for python2
import numpy as np
import rospy
import math
import torch
import rospkg
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torchvision
import cv2
import os
from torchvision import transforms, utils, datasets
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


#########################################
###             NEW ADDED             ###
#########################################
import torch.nn.functional as F
from PIL import Image as im

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 32, 5, stride = 3)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 7, stride = 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        # self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4480, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 18)


    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return x


# load model
model = torch.load('weight_group5_1.pth')
bridge = CvBridge()





class Lane_follow(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initial()
        self.omega = 0
        self.count = 0

        self.data_transform = transforms.Compose(
            [transforms.Resize(480), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])])
        # motor omega output (the number of rotation, from L_1 to S_R)
        self.Omega = np.array([0.1,0.17,0.24,0.305,0.37,0.44,0.505,0.73,-0.1,-0.17,-0.24,-0.305,-0.37,-0.44,-0.505,-0.73,0.0,0.0])
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.pub_car_cmd = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)
        self.pub_cam_tilt = rospy.Publisher("/tilt/command", Float64, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1)
    
    # load weight
    def initial(self):
        self.model = model
        self.model = self.model.to(self.device)

       
    # load image to define omega for motor controlling
    def img_cb(self, data):
        #self.dim = (101, 101)  # (width, height)
        self.count += 1
        self.pub_cam_tilt.publish(0.9)
        if self.count == 6:
            self.count = 0
            try:
                # convert image_msg to cv format
                img = bridge.imgmsg_to_cv2(data, desired_encoding = "passthrough")
                img=im.fromarray(img) # new added / bug should be fixed
                img = self.data_transform(img)
                images = torch.unsqueeze(img,0)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                images = images.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(images)
                top1 = output.argmax()
                self.omega = self.Omega[top1]
                
                # motor control
                car_cmd_msg = Twist()
                car_cmd_msg.linear.x = 0.16
                car_cmd_msg.angular.z = self.omega*0.74
                
                self.pub_car_cmd.publish(car_cmd_msg)
                
                rospy.loginfo('\n'+str(self.omega)+'\n'+str(top1))

            except CvBridgeError as e:
                print(e)



if __name__ == "__main__":
    rospy.init_node("lane_follow", anonymous=False)
    lane_follow = Lane_follow()
    rospy.spin()