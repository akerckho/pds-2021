import gym
import numpy as np
import sys
import torch
from torch.autograd import Variable
import PIL
from torch.nn import Softmax
from pyglet.window import key

from model import CustomModel
from data import transform_driving_image, LEFT, RIGHT, GO

import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from car_racing import *


id_to_steer = {
    LEFT: -1,
    RIGHT: 1,
    GO: 0.3,
}

if __name__=='__main__':

    if len(sys.argv) < 2:
        sys.exit("Usage : python drive.py path/to/weights")
    # load the model
    model_weights = sys.argv[1]
    model = CustomModel()
    model.load_state_dict(torch.load(model_weights))

    env = CarRacingImage()
    env.reset()

    a = np.array([0.0, 0.0, 0.1])

    def key_press(k, mod):
        global restart
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = 0.3
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    
            a[1] = 0
            a[2] = 0.1
        if k==key.DOWN:  a[2] = 0
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    
    # initialisation
    for i in range(50):
        env.step([0, 0, 0])
        #env.render()
    
    i = 0
    isopen = True
    while isopen:
        env.reset()
        while True:
            s, r, done = env.step(a)
            s = s.copy()
            # We transform our numpy array to PIL image
            # because our transformation takes an image as input
            s  = PIL.Image.fromarray(s)  
            input = transform_driving_image(s).to(model.device)
            with torch.no_grad():
                input = Variable(input[None, :])
            output = Softmax()(model(input))
            data, index = output.max(1)
            index = index.data[0]
            if index in (0, 1):
                a[0] = id_to_steer[index.item()] * output.data[0, index] * 0.3  # lateral acceleration
            else:
                a[0] = 0 
                a[1] = 0.3 * output.data[0, index]
                a[2] = 0.1

            isopen = env.render()
            if done or not isopen:
            	break
    env.close()

    
