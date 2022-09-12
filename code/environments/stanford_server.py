import gym
import json
import numpy as np
import pickle
import time
import zlib
import zmq

from stanford_client import StanfordEnvironmentClient
from humanav_examples.examples import * 

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
def generate_dummy_observation(state_arr):
    # generate a dummy 32 by 32 observation
    low = np.zeros([32, 32, 3], dtype=np.uint8)
    high = np.ones([32, 32, 3], dtype=np.uint8)*255
    observation_space = gym.spaces.Box(low, high)
    return observation_space.sample()



while True:
    #  Wait for next request from client
    flags=0
    copy=True
    track=False
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    state_arr = np.frombuffer(buf, dtype=md['dtype'])
    state_arr = state_arr.reshape(md['shape'])
    print("Received request for state", state_arr)
    
    #  Send reply back to client
    # img = generate_observation_retimg(state_arr)
    img = generate_dummy_observation(state_arr)
    flags=0
    copy=True
    track=False
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(img.dtype),
        shape = img.shape
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    socket.send(img, flags, copy=copy, track=track)

