import numpy as np
from tqdm import tqdm

with open('glove/new-try.txt', 'r') as f:
    glove = {}
    for line in tqdm(f):
        vals = line.rstrip().split(' ')
        glove[vals[0]] = np.array([float(x) for x in vals[1:]])    

def closest(vec):
    mindist = np.inf
    closest = None
    for word, vec2 in glove.items():
        dist = np.sum((vec-vec2)**2)
        if dist < mindist:
            mindist = dist
            closest = word
    return closest

class V:
    def __init__(self, x):
        if type(x) is str:
            self.vector = glove[x]
        else:
            self.vector = x
        
    def __str__(self):
        return closest(self.vector)
    
    def __repr__(self):
        return closest(self.vector)
    
    def __add__(self, other):
        return V(self.vector + other.vector)
    
    def __sub__(self, other):
        return V(self.vector - other.vector)
