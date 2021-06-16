# -*- coding: utf-8 -*-

## make this version of pyqtgraph importable before any others
## we do this to make sure that, when running examples, the correct library
## version is imported (if there are multiple versions present).
import sys, os

if not hasattr(sys, 'frozen'):
    if __file__ == '<stdin>':
        path = os.getcwd()
    else:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path.rstrip(os.path.sep)
    if 'pyqtgraph' in os.listdir(path):
        sys.path.insert(0, path) ## examples adjacent to pyqtgraph (as in source tree)
    else:
        for p in sys.path:
            if len(p) < 3:
                continue
            if path.startswith(p):  ## If the example is already in an importable location, promote that location
                sys.path.remove(p)
                sys.path.insert(0, p)

import pyqtgraph as pg    
print("Using", pg.Qt.QT_LIB)

## Enable fault handling to give more helpful error messages on crash. 
## Only available in python 3.3+
try:
    import faulthandler
    faulthandler.enable()
except ImportError:
    pass

from pyqtgraph.Qt import QtCore, QtGui, mkQApp
import pyqtgraph.opengl as gl

from OpenGL.GL import *
from OpenGL.GL import shaders

import numpy as np
from PIL import Image

import torch
import datetime
from raymarching import raymarching_fastnerf


def shaderFromFile(shaderType, shaderFile):
    '''create shader from file'''
    shaderSrc = ''
    with open(shaderFile) as sf:
        shaderSrc = sf.read()
    return shaders.compileShader(shaderSrc, shaderType)

class ImageViewWidget(gl.GLViewWidget):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # buffer object ids
        self.vaoID = None
        self.vboVerticesID = None
        self.vboIndicesID = None
        self.textureID = None
        self.sprogram = None
        
        self.vertices = None
        self.indices = None

        self.window_size = (800, 800)
        self.focal = 3804 * (800 / 4032)
        self.near = 0.1 # 1.3333333333333333
        self.far = 5 # 4.905785983822146
        self.xmin, self.xmax = -1, 1 # left/right range
        self.ymin, self.ymax = -1, 1 # forward/backward range
        self.zmin, self.zmax = -2.7, -0.7 # up/down range  
        
        self.opts['center'] = pg.Vector((self.xmin + self.xmax)/2, (self.ymin + self.ymax)/2, (self.zmin + self.zmax)/2)
        self.opts['distance'] = 2

        self.setGeometry(40, 40, self.window_size[0], self.window_size[1])
        self.setWindowTitle('test')

        self.volume = torch.from_numpy(np.load('/home/tang/projects/nerf_pl/fastnerf/silica_hr_cache.npy')).cuda()
        print(f'load volume: {self.volume.shape} {self.volume.dtype}')
        
        self.im = None
    
    def mouseMoveEvent(self, ev):
        lpos = ev.localPos()
        diff = lpos - self.mousePos
        self.mousePos = lpos

        # slow down
        dx = diff.x() / 5
        dy = diff.y() / 5
        
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(dx, dy, 0, relative='view')
            else:
                self.orbit(-dx, dy)
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.pan(dx, 0, dy, relative='view-upright')
            else:
                self.pan(dx, dy, 0, relative='view-upright')
    
    def initializeGL(self):
        glClearColor(0, 0, 0, 0)
        
        # create shader from file
        vshader = shaderFromFile(GL_VERTEX_SHADER, 'shader.vert')
        fshader = shaderFromFile(GL_FRAGMENT_SHADER, 'shader.frag')
        # compile shaders
        self.sprogram = shaders.compileProgram(vshader, fshader)
        
        # get attribute and set uniform for shaders
        glUseProgram(self.sprogram)
        self.vertexAL = glGetAttribLocation(self.sprogram, 'pos')
        self.tmUL = glGetUniformLocation(self.sprogram, 'textureMap')
        glUniform1i(self.tmUL, 0)
        glUseProgram(0)
        
        # two triangle to make a quad
        self.vertices = np.array((0.0, 0.0, 
                                  1.0, 0.0, 
                                  1.0, 1.0, 
                                  0.0, 1.0), dtype=np.float32)
        self.indices = np.array((0, 1, 2, 
                                 0, 2, 3), dtype=np.ushort)
        
        # set up vertex array
        self.vaoID = glGenVertexArrays(1)
        self.vboVerticesID = glGenBuffers(1)
        self.vboIndicesID = glGenBuffers(1)
        
        glBindVertexArray(self.vaoID)
        glBindBuffer(GL_ARRAY_BUFFER, self.vboVerticesID)
        # copy vertices data from memery to gpu memery
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # tell opengl how to procces the vertices data
        glEnableVertexAttribArray(self.vertexAL)
        glVertexAttribPointer(self.vertexAL, 2, GL_FLOAT, GL_FALSE, 0, None)
        # send the indice data too
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vboIndicesID)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        
        print("Initialization successfull")
        
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        
    def paintGL(self, *args, **kwargs):
        # get camera
        mat = np.array(self.viewMatrix().data()).reshape(4, 4).T
        c2w = mat[:3, :] # [3, 4]
        c2w = torch.FloatTensor(c2w).cuda()

        # get image
        start_time = datetime.datetime.now().time()
        image = raymarching_fastnerf(self.volume, self.window_size[0], self.window_size[1], self.focal, c2w, self.near, self.far, self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)
        image = image.cpu().numpy()
        frames = datetime.datetime.now().time()
        frames = frames.second - start_time.second + (frames.microsecond - start_time.microsecond) / 1e6
        print(f'framerate: {1 / frames: .4f}')

        image = (image * 255).astype(np.uint8)
        self.im = Image.fromarray(image)

        # flip the image in the Y axis
        self.im = self.im.transpose(Image.FLIP_TOP_BOTTOM)
        
        # set up texture
        self.textureID = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        # set filters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # set uv coords mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        # send the image data to gpu memery
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.im.size[0], self.im.size[1], 
                     0, GL_RGB, GL_UNSIGNED_BYTE, self.im.tobytes())

        # clear the buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # active shader
        glUseProgram(self.sprogram)
        # draw triangles
        glDrawElements(GL_TRIANGLES, self.indices.size, GL_UNSIGNED_SHORT, None)
        glUseProgram(0)


app = mkQApp("GLViewWidget Example")
w = ImageViewWidget(rotationMethod='quaternion')
w.show()

if __name__ == '__main__':
    pg.exec()