#! -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
import glob
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

class CycleGAN():
    def __init__(self):
        #参数与结构
        self.img_dim = 128
        self.z_dim = 100
        self.img_path = glob.glob(r'')

    def load_img(self,im):
        x = misc.imread(im)
        h,w = x.shape[:2]
        if(h >= w):
            center_h = int((h - w) / 2)
            x = x[center_h:center_h + w,:]
        if(h < w):
            center_w = int((w - h) / 2)
            x = x[:,center_w:center_w + h]
        return x.astype(np.float32) / 255

    def data_generator(self, batch_size,im):
        x = []
        while True:
            np.random.shuffle(im)
            for i in im:
                x.append(self.load_img(i))
                if len(x) == batch_size:
                    x = np.array(x)
                    yield x
                    x = []

    def G_model(self,z_dim,img_dim):
        z_in = Input(shape=(z_dim,))
        z = z_in
        z = Dense(4 * 4 * img_dim * 8)(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = Reshape((4, 4, img_dim * 8))(z)

        for i in range(3):
            z = Conv2DTranspose(img_dim * 4 // 2**i,
                                (5,5),
                                strides=(2,2),
                                padding='same')(z)
            z = BatchNormalization()(z)
            z = Activation('relu')(z)
        
        z = Conv2DTranspose(3,
                            (5,5),
                            strides=(2,2),
                            padding='same')(z)
        z = Activation('tanh')(z)

        g_model = Model(z_in, z)
        return g_model

    def D_model(self,img_dim):
        x_in = Input(shape=(img_dim, img_dim, 3))
        x = x_in

        x = Conv2D(img_dim,
                    (5,5),
                    strides=(2,2),
                    padding='same')(x)
        x = LeakyReLU()(x)
        
        for i in range(3):
            x = Conv2D(img_dim * 2**(i + 1),
                    (5,5),
                    strides=(2,2),
                    padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(1,activition='sigmoid')(x) #这里可能需要修改

        d_model = Model(x_in,x)
        return d_model