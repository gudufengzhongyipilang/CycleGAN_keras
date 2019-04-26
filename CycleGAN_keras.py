#! -*- coding: utf-8 -*-
'''
Created by FSJ,2019.04.25
'''
import numpy as np
from scipy import misc
import glob
import datetime
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam,RMSprop
from keras import backend as K

class CycleGAN():
    def __init__(self):
        '''
        参数与结构

        Q1：为什么要定义输出？
        A1: 因为使用keras框架，只有定义了输入输出，才能定义一个模型。

        Q2: 为何输入输出都不用定义为私有变量？
        A2：如上所述，输入输出只用于定义模型，不会涉及类内参数或函数传导；简便起见，不作为私有变量。
        '''
        #定义 输入图像
        self.img_dim = 64
        img_x = Input(shape=(self.img_dim,self.img_dim,3))
        img_y = Input(shape=(self.img_dim,self.img_dim,3))
        #定义 循环一致性的变换函数（包含域映射结构 和 生成器）
        self.F_x2y = self.tra_F_G()
        self.G_y2x = self.tra_F_G()
        #定义 域映射结果
        im_fake_y = self.F_x2y(img_x)
        im_fake_x = self.G_y2x(img_y)
        #定义 域返回映射结果
        reconstr_x = self.G_y2x(im_fake_y)
        reconstr_y = self.F_x2y(im_fake_x)
        '''训练判别器的过程'''
        #定义 GAN的判别器D
        self.D_x = self.Discriminator()
        self.D_y = self.Discriminator()
        #定义 判别器D的输出结果
        '''
        注意，判别器输入输出都是单个量
        '''
        valid_x1 = self.D_x(im_fake_y)
        valid_x2 = self.D_x(img_y)
        loss1 = K.mean(K.log(img_x)) + K.mean(K.log(1 - self.D_x(im_fake_x)))
        self.D_x_train_model = Model([img_x,img_y],[valid_x1,valid_x2])

        valid_y1 = self.D_y(im_fake_x)
        valid_y2 = self.D_y(img_x)
        loss2 = K.mean(K.log(img_y)) + K.mean(K.log(1 - self.D_y(im_fake_y)))
        self.D_y_train_model = Model([img_y,img_x],[valid_y1,valid_y2])

        self.D_x.add_loss(loss1)
        self.D_y.add_loss(loss2)
        self.D_x_train_model.compile(optimizer = Adam(2e-4, 0.5))
        self.D_y_train_model.compile(optimizer = Adam(2e-4, 0.5))
        '''训练生成器的过程'''
        self.D_x_train_model.trainable = False
        self.D_y_train_model.trainable = False
        #定义 整个模型
        '''
        对输入输出进行解释：
        输入：两张图
        输出：1、判别器输出；2、循环一致性输出；3、期望输出（即 进行风格转换的输出）
        '''
        self.g_model = Model([img_x,img_y],
                            [valid_x1,valid_y1,
                            reconstr_x,reconstr_y,
                            im_fake_y,im_fake_x])
        #定义 整个CycleGAN的Loss
        lamda = 0.1
        cyc_loss = K.mean(K.sum(K.abs(reconstr_x - img_x))) + K.mean(K.sum(K.abs(reconstr_y - img_y)))
        total_loss = loss1 + loss2 + lamda * cyc_loss
        #加入Loss并编译
        self.g_model.add_loss(total_loss)
        self.g_model.compile(optimizer = Adam(2e-4, 0.5))

    #通过路径读取图像(内部调用，不用传参)
    def load_img(self,im_path):
        x = misc.imread(im_path)
        h,w = x.shape[:2]
        if(h >= w):
            center_h = int((h - w) / 2)
            x = x[center_h:center_h + w,:]
        if(h < w):
            center_w = int((w - h) / 2)
            x = x[:,center_w:center_w + h]
        return x.astype(np.float32) / 255

    #成批次的随机抽取样本（传入glob的路径列表）
    def data_generator(self, batch_size,im_path):
        x = []
        while True:
            np.random.shuffle(im_path)
            for i in im_path:
                x.append(self.load_img(i))
                if len(x) == batch_size:
                    x = np.array(x)
                    yield x
                    x = []

    #定义映射网络
    def tra_F_G(self,img_dim):
        #域间映射函数
        x_in = Input(shape=(self.img_dim,self.img_dim,3))
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

        #生成器
        z = x
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

        model = Model(x_in, z)
        return model

    #判别器
    def Discriminator(self,img_dim):
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
        x = Dense(1,activition='sigmoid')(x) 

        model = Model(x_in,x)
        return model

    #定义训练函数，实质上就是将之前的结构传入具体的数据和参数
    def train(self):
        im_path = glob.glob(r"data/*/*.jpg")
        total_item = 200000
        steps_per_sample = 200
        batch_size = 64
        img_generator1 = data_generator(batch_size,im_path)
        img_generator2 = data_generator(batch_size,im_path)
        
        '''train D & G'''
        star_time = datetime.datetime.now()
        for i in range(total_item):
            d1_loss = self.D_y_train_model.train_on_batch([img_generator1.__next__(),img_generator2.__next__()],None)
            d2_loss = self.D_x_train_model.train_on_batch([img_generator1.__next__(),img_generator2.__next__()],None)
            cycle_loss = self.g_model.train_on_batch([img_generator1.__next__(),img_generator2.__next__()],None)
        train_sque = datetime.datetime.now() - star_time