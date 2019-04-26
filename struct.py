#! -*- coding: utf-8 -*-
'''
Created by FSJ,2019.04.24
'''
import numpy as np
from scipy import misc
import glob
from keras.layers import Input
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
        self.F_x2y = self.F_x2y()
        self.G_y2x = self.G_y2x()
        #定义 域映射结果
        im_fake_y = self.F_x2y(img_x)
        im_fake_x = self.G_y2x(img_y)
        #定义 域返回映射结果
        reconstr_x = self.G_y2x(im_fake_y)
        reconstr_y = self.F_x2y(im_fake_x)
        #定义 通过F、G映射进行风格转换的输出
        translation_x2y = self.F_x2y(img_x)
        translation_y2x = self.G_y2x(img_y)
        #定义 GAN的判别器D
        self.D_x = self.D_x()
        self.D_y = self.D_y()
        #定义 判别器D的输出结果
        '''
        注意，判别器输入输出都是单个量
        '''
        valid_x = self.D_x(im_fake_x)
        valid_y = self.D_y(im_fake_y)
        #向D中加入loss并编译
        loss1 = K.mean(K.log(valid_x)) + K.mean(K.log(1 - self.D_x(im_fake_x)))
        loss2 = K.mean(K.log(valid_y)) + K.mean(K.log(1 - self.D_y(im_fake_y)))
        self.D_x.add_loss(loss1)
        self.D_y.add_loss(loss2)
        self.D_x.compile(optimizer = Adam(2e-4, 0.5))
        self.D_y.compile(optimizer = Adam(2e-4, 0.5))
        #定义 整个模型
        '''
        对输入输出进行解释：
        输入：两张图
        输出：1、判别器输出；2、循环一致性输出；3、期望输出（即 进行风格转换的输出）
        '''
        self.C_gan = Model([img_x,img_y],
                            [valid_x,valid_y,
                            reconstr_x,reconstr_y,
                            translation_x2y,translation_y2x])
        #定义 整个CycleGAN的Loss
        lamda = 0.1
        cyc_loss = K.mean(K.sum(K.abs(translation_x2y - img_x))) + K.mean(K.sum(K.abs(reconstr_y - img_y)))
        total_loss = loss1 + loss2 + lamda * cyc_loss
        #加入Loss并编译
        self.C_gan.add_loss(total_loss)
        self.C_gan.compile(optimizer = Adam(2e-4, 0.5))