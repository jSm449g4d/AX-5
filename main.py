import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm
import random
import argparse
import os

from ARutil import ffzk,mkdiring,rootYrel



def img2np(dir=[],img_len=0):
    img=[]
    for x in dir:
        try:img.append(cv2.imread(x))
        except:continue
        if img_len!=0:img[-1]=cv2.resize(img[-1],(img_len,img_len))
        elif img[-1].shape!=img[0].shape:img.pop(-1);continue#Leave only the same shape
        img[-1] = img[-1].astype(np.float32)/ 256
    return np.stack(img, axis=0)

def tf2img(tfs,dir="./",name="",epoch=0,ext=".png"):
    mkdiring(dir)
    if type(tfs)!=np.ndarray:tfs=tfs.numpy()
    tfs=(tfs*256).astype(np.uint8)
    for i in range(tfs.shape[0]):
        cv2.imwrite(rootYrel(dir,name+"_epoch-num_"+str(epoch)+"-"+str(i)+ext),tfs[i])
        
def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU failed!")
    return len(physical_devices)
tf_ini()
    

class c3c(keras.Model):
    def __init__(self,dim=4):#plz define used layers below...
        super().__init__()
        self.layer1_1=[Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same"),
                       Conv2D(dim,3,padding="same",activation="relu"),
                       Dropout(0.05),
                ]
        self.layer1=[Conv2D(dim,1,padding="same",activation="relu"),
                     Dropout(0.05),
                ]
        return
    def call(self,mod):#plz add layers below...
        mod_1=mod
        for i in range(len(self.layer1_1)):mod_1=self.layer1_1[i](mod_1)
        for i in range(len(self.layer1)):mod=self.layer1[i](mod)
        mod=keras.layers.add([mod,mod_1])
        return mod

#keras.applications.MobileNetV2()
class AE(tf.keras.Model):
    def __init__(self,trials={},opt=keras.optimizers.Adam(1e-3)):
        super().__init__()
        self.seedshape=np.array([1,1,8]).astype(np.int)
        self.layer1=[Conv2D(4,3,2,padding="same",activation="relu"),
                     c3c(4),
                     c3c(6),
                     Conv2D(6,3,2,padding="same",activation="relu"),
                     c3c(6),
                     Conv2D(8,3,2,padding="same",activation="relu"),
                     c3c(8),
                     Conv2D(12,3,2,padding="same",activation="relu"),
                     Conv2D(16,3,2,padding="same",activation="relu"),
                     Dropout(0.05),
                     ]
        self.layer2_1=[c3c(16),
                       c3c(12),
                       c3c(8),
                       Flatten(),
                       Dropout(0.05),
                       Dense(self.seedshape.prod(),activation="sigmoid"),
                       ]
        self.layer2_2=[c3c(16),
                       c3c(12),
                       c3c(8),
                       Flatten(),
                       Dropout(0.05),
                       Dense(self.seedshape.prod(),activation="sigmoid"),
                       ]
        self.layer3=[Reshape(self.seedshape),   
                     UpSampling2D(2),
                     c3c(12),
                     UpSampling2D(2),
                     c3c(16),
                     UpSampling2D(2),
                     c3c(24),
                     UpSampling2D(2),
                     c3c(32),
                     UpSampling2D(2),
                     c3c(36),
                     c3c(36),
                     UpSampling2D(2),
                     c3c(36),
                     c3c(24),
                     Conv2D(3,1,padding="same",activation="sigmoid"),
                     ]
    
    def reparameterize(self, mod, log_var):
        normal = tf.random.normal(tf.shape(log_var))
        return keras.layers.add([mod , normal * tf.exp(log_var * 0.5)])
    
    def call(self,mod):
        print(mod.shape)
        for i in range(len(self.layer1)):mod=self.layer1[i](mod)
        mod2=mod;
        for i in range(len(self.layer2_1)):mod=self.layer2_1[i](mod)
        for i in range(len(self.layer2_2)):mod2=self.layer2_2[i](mod2)
        
        mod=self.reparameterize(mod,mod2)
        
        for i in range(len(self.layer3)):mod=self.layer3[i](mod)
        return mod
    
    def pred(self,batch=1):
        mod=self.reparameterize(np.random.rand(batch,self.seedshape.prod()).astype(np.float32),
                                np.random.rand(batch,self.seedshape.prod()).astype(np.float32))
        for i in range(len(self.layer3)):mod=self.layer3[i](mod)
        return mod

class K_B(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        tf2img(self.model.pred(4),epoch=epoch,dir=os.path.join(args.outdir,"1"))

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=16,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=4,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=5,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    mkdiring(args.outdir)
    img=img2np(ffzk(args.train),64)
    model = AE()
    model.build(input_shape=(args.batch,img.shape[1],img.shape[2],img.shape[3]))
    model.summary()
    model.compile(optimizer =keras.optimizers.Adam(1e-3),
                          loss=keras.losses.binary_crossentropy,
                          metrics=['accuracy'])
    model.fit(img,img,batch_size=args.batch,epochs=args.epoch,
              callbacks=[K_B(),
                         #keras.callbacks.TensorBoard(log_dir=os.path.join(args.outdir,"logs"))
                         ])
    tf2img(img[:args.batch],os.path.join(args.outdir,"0_0"))
    tf2img(model(img[:args.batch]),os.path.join(args.outdir,"0_1"))
    
    