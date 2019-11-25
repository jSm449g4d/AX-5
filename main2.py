#functionalAPI
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import random
import argparse
import os

#from AR9
def mkdiring(input):
    arr=input.split("/");input=""
    for inp in arr:
        if not os.path.exists(input+inp+"/"):os.mkdir(input+inp)
        input+=inp+"/"
    return input.rstrip("/")
def ffzk(input_dir):#Relative directory for all existing files
    imgname_array=[];input_dir=input_dir.strip("\"\'")
    for fd_path, _, sb_file in os.walk(input_dir):
        for fil in sb_file:imgname_array.append(fd_path.replace('\\','/') + '/' + fil)
    if os.path.isfile(input_dir):imgname_array.append(input_dir.replace('\\','/'))
    return imgname_array

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
        cv2.imwrite(os.path.join(dir,name+"_epoch-num_"+str(epoch)+"-"+str(i)+ext),tfs[i])
        
def tf_ini():#About GPU resources
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
    if len(physical_devices)==0:print("GPU failed!")
    return len(physical_devices)
tf_ini()
    

class c3c():
    def __init__(self,dim=4):#plz define used layers below...
        self.dim=dim
        return
    def __call__(self,mod):#plz add layers below...
        with tf.name_scope("c3c"):
            mod_1=mod
            mod=Conv2D(self.dim,1,padding="same")(mod)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod_1=Conv2D(self.dim,3,padding="same")(mod_1)
            mod=keras.layers.add([mod,mod_1])
            mod=Dropout(0.05)(mod)
            mod=Activation("relu")(mod)
        return mod

def DEC(input_shape=(64,64,3,),dim_h=8):
    mod=mod_inp = Input(shape=input_shape)
    mod=Conv2D(4,3,2,padding="same",activation="relu")(mod)
    mod=c3c(4)(mod)
    mod=c3c(6)(mod)
    mod=Conv2D(6,3,2,padding="same",activation="relu")(mod)
    mod=c3c(6)(mod)
    mod=Conv2D(12,3,2,padding="same",activation="relu")(mod)
    mod=c3c(16)(mod)
    mod=Conv2D(24,3,2,padding="same",activation="relu")(mod)
    mod=Conv2D(32,3,2,padding="same",activation="relu")(mod)
    mod=Dropout(0.05)(mod)
    mod_1=mod
    #mu
    mod=c3c(16)(mod)
    mod=c3c(12)(mod)
    mod=Flatten()(mod)
    mod=Dropout(0.05)(mod)
    mod=Dense(dim_h,activation="relu")(mod)
    #sigma
    mod_1=c3c(16)(mod_1)
    mod_1=c3c(12)(mod_1)
    mod_1=Flatten()(mod_1)
    mod_1=Dropout(0.05)(mod_1)
    mod_1=Dense(dim_h,activation="relu")(mod_1)
    mod=keras.layers.add([mod , tf.random.normal(tf.shape(mod_1)) * tf.exp(mod_1 * 0.5)])
    return keras.models.Model(inputs=mod_inp, outputs=mod)

def ENC(dim_h=8):
    mod=mod_inp = Input(shape=(dim_h,))
    mod=Reshape((1,1,dim_h))(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(12)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(16)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(24)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(32)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(36)(mod)
    mod=c3c(48)(mod)
    mod=c3c(36)(mod)
    mod=UpSampling2D(2)(mod)
    mod=c3c(24)(mod)
    mod=c3c(24)(mod)
    mod=c3c(12)(mod)
    mod=Conv2D(3,1,padding="same",activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)
    
    

class K_B(keras.callbacks.Callback):
    def __init__(self,enc,dec,dim_h=8,batch_pred=8):
        super().__init__()
        self.enc=enc;self.dec=dec;self.dim_h=dim_h;self.batch_pred=batch_pred
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(args.outdir,"mod.h5"))
        tf2img(self.enc(np.random.rand(self.batch_pred,self.dim_h).astype(np.float32))
        ,epoch=epoch,dir=os.path.join(args.outdir,"1"))
        
        

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train' ,help="train_data",default="./lfw")
parser.add_argument('-o', '--outdir' ,help="outdir",default="./output")
parser.add_argument('-b', '--batch' ,help="batch",default=32,type=int)
parser.add_argument('-p', '--predbatch' ,help="batch_size_of_prediction",default=8,type=int)
parser.add_argument('-e', '--epoch' ,help="epochs",default=50,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    mkdiring(args.outdir)
    img=img2np(ffzk(args.train),64)
    
    dec=DEC()
    enc=ENC()
    
    mod=mod_inp = Input(shape=(64,64,3,))
    mod=enc(dec(mod))
    model =keras.models.Model(inputs=mod_inp, outputs=mod)
    
    
    model.build(input_shape=(args.batch,img.shape[1],img.shape[2],img.shape[3]))
    model.summary()
    model.compile(optimizer =keras.optimizers.Adam(1e-4),
                          loss=keras.losses.binary_crossentropy,
                          metrics=['accuracy'])
    
    try:model.load_weights(os.path.join(args.outdir,"mod.h5"))
    except:print("\nCannot_use_savedata...")
    mkdiring("logs")#error : os.path.join(args.outdir,"logs/")
    model.fit(img,img,batch_size=args.batch,epochs=args.epoch,
              callbacks=[K_B(enc,dec),
                         keras.callbacks.TensorBoard(log_dir="logs") ,
                         ])
    tf2img(img[:args.batch],os.path.join(args.outdir,"0_0"))
    tf2img(enc(img[:args.batch]),os.path.join(args.outdir,"0_1"))
    
    