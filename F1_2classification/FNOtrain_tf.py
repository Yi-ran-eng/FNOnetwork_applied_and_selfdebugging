# use tf_FNO to train 2 category problem
import tensorflow as tf
from tensorflow import keras
import numpy as np
from dataset_processing import Nanrots
import os,inspect
import pandas as pd
from Intialdots import StringpackedInitial
import random
from Trainertf import FNOtrainer,Frozen
import matplotlib.pyplot as plt
class FNOlayer_tf(keras.layers.Layer):
    def __init__(self,fnonum,twoside=False,**kw):
        super().__init__(**kw)
        self.fnonum=fnonum
        self.couple=twoside
    def build(self,input_shape):
        #input: samples x features
        self.freq = self.add_weight(
            name='freq',
            shape=(self.fnonum,),
            initializer=tf.keras.initializers.Constant(value=1.0),  # 从标准频率开始
            trainable=True
        )
        self.kernel=self.add_weight(
            name='kernel',
            shape=(self.fnonum,input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias=self.add_weight(
            name='bias',
            shape=(self.fnonum,),
            initializer='zeros',
            trainable=True
        )
        self.c_ker=self.add_weight(
            name='ckernal',
            shape=(self.fnonum,input_shape[-1]) if not self.couple else (2*self.fnonum,input_shape[-1]),
            initializer=tf.keras.initializers.VarianceScaling(scale=0.01),
            trainable=1
        )
        self.v_bia=self.add_weight(name='vbias',shape=(input_shape[-1],),initializer='zeros',trainable=True)
        self.featsweight=self.add_weight(name='featwe',shape=(input_shape[-1],input_shape[-1]),
                                         initializer=tf.keras.initializers.VarianceScaling(scale=20),trainable=True)
        self.lambd=self.add_weight(name='lambda',shape=(1,),initializer=tf.keras.initializers.Constant(value=0.1),trainable=True)
        # print(self.v_bia.shape,'???')
        super().build(input_shape)
    def call(self,features):
        self.x_input=features
        self.Z=tf.matmul(features,tf.transpose(self.kernel))+self.bias
        self.Z*=self.freq
        if self.couple:
            self.CoZ=tf.concat([tf.cos(self.Z),tf.sin(self.Z)],axis=1)
        else:
            self.CoZ=tf.cos(self.Z)
        output=0.01*tf.matmul(self.CoZ,self.c_ker) + tf.matmul(features,self.featsweight)+self.v_bia
        self.cache={'inputs':features,'Z':self.Z,'Coz':self.CoZ,'output':output}
        # output=0.01*tf.matmul(self.CoZ,self.c_ker) + tf.matmul(features,self.featsweight)+self.v_bia
        return output
    def get_config(self):
        config=super(FNOlayer_tf,self).get_config()
        config.update({
            'fnonum':self.fnonum,
            'twoside':self.couple
        })
        return config
    @classmethod
    def from_config(cls,config):
        #find the target layers
        return cls(**config)
class SimpleBN(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.gamma =self.add_weight(
            name='gamma', shape=(input_shape[-1],),
            initializer='ones', trainable=True
        )
        self.beta = self.add_weight(
            name='beta', shape=(input_shape[-1],),
            initializer='zeros', trainable=True
        )
        self.eps = 1e-5
    def call(self,inputs, training=None):
        self.inputs=inputs
        self.mu=tf.reduce_mean(inputs,axis=0)
        self.var=tf.reduce_mean(tf.square(inputs-self.mu), axis=0)
        self.sigma=tf.sqrt(self.var+self.eps)
        x_hat=(inputs-self.mu)/self.sigma
        self.x_hat=x_hat
        return self.gamma * x_hat + self.beta

@tf.keras.utils.register_keras_serializable()
class ForwardModel(tf.keras.Model):
    def __new__(cls,*args,**kwargs):
        instance=object().__new__(cls)
        instance._call=True
        return instance
    def __init__(self,outdim,fnolaynum:list,twoside=False,alllayer=7):
        super().__init__()
        sig=inspect.signature(self.__init__)
        self.initparams={
            k:v for k,v in locals().items() if k in sig.parameters and k != 'self'
        }
        #define layers here
        dropout=0.2
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.out,self.fnolist=outdim,fnolaynum
        self.couple=twoside
        self.fnolayer=FNOlayer_tf(fnolaynum[0],twoside)
        self.fnolayers=[self.fnolayer]
        self.indense=keras.layers.Dense(16,activation='relu')
        self.bn_in=SimpleBN()
        self.medense=keras.layers.Dense(16,activation='sigmoid')
        for fno in range(1,len(fnolaynum)):
            self.fnolayers.append(FNOlayer_tf(fnolaynum[fno],twoside))
        self.fnolayers.append(SimpleBN())
        self.outdense=keras.layers.Dense(outdim,activation='sigmoid')
        self.laynum=alllayer
    def call(self,inputs,debugmodel=False,frozen=None,firstmodel=None):
        def changeiner(output,input,cache,name):
            if self._call:
                if len(cache) == self.laynum:
                    self._call=False
                return
            for item in cache:
                if name == item[0]:
                    item[1],item[2]=output,input
                    break
        #set a list to save every layer in order
        if debugmodel:
            if not hasattr(self,'layer_cache'):
                self.runfirst(inputs)
                frozenlayer:list[str]=frozen
                f=Frozen(self.layer_cache)
                self.alive_layers=f.fronzenlayer(frozenlayer,inputs)
            for layer in self.alive_layers:
                if len(layer) == 2:
                    layer.insert(1,inputs)
                else:
                    layer[2]=inputs#input
                inputs=layer[-1](inputs)
                if len(layer) == 3:
                    layer.insert(1,inputs)
                else:
                    layer[1]=inputs#output
            output=inputs
            return output
        self.layer_cache=[]
        input1=self.indense(inputs)
        #[layername,output,input,_,layer]
        if self._call:
            self.layer_cache.append(['indense',input1,inputs,None,self.indense]
                                if not firstmodel else ['indense',self.indense])
        changeiner(input1,inputs,self.layer_cache,'indense')
        input2=self.bn_in(input1)
        if self._call:
            self.layer_cache.append(('bn_0',input2,input1,None,self.bn_in) if 
                                not firstmodel else ['bn_0',self.bn_in])
        changeiner(input2,input1,self.layer_cache,'bn_0')
        input=tf.identity(input2)
        for idx,layer in enumerate(self.fnolayers):
            x_in=input
            input=layer(x_in)
            if idx != len(self.fnolayers)-1:
                if self._call:
                    self.layer_cache.append((f'fnolayer_{idx}',input,x_in,None,layer)
                                        if not firstmodel else [f'fnolayer_{idx}',layer])
                changeiner(input,x_in,self.layer_cache,f'fnolayer_{idx}')
            else:
                if self._call:
                    self.layer_cache.append(('bn_1',input,x_in,None,layer)
                                        if not firstmodel else ['bn_1',layer])
                changeiner(input2,input1,self.layer_cache,'bn_0')
        input3=self.medense(input)
        if self._call:
            self.layer_cache.append(('medense',input3,input,None,self.medense) if not firstmodel 
                                                            else ['medense',self.medense])
        changeiner(input3,input,self.layer_cache,'medense')
        output=self.outdense(input3)
        if self._call:
            self.layer_cache.append(('outdense',output,input3,None,self.outdense)
                                if not firstmodel else ['outdense',self.outdense])
        changeiner(output,input3,self.layer_cache,'outdense')
        return output
    def runfirst(self,inputs):
        self.call(inputs,firstmodel=True)
    def get_config(self):
        config=super().get_config()
        config.update({'outdim':self.out,'fnolaynum':self.fnolist,'twoside':self.couple})
        #use self.params will be more brief
        config.update(self.initparams)
        return config
    @classmethod
    def from_config(cls,config):
        #or use init dict
        if random.random() > 0.5:
            return cls(outdim=config['outdim'],fnolaynum=config['fnolaynum'],
                                twoside=config['twoside'])
        outputdim=config.pop('outdim')
        fnolist=config.pop('fnolaynum',None)
        couple=config.pop('twoside')
        #initparams is one class instance's attribute,not cls's attribute
        return cls(outdim=outputdim,fnolaynum=fnolist,twoside=couple)
def updatep(model,trainer):
    loss_object=tf.keras.losses.MeanSquaredError()
    opt=tf.keras.optimizers.SGD(learning_rate=0.005)
    # model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False))
    # model.fit(features,targs,epochs=maxinter)
    trainloss=tf.keras.metrics.Mean(name='train_loss')
    trainacc=tf.keras.metrics.BinaryAccuracy(name='train_acc')
    # @tf.function
    def trainstep(features,labels):
        output=model(features,debugmodel=True,frozen=['bn_0','bn_1','fnolayer_0','medense'])
        if hasattr(model,'alive_layers'):
            trainer.runall(output,labels,model.alive_layers)
        else:
            trainer.runall(output,labels)
        loss=loss_object(labels,output)
        trainloss.update_state(loss)
        trainacc.update_state(labels,output)
        return loss,output
    return trainstep,trainloss,trainacc,opt
def trainmol(rawdataset,nanpass:bool,trainheads:dict,maxinter,**kw):
    if nanpass:
        mo=Nanrots(rawdataset)
        mo.linear_nan()
        coldir=os.path.dirname(rawdataset)+'/newxlsx.xlsx'
    else:
        coldir=rawdataset
    feats,targs=trainheads['feature'],trainheads['target']
    df=pd.read_csv(coldir)
    df_co=df.copy()
    mo=StringpackedInitial(df_co,targs[0])
    processed=mo.call(df[feats])
    targs=df[targs].to_numpy(dtype=np.float32)
    return processed,targs,mo
def predict_with_preprocessing(raw_df_or_path, feature_cols, preprocessor, modelname):
    """use the same initilazing model to process datas"""
    #load dataset
    if isinstance(raw_df_or_path, str):
        if raw_df_or_path.endswith('.xlsx'):
            df = pd.read_csv(raw_df_or_path)
        else:
            df = pd.read_csv(raw_df_or_path)
    else:
        df = raw_df_or_path
    processed_features = preprocessor.call(df[feature_cols])
    #predict
    reloaded = tf.keras.models.load_model(modelname)
    predictions = reloaded(processed_features)
    return predictions
def _predict(datas:np.ndarray | tf.Tensor,modelname):
    #we are about to using this model to predict some datas
    reloaded=tf.keras.models.load_model(modelname)
    after=reloaded(datas if isinstance(datas,tf.Tensor) else tf.convert_to_tensor(datas))
    return after

model=ForwardModel(outdim=1,fnolaynum=[64,64],twoside=1)
trainer=FNOtrainer(0.000001,model,'mseloss')
trainstep,trainloss,trainacc,opt=updatep(model,trainer)
x,y,preprocessor=trainmol("C:/Users/23322/Downloads/archive (1)/newxlsx.xlsx",False,
         {'feature':['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],'target':['Survived']},maxinter=50)
# print(model(x,debugmodel=True,frozen=['bn_0','bn_1','fnolayer_0','fnolayer_1']))
print(y[:20])
losses=[]
for epoch in range(3):
    trainloss.reset_state()
    trainacc.reset_state()
    _,output=trainstep(x,y)
    losses.append(trainloss.result())
    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}: Loss={trainloss.result():.4f}, ')# Acc={trainacc.result():.4f}')
        tf.print("Pred mean:", tf.reduce_mean(output), 
                "std:", tf.math.reduce_std(output),
                "min:", tf.reduce_min(output), 
                "max:", tf.reduce_max(output))
        
        uncertain = tf.reduce_mean(tf.cast(tf.abs(output - 0.5) < 0.1, tf.float32))
        tf.print("Uncertain ratio:", uncertain)
print(output[:20])
model.save('fnomodel.keras')
plt.plot(losses)
plt.show()
# print("Predictions:", predictions[:20])