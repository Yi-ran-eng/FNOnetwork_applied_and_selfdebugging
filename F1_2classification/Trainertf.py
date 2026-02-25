import tensorflow as tf
import numpy as np
from typing import Literal,Type

def fnnoupdate(lr,layer:tf.keras.layers.Layer,args):
    #args means parameters' position is of great importance in using this method
    layer.v_bia.assign_sub(lr*args[0])
    layer.featsweight.assign_sub(lr*args[1])
    layer.c_ker.assign_sub(lr*args[2])
    layer.freq.assign_sub(lr*args[3])
    layer.kernel.assign_sub(lr*args[4])
    layer.bias.assign_sub(lr*args[5])
class FNOtrainer:
    def __init__(self,learning_rate,fnomodel,loss_type='binarycrossentropy',clip_drad=None,**kw):
        '''
        customize layer class should be pass in through kw
        '''
        fnolayer=kw.get('fnolayer_cls')
        self.fnomodel=fnomodel
        if fnolayer is not None:
            self.fnolayer=fnolayer
        self.lr=learning_rate
        self.clip_grad=clip_drad
        self.lt=loss_type
    @staticmethod
    def ndimtranform(tens:tf.Tensor,outputdim:int,maplie:Literal['row','col']=None):
        assert tens.ndim <= 3 and outputdim <= 3,"input tensor's dim is so big,only smaller or equal to 3 is allowed"
        if tens.ndim < outputdim:
            #target dim is bigger than existing dim
            newtens=tf.expand_dims(tens,axis=0 if maplie == 'row' else -1)
        elif tens.ndim > outputdim:
            arat=tens.shape.as_list()
            aratensor=tf.convert_to_tensor(arat,dtype=tf.int32)
            if tf.reduce_sum(tf.cast(tf.equal(aratensor,1),tf.int32)) < tens.ndim-outputdim:
                raise ValueError('outputdim is too small')
            if outputdim == 1:
                newtens=tf.squeeze(tens)
            elif outputdim == 2:
                great=tf.cast(tf.greater(aratensor,1),tf.int32)
                if tf.reduce_sum(great) == 2:
                    newtens=tf.squeeze(tens)
                elif tf.reduce_sum(great) == 1:
                    indec=tf.where(tf.greater(aratensor,1))
                    idx=indec[0,0].numpy()
                    if maplie == 'row':
                        newtens=tf.expand_dims(tf.squeeze(tens),axis=0) if \
                        idx == 0 else tf.squeeze(tens,axis=-1 if idx == 1 else 0)
                    else:
                        newtens=tf.expand_dims(tf.squeeze(tens),axis=-1) if \
                        idx == 2 else tf.squeeze(tens,axis=0 if idx == 1 else -1)
        else:
            newtens=tens
        return newtens
    def loss_gradient(self,output:tf.Tensor,target:tf.Tensor):
        outputcop=FNOtrainer.ndimtranform(output,2,'col')
        targetcop=FNOtrainer.ndimtranform(target,2,'col')
        if self.lt == 'binarycrossentropy':
            #this way we use the default value of from_logist:0
            #and it is the start of back propagation
            eps=1e-7
            outputcop=tf.clip_by_value(outputcop,eps,1-eps)
            self.dLdyhat=(outputcop-targetcop)/(outputcop*(1-outputcop))
        elif self.lt == 'mseloss':
            self.dLdyhat=outputcop-targetcop
        return self.dLdyhat
    def dense_gradient(self,output:tf.Tensor,inputx:tf.Tensor,lastgradient:tf.Tensor,
                       actype:Literal['sigmoid','relu'],layer,ndimagust=True):
        outputcop=FNOtrainer.ndimtranform(output,2,'col') if not ndimagust else output
        dactivate=outputcop*(1-outputcop) if actype == 'sigmoid' else tf.cast(outputcop > 0,tf.float32)
        dLdz=lastgradient*dactivate
        W=layer.kernel
        b=layer.bias
        dLdx=tf.matmul(dLdz,tf.transpose(W))
        # print(dLdx.shape,'/x')
        dLdW=tf.matmul(tf.transpose(inputx),dLdz)
        # print(dLdW.shape,'/w')
        dLdb=tf.reduce_sum(dLdz,axis=0)
        # print(dLdb.shape,'?b')
        return dLdx,dLdW,dLdb
    def Bat_Nor_gradient(self,lastgradient,layer):
        '''
        have a quite look at how this layer works:
        out layer:y=gamma*x_hat+beta which means if in the last layer we successfully calculate the loss:dLdx,
        of which x is the output of the bat_nor layer,y,then we can rewrite this loss like it:
        dLdx=dLdy,which will be loaded in through input parameters lastgradient

        what should be computed next is calculating dydx_hat
        dydgamma,dydbeta
        '''
        gama=layer.gamma
        beta,x_hat=layer.beta,layer.x_hat
        mu,var=layer.mu,layer.var
        x_input=layer.inputs
        sigma=layer.sigma
        dydx_hat=gama
        dLdx_hat=lastgradient*dydx_hat
        dLdgamma=tf.reduce_sum(lastgradient*x_hat,axis=0)
        dLdbeta=tf.reduce_sum(lastgradient,axis=0)
        B=tf.cast(tf.shape(x_input)[0],tf.float32)
        dLdvar=tf.reduce_sum(dLdx_hat*(x_input-mu),axis=0)*(-0.5)*tf.pow(sigma,-3)
        dLdmu=tf.reduce_sum(dLdx_hat*(-1/sigma),axis=0)+dLdvar*tf.reduce_mean(-2*(x_input-mu),axis=0)
        dLdx=dLdx_hat/sigma+dLdvar*2*(x_input-mu)/B+dLdmu/B
        return dLdx,dLdgamma,dLdbeta
    def FNO_gradient(self,lastgradient,layer,**kw):
        '''
        params:freq,kernel,bias,c_ker,featsweight,v_bias
        the forward part works like:
        Zlinear=x@kernel+bias -->  Z=Zlinear*freq --> coz(Z)=[cos(Z),sin(Z)] or cos(Z) --> 
        fourierpart=coz@c_ler while residualpart=x@featsweight+v_bias --> y=0.01*fourierpart+residualpart

        lastgradient is dLdy
        '''
        # for name,cacheout,_,layer in self.fnomodel.layer_cache:
        #     layer:tf.keras.layers.Layer
        #     if name == layername:
        kernel=layer.kernel
        freq,c_ker=layer.freq,layer.c_ker
        featswe=layer.featsweight
        throudict=layer.cache
                # break
        dLdfou,dLdfeat=0.01*lastgradient,lastgradient
        dLdvbias=tf.reduce_sum(lastgradient,axis=0)
        #this is features' direction ,which is easier than fno direction
        x=throudict['inputs']
        dLdfeatweight=tf.matmul(tf.transpose(x),dLdfeat)
        dLdx_feat=tf.matmul(dLdfeat,tf.transpose(featswe))
        #this is fno part's back propagation
        dLdcker=tf.matmul(tf.transpose(throudict['Coz']),dLdfou)
        dLdCoz=tf.matmul(dLdfou,tf.transpose(c_ker))
        try:
            z=throudict['Z']
            dLdZ=-dLdCoz*tf.sin(z) if not layer.couple else\
                    -dLdCoz[:,:layer.fnonum]*tf.sin(z)+dLdCoz[:,layer.fnonum:]*tf.cos(z)
        except TypeError:
            print('if twoside is True ,you must pass fnonum into this method')
        dLdfreq=tf.reduce_sum(dLdZ*(tf.matmul(x,tf.transpose(kernel))),axis=0)
        dLdkernel=tf.matmul(tf.transpose(dLdZ*freq),x)
        dLdbias=tf.reduce_sum(dLdZ*freq,axis=0)
        #this is gradient about inputx,act as lastgradient for previous layer
        dLdx=tf.matmul(dLdZ*freq,kernel)+dLdx_feat
        return dLdvbias,dLdfeatweight,dLdcker,dLdfreq,dLdkernel,dLdbias,dLdx
    def runall(self,y_hat:tf.Tensor,target:tf.Tensor,layerdebug=None):
        #layerdebug is alive layer,which is [name,output,input,layer]
        layers:list=self.fnomodel.layer_cache if layerdebug is None else layerdebug
        #we start it by calculating loss item
        grast=self.loss_gradient(y_hat,target)
        for layeritem in reversed(layers):
            layername,output,inputi,layer=layeritem
            if layername.startswith('fno'):
                #means this is a fnolayer
                *grads,grast=self.FNO_gradient(grast,layer)
                fnnoupdate(self.lr,layer,grads)
            elif layername.startswith('bn'):
                grast,dLdgamma,dLdbeta=self.Bat_Nor_gradient(grast,layer)
                layer.gamma.assign_sub(self.lr*dLdgamma)
                layer.beta.assign_sub(self.lr*dLdbeta)
            elif layername.endswith('dense'):
                grast,dLdW,dLdb=self.dense_gradient(output,inputi,grast,layer.activation,layer)
                layer.kernel.assign_sub(self.lr*dLdW)
                layer.bias.assign_sub(self.lr*dLdb)
class Frozen(FNOtrainer):
    def __init__(self,layerlist):
        self.layerlist=layerlist
    def fronzenlayer(self,lis:list[str],input):
        '''
        this lis contains layers'name which are ready to be frozen
        '''
        if hasattr(self,'keypairs'):
            self._clear()
        k=0
        self.keypairs=[]
        for name,layer in self.layerlist:
            if name not in lis:
                self.keypairs.append([name,layer])

        return self.keypairs
    def _clear(self):
        self.keypairs.clear()
    # def forward(self,yhat,target):
    #     self.runall(yhat,target,self.debugsys)