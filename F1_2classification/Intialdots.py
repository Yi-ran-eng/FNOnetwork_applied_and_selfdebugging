import pandas as pd
from scipy.stats import norm
import numpy as np
import tensorflow as tf
from tensorflow import keras
def outer(func):
    def wrapper(self,*args,**kw):
        if kw.get('destination') is not None:
            d=kw.get('destination')
            newkw={k:e for k,e in kw.items() if k != 'destination'}
            newx=func(self,*args,**newkw)
            newx*=d
            return newx
        else:
            return func(self,*args,**kw)
    return wrapper
class normalize_centralize:
    sortedarray=[]
    def __init__(self,*args,**kw):
        if (not kw) and args:
            x=args[0]
            self.newx=np.zeros(x.shape)
        if (not args) and kw:
            x=kw.get('x')
            self.newx=np.zeros(x.shape)

    def backcentral(self,x:np.ndarray):
        self.x=x
        for feat in range(x.shape[1]):
            allx_i=x[:,feat]
            maxx,minx=allx_i.max(),allx_i.min()
            medial=(maxx+minx)/2
            crssed=maxx-minx
            self.newx[:,feat]=(allx_i-medial)/crssed
        return self.newx
    @outer
    def backzero_one(self,x:np.ndarray):
        self.newx=np.zeros(x.shape)
        for feat in range(x.shape[1]):
            allx_f=x[:,feat]
            maxx=allx_f.max()
            self.newx[:,feat]=allx_f/maxx
        return self.newx
    def backBox_Nor(self,x:np.ndarray):
        self.x=x
        filt=x > 0
        assert filt.all(),'input datas must be positive'
        samples=x.shape[0]
        self.sa=samples
        features=x.shape[1]
        p=[]
        for feat in range(x.shape[1]):
            xpiece=x[:,feat].squeeze()
            xsorted=np.sort(xpiece)
            self.sortedarray.append(xsorted)
            for s in range(1,samples+1):
                p.append(s/(samples+1))
            newps=np.array([
                norm.ppf(x) for x in p
            ])
            insetdic={
                xsorted[k]:newps[k] for k in range(samples)
            }#正态分布映射关系
            setattr(self,f'featuredic_{feat-1}',insetdic)
        m=0
        #得到新的数组
        while m < features:
            self.newx[:,m]=np.array(
                [getattr(self,f'featuredic_{m}')[x] for x in x[:,m]]
                )
            m+=1
        return self.newx
    def addnewx_Nor(self,x,featnum:int):
        '''
        featnum给出了是第几个feature
        '''
        if not normalize_centralize.sortedarray:
            raise ValueError('需要先运行backBox_Nor函数以获取已排序的原始数组列表')
        gettarget=normalize_centralize.sortedarray[featnum]
        idx=np.searchsorted(gettarget,x)
        if 1 <= idx <= self.sa-2:
            xraw,xnxt=gettarget[idx],gettarget[idx+1]
            yraw=getattr(self,f'featuredic_{featnum}')[xraw]
            ynxt=getattr(self,f'featuredic_{featnum}')[xnxt]
            ynew=yraw+(x-xraw)/(xnxt-xraw)*(ynxt-yraw)
        elif idx == 0:
            ynew=norm.ppf(1e-4)
        else:
            ynew=norm.ppf(1-1e-4)
        return ynew
class StringpackedInitial(keras.Model):
    def __init__(self,df:pd.DataFrame,trainhead:str):
        super().__init__()
        self.mdf=df.copy()
        self.df=df.copy()
        self.traindata=self.df.pop(trainhead)
        #construct head-Input dict
        self.inputs={}
        self.numerical,self.varistring=[],[]
        for head,col in self.df.items():
            dtype=col.dtype
            try:
                inmifloat=pd.to_numeric(col)
                inmifloat=inmifloat.astype(float)
            except ValueError:
                dtype=tf.string
                self.varistring.append(head)
                pass
            else:
                dtype=tf.float32
                self.df[head]=inmifloat
                self.numerical.append(head)
            self.inputs[head]=keras.Input(shape=(1,),name=head,dtype=dtype)
        self.nmerc=keras.layers.Concatenate()
        self.final=keras.layers.Concatenate()
        self.normalizepart()
        self.encodingpart()
    def normalizepart(self):
        numeric={key:self.inputs[key] for key in self.inputs if self.inputs[key].dtype == tf.float32}
        #use tf.concatentate to concatentate these numeric features
        numeric_cat=tf.keras.layers.Concatenate()(list(numeric.values()))
        self.norm=tf.keras.layers.Normalization()
        self.norm.adapt(np.array(self.df[numeric.keys()]))#this step is to make your norm adapted to your own features
        numericanorm=self.norm(numeric_cat)#then we get the normalized datas in digit
        self.processed=[numericanorm]
    def encodingpart(self):
        assert hasattr(self,'processed'),'run normalizepart first to get self.processed'
        #use tf.keras.layers.DtringLookup to map the string to int index,
        #then use tf.keras.layers.CategoryEncoding to transfer these indexes to proper float32 datas
        self.lookups,self.encodings={},{}
        for name,cols in self.inputs.items():
            if cols.dtype == tf.float32:
                continue
            lookup=keras.layers.StringLookup(vocabulary=np.unique(self.df[name]))#put string array in vocabulary
            self.lookups[name]=lookup
            onehotcoding=keras.layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())
            self.encodings[name]=onehotcoding#use dict to complete mapping relationship
            x=lookup(cols)
            coding_x=onehotcoding(x)
            self.processed.append(coding_x)
    def call(self,inputs:dict | pd.DataFrame=None):
        print('//////////////////////'
        '...................................')
        if inputs is None:
            inputs=self.df.to_dict('list')
            # return keras.layers.Concatenate()(self.processed)
        print('???')
        dicputs=inputs.to_dict('list') if isinstance(inputs,pd.DataFrame) else inputs
        first_key = next(iter(dicputs))
        batch_size = len(dicputs[first_key])
        numeputs = {}
        for nukey in dicputs:
            if nukey in self.numerical:
                # 先转为 numpy float32，确保不是 object 类型
                vals = np.array(dicputs[nukey], dtype=np.float32)
                numeputs[nukey] = tf.convert_to_tensor(vals, dtype=tf.float32)
        stringputs={stkey:[str(x) for x in dicputs[stkey]] for stkey in dicputs if stkey not in self.numerical}
        numercat_inputs = []
        for tensor in numeputs.values():
            if len(tensor.shape) == 1:
                tensor = tf.expand_dims(tensor, axis=-1)  # 变成 [batch_size, 1]
            numercat_inputs.append(tensor)
        numercat=self.nmerc(numercat_inputs)
        normputs=self.norm(numercat)
        processed=[normputs]
        for name in self.varistring:
            if name in stringputs:
                str_vals=stringputs[name]
                xt=tf.convert_to_tensor(str_vals,dtype=tf.string)
                x=self.lookups[name](xt)
                x=tf.expand_dims(x,axis=-1)
                x=self.encodings[name](x)
                processed.append(x)
        #transfer list to tensor
        tensorcat=self.final(processed)
        return tensorcat
# df=pd.read_csv("C:/Users/23322/Downloads/archive (1)/newxlsx.xlsx")
# mo=StringpackedInitial(df,'Survived')
# # 创建测试数据
# test_data = np.array([
#     [10, 20, 30],
#     [5,  15, 25],
#     [2,  8,  12]
# ], dtype=float)

# print("原始数据：")
# print(test_data)
# print()

# # 创建实例
# norm = normalize_centralize(test_data)

# # 测试1：不带 destination 参数（正常归一化）
# result1 = norm.backzero_one(test_data)
# print("测试1 - 不带 destination：")
# print(result1)
# print()

# # 测试2：带 destination 参数（归一化后再乘以 destination）
# result2 = norm.backzero_one(test_data, destination=2.0)
# print("测试2 - 带 destination=2.0：")
# print(result2)
# print("验证：应该是测试1结果的2倍 →", np.allclose(result2, result1 * 2.0))
# print()

# # 测试3：带不同的 destination 值
# result3 = norm.backzero_one(test_data, destination=0.5)
# print("测试3 - 带 destination=0.5：")
# print(result3)
# print("验证：应该是测试1结果的0.5倍 →", np.allclose(result3, result1 * 0.5))