import pandas as pd
import os
import numpy as np

class Nanrots:
    def __init__(self,filexlsx,outputpath=None):
        #first check if the filecxlsx is ended with .csv
        if outputpath is None:
            outputpath=os.path.dirname(filexlsx)+'/newxlsx.xlsx'
        self.out=outputpath
        self.xlsx=filexlsx
        _name=os.path.basename(filexlsx).split('.')[1]
        self.name=_name
        if _name == 'xlsx':
            self.df=pd.read_excel(filexlsx)
        else:
            self.df=pd.read_csv(filexlsx)
        self.dataset=self.df.to_numpy()
    @staticmethod
    def _cluster(array):
        cluster=[]
        opo=[]
        k=0
        while k < len(array):
            if (opo and array[k] == opo[-1]+1) or len(opo) == 0:
                opo.append(array[k])
            else:
                cluster.append(opo)
                opo=[]
                opo.append(array[k])
            k+=1
        if opo:
            cluster.append(opo)
        for clus in cluster:
            yield clus
    @staticmethod
    def _inarray(ind,array):
        if 0 <= ind < len(array):
            return True
        return False
    def linear_nan(self):
        self.dataset=self.dataset.reshape(-1,1) if self.dataset.ndim == 1 else self.dataset
        getcolnumber,getrownumber=self.dataset.shape[1],self.dataset.shape[0]
        for i in range(getcolnumber):
            try:
                '''
                this piece of code is trying to adjust the type of arrays into float while changing the value into nan 
                when encounting with non_digital value
                '''
                inmifloat=pd.to_numeric(self.dataset[:,i])#,errors='coerce')
                inmifloat=inmifloat.astype(float)
            except ValueError:
                continue
            else:
                self.dataset[:,i]=inmifloat#directly change the type of raw dataset
                nan_ind=np.argwhere(np.isnan(inmifloat))
                # print(nan_ind,'?')
                if nan_ind.size != 0 and len(nan_ind) < getrownumber:
                    print(nan_ind,'?')
                    #indicating that there are nans in this array
                    for clusind in Nanrots._cluster(nan_ind[:,0]):
                        firstind,endind=clusind[0],clusind[-1]
                        print(firstind,endind)
                        if Nanrots._inarray(firstind-1,self.dataset[:,i]):
                            stnum=self.dataset[:,i][firstind-1]
                        else:
                            #means the first number is nan,we
                            #start find the first non_nan number from the last number
                            mini=-1
                            while np.isnan(self.dataset[:,i][mini]):
                                mini-=1
                            stnum=self.dataset[:,i][mini]
                        if Nanrots._inarray(endind+1,self.dataset[:,i]):
                            ennum=self.dataset[:,i][endind+1]
                        else:
                            mani=0
                            while np.isnan(self.dataset[:,i][mani]):
                                mani+=1
                            ennum=self.dataset[:,i][mani]
                        inarray=np.linspace(stnum,ennum,len(clusind)+2)
                        self.dataset[:,i][firstind:endind+1]=inarray[1:-1]
        #save dataset
        newdf=pd.DataFrame(self.dataset,columns=self.df.columns)
        if self.name == 'xlsx':
            newdf.to_excel(self.out,index=False)
        else:
            newdf.to_csv(self.out,index=False,encoding='utf-8-sig')
        print('ok')
    def _stringnan(self):
        for head,col in self.df.items():
            if col.dtype == object and col.hasnans:
                col_filled=col.fillna('unknown')
                self.df[head]=col_filled
        diname=os.path.dirname(self.out)+'/strfilled.xlsx'
        self.df.to_excel(diname,index=False)
        self.dataset=self.df.to_numpy()

# mo=Nanrots("C:/Users/23322/Downloads/archive (1)/train.csv")
# mo._stringnan()
# mo.linear_nan()