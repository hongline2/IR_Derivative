'''
Smith-Wilson Yield Curve Fitting Implementation
Using pandas and numpy or just datetime or string
하지만 일단 pandas 를 사용하자 편하니까
'''

import numpy as np 
import pandas as pd 
import datetime

import matplotlib.pyplot as plt 
#zero coupon input
#[
#[가격, 만기일(datetime.datetime)],
#[가격, 만기],
#]
# 가격은 zero copupon 가격 그대로. 만기는 datetime 일단 쓰자
class Smith_Wilson_Discount_Curve:
    def __init__(self):
        self.is_fitted = False 
        self.today=datetime.datetime(datetime.datetime.today().year,datetime.datetime.today().month,datetime.datetime.today().day)
        self.alpha = 0.1
        self.ufr=0.023
        self.zero_coupon_input=None 
        self.general_input=None 
        self.zetas = None
        pass 

    def set_market_data_zerocoupon(self,df):
        #날자전처리해서 ttm으로 바꿔놔야함
        for i in range(len(df)):
            df[i][1] = (df[i][1] - self.today).days / 365
        self.zero_coupon_input = np.array(df)
    
    def set_market_data_general(self,df):
        pass 
    
    def set_alpha(self,alpha):
        self.alpha=alpha
    
    def set_ufr(self,ufr):
        self.ufr=ufr
    
    def fit(self):
        if self.zero_coupon_input is not None:
            self._fit_zerocoupon()
            self.is_fitted=True
        elif self.general_input is not None:
            self._fit_general()
            self.is_fitted=True
        else:
            print('There is no market data. try .set_market_data_zerocoupon(df) or .set_market_data_general(df) first')
        
 

    def _fit_zerocoupon(self):
        prices = self.zero_coupon_input[:,0].copy()
        prices=prices.reshape(-1,1)
        ttms=self.zero_coupon_input[:,1].copy()
        ttms=ttms.reshape(-1,1)
        t_mat= np.dot(ttms,np.ones( ( 1,len(ttms) ) ) )
        u_mat=t_mat.copy()
        u_mat=u_mat.T

        temp1=np.exp(-1*self.ufr*(t_mat + u_mat))
        temp2=self.alpha * np.minimum(t_mat,u_mat)
        temp3=-0.5*np.exp ( np.exp(-1*self.alpha*np.maximum(t_mat,u_mat)))
        temp4=np.exp(self.alpha*np.minimum(t_mat,u_mat)) - np.exp(-1*self.alpha*np.minimum(t_mat,u_mat))

        w_mat= temp1*(temp2-temp3*temp4)
        mus=np.exp(-self.ufr*ttms)

        self.zetas=np.linalg.solve(w_mat,prices-mus)
        #식에서 t mat은 ttm,ones u mat은 ttm,ones.T

        
    def _fit_general(self):
        pass

    def _vectorized_w_zerocoupon(self,datetime_list):
        #list든지, (n,)든지, (n,n)
        pass 






    def df(self,t):
        if not self.is_fitted :
            print('Curve is not fitted yet. Try .fit() first.')
            return 
        if self.zero_coupon_input is not None:
            return self._df_zero(t)
        elif self.general_input is not None:
            return self._df_general(t)

    def _df_zero(self,t): # 뭔가 계산 오류 있다. 다시확인할것

        if type(t) == datetime.datetime:
            t= (t-self.today).days / 365
        u=self.zero_coupon_input[:,1].reshape(-1,1)
        temp1=np.exp(-1*self.ufr*(u+t))
        temp2=self.alpha*np.minimum(t,u)
        temp3=0.5*np.exp(-1*self.alpha*np.maximum(t,u))
        temp4=np.exp(self.alpha*np.minimum(t,u)) - np.exp(-1*self.alpha*np.minimum(t,u))
        w=temp1*(temp2-temp3*temp4)

        
        return np.exp(-1*self.ufr*t) + (self.zetas*w).sum()       


    
    def _df_general(self,t):
        if type(t) == datetime.datetime:
            t= (t-self.today).days / 365

        pass 




    def df_curve(self,maturity=100):
        if not self.is_fitted:
            print('curve is not fitted yet. Try .fit() first.')
            return 

        if self.zero_coupon_input is not None:
            return self._df_curve_zero(maturity)
        
    def _df_curve_zero(self,maturity=100):
        # 최대한 vectorize해서 전체커브 돌려주는방법으로.
        maturity_span=np.linspace(0,maturity,maturity*12).reshape(1,-1) # maturity_span : 1,1200
        u_matrix = np.dot(np.ones((maturity*12,1)),self.zero_coupon_input[:,1].reshape(1,-1)) # ttms : u1~u6
        maturity_matrix = np.dot(maturity_span.reshape(-1,1) , np.ones((1,len(self.zetas))) )
        
        zetas_matrix = self.zetas * np.ones((self.zetas.shape[0],maturity_span.shape[1]))


        temp1=np.exp(-1*self.ufr*(maturity_matrix + u_matrix))
        temp2=self.alpha * np.minimum(maturity_matrix,u_matrix)
        temp3=-0.5*np.exp ( np.exp(-1*self.alpha*np.maximum(maturity_matrix,u_matrix)))
        temp4=np.exp(self.alpha*np.minimum(maturity_matrix,u_matrix)) - np.exp(-1*self.alpha*np.minimum(maturity_matrix,u_matrix))

        w_mat=temp1*(temp2-temp3*temp4)

        return np.exp(-1*self.ufr * maturity_span).T + np.dot(w_mat,self.zetas)


zci=[
[0.99,datetime.datetime(2022,5,31)],
[0.975,datetime.datetime(2023,5,31)],
[0.945,datetime.datetime(2025,5,31)],
[0.85,datetime.datetime(2032,5,31)],
[0.72,datetime.datetime(2042,5,31)],
[0.50,datetime.datetime(2052,5,31)],
]

swdf=Smith_Wilson_Discount_Curve()
swdf.set_market_data_zerocoupon(zci)
swdf.fit()

plt.plot(swdf.df_curve())
plt.show()


'''
general_input=[
    [[][][][]]
    [[][][][]]
    [[][][][]]
    [[][][][]]
    [[][][][]]
    [[][][][]]
]
'''

swdf_general=Smith_Wilson_Discount_Curve()
swdf.set_market_data_general(general_input)
swdf.fit()

plt.plot(swdf.df_curve())
plt.show()