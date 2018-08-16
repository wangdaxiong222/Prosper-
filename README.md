# Prosper-
这里我会贴上
一、背景介绍
Prosper是美国的一家P2P（个人对个人）借贷网站，网站撮合了一些有闲钱的人和一些急于用钱的人双方的需求。
二、项目内容介绍
本文全部数据来源于kaggle平台 Prosper Loan Data | Kaggle，是平台上一个供感兴趣的人分析的实例项目，并非一个竞赛项目。

三、数据特征
数据集有81个特征，若需详细了解特征含义，请点击prosper loan data EDA分析（特征字典）

四、数据预处理
加载库：
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
加载数据：
loanData=pd.read_csv('D:/kaggle/prosper/prosperLoanData.csv')
4.1 LoanStatus数据转换，设置Status特征
loanData.groupby(['LoanStatus'])['LoanStatus'].count()

平台把借款状态分为12种，变换后分成四组：
Current(包含Current)
Defaulted(包含Defaulted、Chargedoff)
Completed(包含Completed、FinalPaymentInProgress、Past Due六种)
Cancelled（包含Cancelled）。Cancelled的只有5笔，可直接去掉。代码如下：
def loan_status(s):
    if s=='Current':
        d='Current'
    elif s=='Cancelled':
        d='Cancelled'
    elif s=='Defaulted':
        d='Defaulted'
    elif s=='Chargedoff':
        d='Defaulted'
    else:
        d='Completed' 
    return d
loanData['Status']=loanData['LoanStatus'].apply(loan_status)
loanData=loanData[loanData['Status']!='Cancelled']
4.2 CreditScoreRangeUpper和CreditScoreRangeLower数据转换，设置CreditScore特征

将这两个特征取平均值做计算：
loanData['CreditScore']=((loanData['CreditScoreRangeUpper']+loanData['CreditScoreRangeLower'])/2).round(0)
4.3 BankcardUtilization数据转换，设置BankCardUse特征
本文将这个数据分成五组，对之前未在prosper的客户建立库，即其0或NA是未使用过prosper的客户，用no use代替。代码如下：
def bank_card_use(s,oneForth = 0.31,twoForth = 0.6):
    if s<=oneForth:
        b='Mild Use'
    elif (s>oneForth) & (s<=twoForth):
        b='Medium Use'
    elif (s>twoForth) & (s<=1):
        b='Heavy Use'
    elif s>1:
        b='Super Use'
    else:
        b='No Use'
    return b
oneFourth=loanData['BankcardUtilization'].quantile(0.25)
twoFourth=loanData['BankcardUtilization'].quantile(0.5)
loanData['BankCardUse']=loanData['BankcardUtilization'].apply(bank_card_use)
4.4 DelinquenciesLast7Years数据转换，设置Delinquencies特征
Delinquencies特征只有两种结果：Delinquencies代表过去七年违约一次或以上，NoDelinquencies代表过去七年没有违约过。
def Delin(s):
    if s>0:
        d='Delinquencies'
    else:
        d='NoDelinquencies'
    return d
loanData['Delinquencies']=loanData['DelinquenciesLast7Years'].apply(Delin)
4.5 LoanOriginationDate数据转换，设置DatePhase特征
因为2009年7月1日是一个数据截点，因此将数据分成两段处理，代码如下：
def Data_Phase(s):
    if s>='2009-07-01':
        d='After Jul.2009'
    else:
        d='Before Jul.2009'
    return d
loanData['DataPhase']=loanData['LoanOriginationDate'].apply(Data_Phase)
4.5 TotalProsperLoans数据转换，设置CustomerClarify特征
对于TotalProsperLoans，我们可以根据数量区分新老用户，0或NA是未使用过prosper的客户,反之是使用过的，代码如下：
def customer_clarify(s):
    if s>0:
        d='Previous Borrower'
    else:
        d='New Borrower'
    return d
loanData['CustomerClarify']=loanData['TotalProsperLoans'].apply(customer_clarify)

五、探索性分析
5.1 借款人消费信用分(CreditScore)越低，违约概率越大
消费信用分衡量一个人在消费中的经济能力，分值高的人交易更活跃、交易活动违约率更低。
matplotlib.rcParams['font.sans-serif'] = ['FangSong']    # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False

loanData1=loanData[loanData['Status']!='Current']
creditScore=loanData1.groupby(['Status','CreditScore'])['Status'].count().unstack(0)
score=pd.DataFrame(creditScore.values[2:],index=creditScore.index[2:],columns=creditScore.columns)
print(score)
index=list(score.index)
fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(111)

score.plot(kind='line',ax=ax1,grid=True)
ax1.set_ylabel('数量')
plt.legend(loc='best')
plt.show()
如图所示，CreditScore < 560的借款人中，违约笔数大于未违约的笔数。从上图也可看出大部分借款人的消费信用分高于600分，说明消费信用分低的人在平台上不容易借到钱。

5.2 信用卡透支比例（BankCardUse）越高，违约概率越大
获取信用卡使用情况和违约情况,代码如下：
bankCardUse=loanData1.groupby(['Status','BankCardUse'])['Status'].count().unstack(0)
index=['Mild Use','Medium Use', 'Heavy Use', 'Super Use', 'No Use',]
bankCardUse=bankCardUse.reindex(index)

fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
bankCardUse.plot(kind='bar',ax=ax1)
ax1.set_xticklabels(index,rotation=0)
ax1.set_ylabel('数量')

defaultRate=(bankCardUse['Defaulted']/(bankCardUse['Defaulted']+bankCardUse['Completed'])).reindex(index)
y=list(defaultRate.values)
ax2=fig.add_subplot(212)
x=np.arange(len(index))+1
ax2.bar(x,y,width=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(index,rotation=0)
ax2.set_ylabel('违约率')
ax2.set_xlabel('bankCardUse')
plt.show()
如图所示，BankCardUse为Mild Use、Medium Use、Heavy Use时违约率在25%左右，而Super Use、No Use违约率在50%左右，所以，对这两种情况的借款人要加强风险管控。

5.3 年收入（IncomeRange）越低，违约概率越大
统计出每个年收入段贷款如期还款笔数和违约笔数，并计算出每段贷款违约笔数所占的比例。代码如下：
incomeRage=loanData1.groupby(['Status','IncomeRange'])['Status'].count().unstack(0)
index=['Not displayed','Not employed','$0 ', '$1-24,999', '$25,000-49,999', '$50,000-74,999', '$75,000-99,999', '$100,000+']
incomeRage=incomeRage.reindex(index)

fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
incomeRage.plot(kind='bar',ax=ax1)
ax1.set_ylabel('数量')
ax1.set_xticklabels(index,rotation=0)

defaultedRate=(incomeRage['Defaulted']/(incomeRage['Defaulted']+incomeRage['Completed'])).reindex(index)
y=list(defaultedRate.values)
x=np.arange(len(index))+1
ax2=fig.add_subplot(212)
ax2.bar(x,y,width=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel('违约率')
plt.show()
如图所示，年收入越低，违约概率越大。

5.4 负债水平（DebtToIncomeRatio）越高，违约概率越大
代码如下：
CompletedRatio=loanData1[loanData1['Status']=='Completed']['DebtToIncomeRatio']
DefaultedRatio=loanData1[loanData1['Status']=='Defaulted']['DebtToIncomeRatio']

fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(111)
CompletedRatio.hist(bins=1000,color='g',label='Compeleted')
DefaultedRatio.hist(bins=1000,color='b',label='Defaulted')
ax1.set_xlim([0,1])
ax1.set_xlabel('DebtToIncomeRatio')
ax1.set_ylabel('数量')
plt.legend(loc='best')
plt.show()
如图所示，DebtToIncomeRatio < 0.6的借款人中，违约笔数小于未违约的笔数。从下图也可看出大部分借款人的债务收入比低于0.25，说明平台违约的整体风险可控。

5.5 过去七年违约次数（DelinquenciesLast7Years）越多，违约概率越大
过去七年违约次数能够衡量一个人在过去七年中征信情况，代码如下：
delinquenciesLast7Years=loanData1.groupby(['Status','DelinquenciesLast7Years'])['Status'].count().unstack(0)
delinquenciesLast7Years26=pd.DataFrame(delinquenciesLast7Years.values[0:26],index=delinquenciesLast7Years.index[0:26],columns=delinquenciesLast7Years.columns)

fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
delinquenciesLast7Years26.plot(kind='line',ax=ax1,grid=True)
ax1.set_ylabel('数量')

delinquencies=loanData1.groupby(['Status','Delinquencies'])['Status'].count().unstack(0)
index=['Delinquencies','NoDelinquencies']
ax2=fig.add_subplot(212)
delinquencies.plot(kind='bar',ax=ax2)
ax2.set_xticklabels(index,rotation=0)
ax2.set_ylabel('数量')
ax2.set_xlabel('过去七年有无违约')
plt.show()
如图所示，可看出大部分借款人的DelinquenciesLast7Years 在1次以下，说明整个平台的风险可控。同时可得出过去七年违约一次或以上的人在借款时违约概率更大，过去七年没有违约过的人违约概率低。

5.6 受雇佣状态持续时间（EmploymentStatusDuration）越短，违约概率越大
受雇佣状态持续时间可够衡量一个人工作生活的稳定情况。代码如下：
employmentStatusDuration=loanData1.groupby(['Status','EmploymentStatusDuration'])['Status'].count().unstack(0)
employmentStatusDuration0to120=pd.DataFrame(employmentStatusDuration.values[0:120],index=employmentStatusDuration.index[0:120],columns=employmentStatusDuration.columns)
fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
employmentStatusDuration0to120.plot(kind='line',ax=ax1,grid=True)
ax1.set_ylabel('数量')

defaultedRate=employmentStatusDuration0to120['Defaulted']/(employmentStatusDuration0to120['Defaulted']+employmentStatusDuration0to120['Completed'])
y=list(defaultedRate.values)

ax2=fig.add_subplot(212)
ax2.plot(defaultedRate.index,defaultedRate.values,'ko--')
ax2.set_xlabel('受雇佣状态持续时间(月)', fontsize=14)
ax2.set_ylabel('违约百分率(%)', fontsize=14)
plt.legend(loc='best')
plt.show()
如图所示，受雇佣状态持续时间越短，违约概率越大。


5.7 2009.07.01之前的情况，信用评级（CreditGrade）越低，违约概率越大
获取2009年7月之前信用等级情况和违约情况，代码如下：
creditGrade=loanData1.groupby(['Status','CreditGrade'])['Status'].count().unstack(0)
index=[ 'NC','HR','E','D', 'C','B', 'A', 'AA']
creditGrade=creditGrade.reindex(index)
fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
creditGrade.plot(kind='bar',ax=ax1)
ax1.set_xticklabels(index,rotation=0)
ax1.set_ylabel('数量')

defaultedRate=creditGrade['Defaulted']/(creditGrade['Defaulted']+creditGrade['Completed']).reindex(index)
y=list(defaultedRate.values)
x=np.arange(len(index))+1
ax2=fig.add_subplot(212)
ax2.bar(x,y,width=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel('违约率')
ax2.set_xlabel('creditGrade')
for a, b in zip(x, y):
        plt.text(a, b + 0.001, '%.2f%%' % (b * 100), ha='center', va='bottom')
plt.show()
如图所示，信用评级越低，违约概率越大。大部分借款人的评级都在D级以上。

5.8 2009.07.01之后的情况，信用评级（ProsperRating (Alpha)）越低，违约概率越大
ProsperRating (Alpha)是2009年之后的信用等级特征，获取2009年7月之后信用等级情况和违约情况，代码如下：
ProsperRating=loanData1.groupby(['Status','ProsperRating (Alpha)'])['Status'].count().unstack(0)
index=[ 'HR','E','D','C','B',  'A', 'AA']
ProsperRating=ProsperRating.reindex(index)
fig=plt.figure()
fig.set_size_inches(16,8)
ax1=fig.add_subplot(211)
ProsperRating.plot(kind='bar',ax=ax1)
ax1.set_xticklabels(index,rotation=0)
ax1.set_ylabel('数量')

defaultedRate=ProsperRating['Defaulted']/(ProsperRating['Defaulted']+ProsperRating['Completed'])
x=np.arange(len(index))+1
y=list(defaultedRate.values)
ax2=fig.add_subplot(212)
ax2.bar(x,y,width=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(index,rotation=0)
ax2.set_ylabel('违约率')
ax2.set_xlabel('ProsperRating (Alpha)')
for a, b in zip(x, y):
        plt.text(a, b + 0.001, '%.2f%%' % (b * 100), ha='center', va='bottom')

plt.show()
如图所示，信用评级越高的人违约率越低，大部分借款人的评级都在D级以上。与2009.07.01之前的信用评级对比去掉了NC级，且整体违约率一一对比，都比之前更低，说明2009年7月1日之后，平台的风控模型进行了非常有成效的调整。


六、缺失值处理
获取缺失情况，代码如下：
missing=pd.concat((loanData.isnull().any(),loanData.count()),axis=1)
missing.columns=['是否缺失','数量']
max=missing['数量'].max()
missing['缺失数量']=max-missing['数量']
missing['缺失率']=missing['缺失数量']/max
miss=missing[missing['数量']<max]
miss
得到缺失情况如图所示（后面部分省略）：

6.1 CreditScore缺失值处理

缺失了590条，所占比例约为0.5%左右，所占比例不大，暂时用中位数进行替换。
loanData['CreditScore']=loanData['CreditScore'].fillna(loanData['CreditScore'].median())
6.2 DebtToIncomeRatio缺失值处理
缺失值为8554条，所占比例很大，且违约比例较大，依据常识债务比例数值越大违约的概率越大，所以根据数据集中债务比例分布情况将0.10~0.50随机赋值给缺失值。代码如下：
def rand_missing(s):
    if s>=0:
        d=s
    else:
        d=random.uniform(0.1,0.5)
    return d
loanData['DebtToIncomeRatio']=loanData['DebtToIncomeRatio'].apply(rand_missing)
6.3 DelinquenciesLast7Years缺失值处理
缺失值为987条，在平台借款违约比例较大，所以将缺失值全部置为1。
loanData['DelinquenciesLast7Years']=loanData['DelinquenciesLast7Years'].fillna(1)
6.4 EmploymentStatusDuration缺失值处理
缺失值为7621条，所占比例很大，且违约比例很大。猜想工作越稳定还款能力越强，故将缺失值置为48。
loanData['EmploymentStatusDuration']=loanData['EmploymentStatusDuration'].fillna(48)
6.5 CreditGrade缺失处理
missIndex2=loanData[(loanData['CreditGrade'].isnull())& (loanData['DataPhase']=='Before Jul.2009') ]
missIndex2

得到结果如图，表面看缺失值有84984条，实际上CreditGrade无缺失值，不用进行填充
6.6 ProsperRating (Alpha)缺失处理
在2009年之后，大约有3万条，筛选出该特征的缺失值，大约有144条，所占比例较小，144/29079 =0.49%，所以直接删除这144条缺失的数据，代码如下：
missIndex=loanData[(loanData['ProsperRating (Alpha)'].isnull()) & (loanData['DataPhase']=='After Jul.2009')]
loanData=loanData.drop(missIndex.index,axis=0)
6.7 InquiriesLast6Months缺失值处理
缺失值为696条，所占比例不大，违约比例跟整体数据相近，故将它的缺失值置为2。
loanData['InquiriesLast6Months'] = loanData['InquiriesLast6Months'].fillna(2)
七、建模分析

7.1 字符串变量转换成数字变量

因为后面模型要用到随机森林模型，而数据中存在字符串变量，故将其用数字变量进行替换。
loanData.loc[loanData['IsBorrowerHomeowner'] == False, 'IsBorrowerHomeowner'] = 0
loanData.loc[loanData['IsBorrowerHomeowner'] == True, 'IsBorrowerHomeowner'] = 1
    
loanData.loc[loanData['CreditGrade'] == 'NC', 'CreditGrade'] = 0
loanData.loc[loanData['CreditGrade'] == 'HR', 'CreditGrade'] = 1
loanData.loc[loanData['CreditGrade'] == 'E', 'CreditGrade'] = 2
loanData.loc[loanData['CreditGrade'] == 'D', 'CreditGrade'] = 3
loanData.loc[loanData['CreditGrade'] == 'C', 'CreditGrade'] = 4
loanData.loc[loanData['CreditGrade'] == 'B', 'CreditGrade'] = 5
loanData.loc[loanData['CreditGrade'] == 'A', 'CreditGrade'] = 6
loanData.loc[loanData['CreditGrade'] == 'AA', 'CreditGrade'] = 7
   
loanData.loc[loanData['ProsperRating (Alpha)'] == 'HR', 'ProsperRating (Alpha)'] = 1
loanData.loc[loanData['ProsperRating (Alpha)'] == 'E', 'ProsperRating (Alpha)'] = 2
loanData.loc[loanData['ProsperRating (Alpha)'] == 'D', 'ProsperRating (Alpha)'] = 3
loanData.loc[loanData['ProsperRating (Alpha)'] == 'C', 'ProsperRating (Alpha)'] = 4
loanData.loc[loanData['ProsperRating (Alpha)'] == 'B', 'ProsperRating (Alpha)'] = 5
loanData.loc[loanData['ProsperRating (Alpha)'] == 'A', 'ProsperRating (Alpha)'] = 6
loanData.loc[loanData['ProsperRating (Alpha)'] == 'AA', 'ProsperRating (Alpha)'] = 7
  
loanData.loc[loanData['IncomeRange'] == 'Not displayed', 'IncomeRange'] = 0
loanData.loc[loanData['IncomeRange'] == 'Not employed', 'IncomeRange'] = 1
loanData.loc[loanData['IncomeRange'] == '$0 ', 'IncomeRange'] = 2
loanData.loc[loanData['IncomeRange'] == '$1-24,999', 'IncomeRange'] = 3
loanData.loc[loanData['IncomeRange'] == '$25,000-49,999', 'IncomeRange'] = 4
loanData.loc[loanData['IncomeRange'] == '$50,000-74,999', 'IncomeRange'] = 5
loanData.loc[loanData['IncomeRange'] == '$75,000-99,999', 'IncomeRange'] = 6
loanData.loc[loanData['IncomeRange'] == '$100,000+', 'IncomeRange'] = 7

loanData.loc[loanData['CustomerClarify'] == 'New Borrower', 'CustomerClarify'] = 0
loanData.loc[loanData['CustomerClarify'] == 'Previous Borrower', 'CustomerClarify'] = 1
loanData.loc[loanData['Status']=='Defaulted','Status']=0
loanData.loc[loanData['Status']=='Completed','Status']=1
loanData.loc[loanData['Status']=='Current','Status']=2

loanData.loc[loanData['BankCardUse'] == 'No Use', 'BankCardUse'] = 0
loanData.loc[loanData['BankCardUse'] == 'Mild Use', 'BankCardUse'] = 1
loanData.loc[loanData['BankCardUse'] == 'Medium Use', 'BankCardUse'] = 2
loanData.loc[loanData['BankCardUse'] == 'Heavy Use', 'BankCardUse'] = 3
loanData.loc[loanData['BankCardUse'] == 'Super Use', 'BankCardUse'] = 4

7.2 建模分析(2009.07.01之前)

7.2.1 数据建模和模型评估

为了评估分类器的性能，将数据集分成训练集和测试集，为了获取各变量对违约情况的影响的重要程度，可以考虑用随机森林算法。将分配的30%的测试集，对训练出的模型进行评估。对预测的准确率进行计算。代码如下：
loanData_usemodel = loanData[loanData['Status'] != 2]
before2009=loanData_usemodel[loanData_usemodel['DataPhase']=='Before Jul.2009']
Y=before2009['Status']
X=before2009[['CreditGrade','CustomerClarify','IncomeRange','DebtToIncomeRatio','DelinquenciesLast7Years','BorrowerRate','IsBorrowerHomeowner','ListingCategory (numeric)','EmploymentStatusDuration','InquiriesLast6Months','CreditScore','BankCardUse']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
rfr=RandomForestClassifier()
rfr.fit(X_train,Y_train)
result=rfr.predict(X_test)
def accuracy(rd,prd):
    count=len(prd)
    sum=0
    for i in range(0,count):
        if rd[i]==prd[i]:
            sum=sum+1
    percent=round(sum/count,4
    return percent
percent=accuracy(list(Y_test.values),list(result))
print(percent)
得到该模型预测结果的准确率为0.6659。
7.2.2 变量的重要性

featureImp=pd.Series(list(rfr.feature_importances_),index=X.columns).sort_values(ascending=False)
print(featureImp)

对模型预测结果的准确率影响最大的前三个特征如上图所示
7.3 建模分析(2009.07.01之后)

7.3.1 数据建模和模型评估

与上面同理，代码实现如下：
loanData_usemodel=loanData[loanData['Status']!=2]
after2009=loanData_usemodel[loanData_usemodel['DataPhase']=='After Jul.2009']

Y=after2009['Status']
X=after2009[['ProsperRating (Alpha)','CustomerClarify','IncomeRange','DebtToIncomeRatio','DelinquenciesLast7Years','BorrowerRate','IsBorrowerHomeowner','ListingCategory (numeric)','EmploymentStatusDuration','InquiriesLast6Months','CreditScore','BankCardUse']]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
rfr=RandomForestClassifier()
rfr.fit(X_train,Y_train)
result2=rfr.predict(X_test)
def accuracy2(rd,prd):
    count=len(prd)
    sum=0
    for i in range(0,count):
        if rd[i]==prd[i]:
            sum=sum+1
    pecent=round(sum/count,4)
    
    return pecent
pecent=accuracy2(list(Y_test.values),list(result2))
print(pecent)
该模型预测结果的准确率为0.7402。
7.3.2 变量的重要性
featureImp=pd.Series(rfr.feature_importances_,index=X.columns).sort_values(ascending=False)
featureImp

对模型预测结果的准确率影响最大的前三个特征如上图所示
7.4 小结
2009.07.01前后的模型特征重要性相比，EmploymentStatusDuration由第三位变为第一位，说明在之后的模型更加强调雇佣状态持续时间的重要性，从模型预测的准确率来看，这种模型的调整是有效的，使得准确率由66.59%增加到74.02%。
八、数据预测
因此根据2009年后的模型对正在贷款状态的客户可能违约情况进行预测，将结果保存到csv文件中，共56576条正在贷款的客户，最后得整体的违约率为6.3%。
loanData_usecurrent=loanData[loanData['Status']==2]

X_current=loanData_usecurrent[['ProsperRating (Alpha)','CustomerClarify','IncomeRange','DebtToIncomeRatio','DelinquenciesLast7Years','BorrowerRate','IsBorrowerHomeowner','ListingCategory (numeric)','EmploymentStatusDuration','InquiriesLast6Months','CreditScore','BankCardUse']]
currentpredict=rfr.predict(X_current)
loanData_usecurrent.loc[loanData_usecurrent.Status.notnull(),'Status']=currentpredict
loanData_usecurrent.to_csv('D:/kaggle/prosper/loanData_usecurrent_predictresu.csv',index=False)
completedRate=currentpredict.sum()/len(currentpredict)
defaultedRate=1-completedRate
print(defaultedRate)
九、总结
本文详述了如何通过数据预处理、探索性分析、缺失值处理、建模分析、模型预测等方法，实现对kaggle上Prosper借贷平台贷款者还款与否这一分类问题的数据分析预测实践。分别对2009.07.01前后的模型进行建模分析对比，得出两个模型的预测准确率和特征重要性对比分析，明确看出2009.07.01前后的模型明显有很大的不同。再基于2009年后模型，对正在贷款状态的客户的违约可能性进行预测。
