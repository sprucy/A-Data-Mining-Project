import time
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

# 基础配置信息
n_splits =10
seed =50
num_class =11
# lgb 参数
params={
    "task":"train",#默认值=train,可选项=train，prediction；
    "boosting_type":"gbdt", 
    "num_leaves":128, #每个树上的叶子数，认值为31,数值越大，深度越高、准确率越高，过大会导致过拟合。
    "lambda_l2":0.25,
    "max_depth":-1, #每棵树的最大深度或生长的层数上限,-1为无限制
    #"n_estimators":2500, 
    "objective":'multiclass',
    "subsample":0.9, 
    "colsample_bytree":0.5, 
    "subsample_freq":1,
    "learning_rate":0.035,  #学习率，默认值：0.1，越大收敛速度越快，减小可提高准确率
    "n_jobs":10,
    "num_class":num_class,#分类数量，默认值为1，用于多分类的场合。
    "seed":seed
}
# params={
#     "boosting_type":"gbdt", 
#     "learning_rate":0.1,
#     "lambda_l2":0.25,
#     "max_depth":-1,
#     "num_leaves":128,
#     "objective":"multiclass",
#     "num_class":num_class,
#     "seed":seed,
# }
#原始分类特征
origin_cate_feature = ['service_type', 'complaint_level', 'contract_type', 'gender', 'is_mix_service',
                       'is_promise_low_consume', 'many_over_bill', 'net_service']
#原始数字特征
origin_num_feature = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'age',
                      'contract_time', 'former_complaint_fee', 'former_complaint_num',
                      'last_month_traffic', 'local_caller_time', 'local_trafffic_month', 'month_traffic',
                      'online_time', 'pay_num', 'pay_times', 'service1_caller_time', 'service2_caller_time']
# 计算特征
compu_feature_list = ['diff_total_fee_1', 'diff_total_fee_2', 'diff_total_fee_3', 'last_month_traffic_rest', 'rest_traffic_ratio', 
                    'total_fee_mean', 'total_fee_max', 'total_fee_min', 'service_caller_time','service2_caller_ratio',  
                    'local_caller_ratio', 'total_month_traffic', 'month_traffic_ratio', 'pay_num_1_total_fee', 
                    '1_total_fee_call_fee', '1_total_fee_call2_fee', '1_total_fee_trfc_fee',
                    'count_1_total_fee', 'count_2_total_fee', 'count_3_total_fee', 'count_4_total_fee',
                    'count_former_complaint_fee', 'count_pay_num', 'count_contract_time', 'count_last_month_traffic',
                    'count_online_time', 'count_service_type_1_total_fee', 'count_service_type_2_total_fee',
                    'count_service_type_3_total_fee', 'count_service_type_4_total_fee',
                    'count_service_type_former_complaint_fee', 'count_service_type_pay_num',
                    'count_service_type_contract_time', 'count_service_type_last_month_traffic',
                    'count_service_type_online_time', 'count_contract_type_1_total_fee',
                    'count_contract_type_2_total_fee', 'count_contract_type_3_total_fee',
                    'count_contract_type_4_total_fee', 'count_contract_type_former_complaint_fee',
                    'count_contract_type_pay_num', 'count_contract_type_contract_time',
                    'count_contract_type_last_month_traffic', 'count_contract_type_online_time',
                    '1_total_fee_rate12','1_total_fee_rate23','1_total_fee_rate34','service_caller_time_diff']
#                    ,
#                    'service_caller_time_sum','service_caller_time_min', 'service_caller_time_max',
#                    'total_fee_std','total_fee_Standardization','month_traffic_last_month_traffic_sum',
#                    'month_traffic_last_month_traffic_diff','month_traffic_last_month_traffic_rate',
#                    'pay_num_per','total_fee_mean4_pay_num_rate' ,'local_trafffic_month_spend',
#                    'month_traffic_1_total_fee_rate','service_caller_time','outer_caller_time',
#                    'month_traffic_1','month_traffic_50','month_traffic_1024','month_traffic_1024_50',
#                    'last_month_traffic_1','last_month_traffic_50','last_month_traffic_1024',
#                    'last_month_traffic_1024_50','local_trafffic_month_1','local_trafffic_month_50',
#                    'local_trafffic_month_1024','local_trafffic_month_1024_50']
count_feature_list = []
def astype(x,t):
    try:
        return t(x)
    except:
        return np.nan
#数据预处理函数
def deal(data):
    for i in origin_num_feature:
        data[i] = data[i].apply(lambda x: astype(x,float))
        data.loc[data[i] < 0, i] = np.nan
        data[i] = data[i].round(4)
    data['age'] = data['age'].apply(lambda x: astype(x,int))
    data.loc[data['age']==0,'age'] = np.nan
    data['gender'] = data['gender'].apply(lambda x: astype(x,int))
    return data
############################### 特征函数 ###########################
def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    count_feature_list.append(new_feature)
    return data

#特征矩阵生成函数
def make_feat(data):
    t0 = time.time()



    data = feature_count(data, ['1_total_fee'])
    data = feature_count(data, ['2_total_fee'])
    data = feature_count(data, ['3_total_fee'])
    data = feature_count(data, ['4_total_fee'])

    data = feature_count(data, ['former_complaint_fee'])
    data = feature_count(data, ['pay_num'])
    data = feature_count(data, ['contract_time'])
    data = feature_count(data, ['last_month_traffic'])
    data = feature_count(data, ['online_time'])

    for i in ['service_type', 'contract_type']:
        data = feature_count(data, [i, '1_total_fee'])
        data = feature_count(data, [i, '2_total_fee'])
        data = feature_count(data, [i, '3_total_fee'])
        data = feature_count(data, [i, '4_total_fee'])
        data = feature_count(data, [i, 'former_complaint_fee'])
        data = feature_count(data, [i, 'pay_num'])
        data = feature_count(data, [i, 'contract_time'])
        data = feature_count(data, [i, 'last_month_traffic'])
        data = feature_count(data, [i, 'online_time'])


    data['diff_total_fee_1'] = data['1_total_fee'] - data['2_total_fee']
    data['diff_total_fee_2'] = data['2_total_fee'] - data['3_total_fee']
    data['diff_total_fee_3'] = data['3_total_fee'] - data['4_total_fee']
    data['pay_num_1_total_fee'] = data['pay_num'] - data['1_total_fee']
    data['last_month_traffic_rest'] = data['month_traffic'] - data['last_month_traffic']
    for i in data['last_month_traffic_rest']:
        if i < 0:
            i = 0
    data['rest_traffic_ratio'] = (data['last_month_traffic_rest'] * 15 / 1024) / data['1_total_fee']
    total_fee = []
    for i in range(1, 5):
        total_fee.append(str(i) + '_total_fee')
    data['total_fee_mean'] = data[total_fee].mean(1)
    data['total_fee_max'] = data[total_fee].max(1)
    data['total_fee_min'] = data[total_fee].min(1)
 #****   
    data['1_total_fee_rate12'] = data['1_total_fee'] / (data['2_total_fee'] + 0.1)
    data['1_total_fee_rate23'] = data['2_total_fee'] / (data['3_total_fee'] + 0.1)
    data['1_total_fee_rate34'] = data['3_total_fee'] / (data['4_total_fee'] + 0.1)

    data['service_caller_time_diff'] = data['service2_caller_time'] - data['service1_caller_time']
    data['service_caller_time_sum'] = data['service2_caller_time'] + data['service1_caller_time']
    data['service_caller_time_min'] = data[['service1_caller_time', 'service2_caller_time']].min(axis=1)
    data['service_caller_time_max'] = data[['service1_caller_time', 'service2_caller_time']].max(axis=1)

    data['total_fee_std'] = data[total_fee[:4]].std(axis=1)
    data['total_fee_Standardization'] = data['total_fee_std'] / (data['total_fee_mean'] + 0.1)

    data['month_traffic_last_month_traffic_sum'] = data['month_traffic'] + data['last_month_traffic']
    data['month_traffic_last_month_traffic_diff'] = data['month_traffic'] - data['last_month_traffic']
    data['month_traffic_last_month_traffic_rate'] = data['month_traffic'] / (data['last_month_traffic'] + 0.01)
    

 
    data['pay_num_per'] = data['pay_num'] / (data['pay_times'] + 0.01)
    data['total_fee_mean4_pay_num_rate'] = data['pay_num'] / (data['total_fee_mean'] + 0.01)
    data['local_trafffic_month_spend'] = data['local_trafffic_month'] - data['last_month_traffic']
    data['month_traffic_1_total_fee_rate'] = data['month_traffic'] / (data['1_total_fee'] + 0.01)

    for traffic in ['month_traffic', 'last_month_traffic', 'local_trafffic_month']:
        data['{}_1'.format(traffic)] = ((data[traffic] % 1 == 0) & (data[traffic] != 0))
        data['{}_50'.format(traffic)] = ((data[traffic] % 50 == 0) & (data[traffic] != 0))
        data['{}_1024'.format(traffic)] = ((data[traffic] % 1024 == 0) & (data[traffic] != 0))
        data['{}_1024_50'.format(traffic)] = ((data[traffic] % 1024 % 50 == 0) & (data[traffic] != 0))

    data['service_caller_time'] = data['service1_caller_time'] + data['service2_caller_time']
    data['outer_caller_time'] = data['service_caller_time'] - data['local_caller_time']

#*****
    data['service2_caller_ratio'] = data['service2_caller_time'] / data['service_caller_time']
    data['local_caller_ratio'] = data['local_caller_time'] / data['service_caller_time']
    data['total_month_traffic'] = data['local_trafffic_month'] + data['month_traffic']
    data['month_traffic_ratio'] = data['month_traffic'] / data['total_month_traffic']
    data['last_month_traffic_ratio'] = data['last_month_traffic'] / data['total_month_traffic']
    data['1_total_fee_call_fee'] = data['1_total_fee'] - data['service1_caller_time'] * 0.15
    data['1_total_fee_call2_fee'] = data['1_total_fee'] - data['service2_caller_time'] * 0.15
    data['1_total_fee_trfc_fee'] = data['1_total_fee'] - (data['month_traffic'] - 2 * data['last_month_traffic']) * 0.3
    data.loc[data.service_type == 1, '1_total_fee_trfc_fee'] = None

    data.reset_index(drop=True, inplace=True)

    print('    特征矩阵大小：{}'.format(data.shape))
    print('  生成特征一共用时{}秒'.format(time.time() - t0))
    return data

# 多分类F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(num_class, -1),axis=0)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return 'F1_score', score_vali,True

# ********************主程序**************************
# 读取数据
print("开始读取训练数据和预测数据...")
t0 = time.time()
train = pd.read_csv('train.csv',low_memory=False)
test = pd.read_csv('test.csv',low_memory=False)
print('  读取训练数据和预测数据一共用时{}秒'.format(time.time() - t0))

# 数据预处理
print("数据预处理...")
t0 = time.time()
#删除预测数据的current_service列
del test['current_service']
# 映射标签编码关系
feetype=set(train['current_service'])
current_service2label=dict(zip(feetype,list(range(0,len(feetype)))))
label2current_service=dict(zip(list(range(0,len(feetype))),feetype))
# 原始数据的标签映射
train['label'] = train['current_service'].map(current_service2label)
train = deal(train)
test = deal(test)

print('    套餐分类标签：current_service')
print('    训练数据大小：',train.shape)
print('    训练数据用户数：',len(set(train['user_id'])))
print('    训练数据分类数：',len(set(train['current_service'])))
print('    预测数据大小：',test.shape)
print('    预测数据用户数：',len(set(test['user_id'])))
print('  数据预处理一共用时{}秒'.format(time.time() - t0))


print('构造特征矩阵...')
cate_feature = origin_cate_feature
num_feature = origin_num_feature + count_feature_list# + compu_feature_list
feature = cate_feature + num_feature
#标识训练数据和预测数据
train['data_type'] = 1
test['data_type'] = 0
data = pd.concat([train, test], ignore_index=True, sort=False)
data['label'].apply(lambda x: astype(x,int))
data['current_service'].apply(lambda x: astype(x,int))
data_feat = make_feat(data)




# 训练
# 分离训练数据和预测数据
print('开始训练和预测...')
t0 = time.time()
train_x = data_feat[(data_feat.data_type == 1)][feature]
train_y = data_feat[(data_feat.data_type == 1)].label
test_x = data_feat[(data_feat.data_type == 0)][feature]
test_y = data_feat[(data_feat.data_type == 0)]['user_id']

train_x,train_y,test_x = train_x.values,train_y,test_x.values
xx_score = []
cv_pred = []
# 采取k折模型方案
skfold = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
for index,(train_index,valid_index) in enumerate(skfold.split(train_x,train_y)):
    print(index)
    trainx = train_x[train_index]
    trainy = train_y[train_index]
    validx = train_x[valid_index]
    validy = train_y[valid_index]
    train_data = lgb.Dataset(trainx,label=trainy)
    validation_data = lgb.Dataset(validx,label=validy)
    gbm=lgb.train(params,train_data,num_boost_round=50000,valid_sets=[validation_data],early_stopping_rounds=50,feval=f1_score_vali,verbose_eval=1)
    valid_pred = gbm.predict(validx,num_iteration=gbm.best_iteration)
    valid_pred = [np.argmax(x) for x in valid_pred]
    xx_score.append(f1_score(validy,valid_pred,average='macro'))
    test_pred = gbm.predict(test_x,num_iteration=gbm.best_iteration)
    test_pred = [np.argmax(x) for x in test_pred]
    if index == 0:
        cv_pred = np.array(test_pred).reshape(-1,1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(test_pred).reshape(-1,1)))
print('    投票...') 
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))
print('  训练数据和预测数据一共用时{}秒'.format(time.time() - t0))
print('保存结果....')
df_test = pd.DataFrame()
df_test['user_id'] =list(test_y.unique())
df_test['current_service'] = submit
df_test['current_service'] = df_test['current_service'].map(label2current_service)
df_test.to_csv('submit_pre.csv',index=False)
print(xx_score,np.mean(xx_score))