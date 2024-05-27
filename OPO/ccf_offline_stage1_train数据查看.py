import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('D:\\Data\\ccf_offline_stage1_train.csv')
print(data)

# 共有多少条记录，一维
sum = data.shape[0]
# 共有多少条优惠券的领取记录
received_count = data['Date_received'].count()
# 共有多少种不同的优惠券
diff_kinds = len(data['Coupon_id'].value_counts())
# 共有多少个用户
users_num = len(data['User_id'].value_counts())
# 共有多少个商家
merchant_num = len(data['Merchant_id'].value_counts())
# 最早领券时间
min_received = str(int(data['Date_received'].min()))
# 最晚领券时间
max_received = str(int(data['Date_received'].max()))
# 最早消费时间
min_date = str(int(data['Date'].min()))
# 最晚消费时间
max_date = str(int(data['Date'].max()))

print(f'总数据：', sum,'条')
print('优惠券领取数量：', received_count,'张')
print('优惠券种类', diff_kinds,'种')
print('用户数量', users_num,'位')
print('商家数量', merchant_num,'家')
print('最早领卷', min_received)
print('最晚领卷', max_received)
print('最早消费', min_date)
print('最晚消费',max_date)

# 数据检查
columns_to_check = ['Date_received', 'Coupon_id', 'Merchant_id', 'User_id', 'Date', 'Distance', 'Discount_rate']

for column_to_check in columns_to_check:
    missing_values = data[column_to_check].isnull().sum()
    if missing_values > 0:
        print(f"列 '{column_to_check}' 中有 {missing_values} 个缺失值。")
    else:
        print(f"列 '{column_to_check}' 中没有缺失值。")

