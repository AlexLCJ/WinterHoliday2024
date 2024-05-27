import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts.charts import Pie
from pyecharts import options as opts

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 读取数据
data = pd.read_csv('C:\\Users\\李昌峻\\Desktop\\ccf_offline_stage1_train.csv')
# copy函数
offline = data.copy()
# 填充Distance中的空值
offline['Distance'].fillna(-1, downcast='infer', inplace=True)
# 将时间转为时间类型
offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
print('test the time forms')
print(offline)

# 将折扣券转为折扣率
offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
(float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
# 打标
offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
        offline['date'],
        offline['date_received']))

# 添加满减列：
offline['is_manjian']=offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
# 添加领券时间为周几
offline['weekday_Receive']=offline['date_received'].apply(lambda x: x.isoweekday())

# paintings1
df_1 = offline[offline['Date_received'].notna()]
tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()
bar_1=Bar(init_opts = opts.InitOpts(width='1500px', height='600px'))
# 横纵坐标设置
axis_x = list(tmp['Date_received'])
axis_y = list(tmp['Coupon_id'])
# set
bar_1.add_xaxis(axis_x)
bar_1.add_yaxis("领取数量", axis_y)
bar_1.set_series_opts(markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="max")]))
bar_1.set_global_opts(
        title_opts = opts.TitleOpts(title='每天被领券的数量'), # title
        legend_opts = opts.LegendOpts(is_show=True), # 显示ToolBox
        xaxis_opts = opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1), # 旋转60度
)
bar_1.render('D:\\Data\\bar_1.html')

# 每月核销人数
print("每月核销人数")
offline['received_month']=offline['date_received'].apply(lambda x:x.month)
consume_coupon = offline[offline['label']==1]['received_month'].value_counts(sort=False)
consume_coupon.sort_index(inplace=True)
print(consume_coupon)

# 每月领取人数
print("每月领取人数")
received=offline['received_month'].value_counts(sort=False)
received.sort_index(inplace=True)
print(received)

# 消费月份次数
print("每月消费次数")
offline['date_month']=offline['date'].apply(lambda x:x.month)
consume_num=offline['date_month'].value_counts(sort=False)
consume_num.sort_index(inplace=True)

print(consume_num)

## 绘制折线图

line_1 = (
    Line()
    .add_xaxis([str(x) for x in range(1, 7)])
    .add_yaxis('核销', list(consume_coupon))
    .add_yaxis('领取', list(received))
    .add_yaxis('消费', list(consume_num))
    .set_global_opts(title_opts={'text': '每月各类消费折线图'})
    .set_series_opts(
        opts.LabelOpts(is_show=False) # 显示值大小
    )
)
line_1=line_1.render('D:\\Data\\line_1.html')

# 统计有距离的消费次数：
dis=offline[offline['Distance']!=-1]['Distance'].value_counts()
dis.sort_index(inplace=True)
print(dis)
bar_2 = (
    Bar()
    .add_xaxis([str(x)for x in range(0, 11)])
    .add_yaxis('消费距离',list(dis))
    .set_global_opts(title_opts={'text': '卖家和买家距离条形图'})
    .set_series_opts(
        opts.LabelOpts(is_show=False)  # 显示值大小
    )
)
bar_2=bar_2.render('D:\\Data\\bar_2.html')

# 核销率：
# distance=i for i in range(0,11)的时候：
"""sum=offline[offline['Distance']==0]['label'].value_counts().sum()"""
# 每一次
rate = [offline[offline['Distance']==i]['label'].value_counts()[1]/
offline[offline['Distance']==i]['label'].value_counts().sum() for i in range(11)]
#绘图
bar_3=(
    Bar()
    .add_xaxis([str(x)for x in range(0, 11)])
    .add_yaxis('核销率',list(rate))
    .set_global_opts(title_opts={'text':'每月核销率'})
    .set_series_opts(
        opts.LabelOpts(is_show=False)
    )
)
bar_3=bar_3.render('D:\\Data\\bar_3.html')

# 满减类型：
# 满减判断以及绘制饼状图：
v1 = ['折扣', '满减']
v2 = list(offline[offline['Date_received'].notna()]['is_manjian'].value_counts(True))
"""print(v2)"""
pie_1 = (
    Pie()
    .add('', [list(v) for v in zip(v1, v2)])
    .set_global_opts(title_opts={'text': '各类优惠券数量占比饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
)
pie_1=pie_1.render('D:\\Data\\pie_1.html')

# 核销优惠券数量占比饼图
# 满足核销：offline['label']==1
v3 = list(offline[offline['label']==1].is_manjian.value_counts(True))
pie_2 = (
    Pie()
    .add('', [list(v) for v in zip(v1, v3)])
    .set_global_opts(title_opts={'text': '核销优惠券数量占比饼图'})
    .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
)
pie_2=pie_2.render('D:\\Data\\pie_2.html')