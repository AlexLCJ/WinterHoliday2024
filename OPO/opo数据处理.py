import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def show_date():
    print('观测文件可知，文件由“User_id”、“Merchant_id”、“Coupon_id”、“Discount_rate”、'
          '“Distance”、“Date_received”、“Date”组成')

    sum = data.shape[0]
    print(f'总数据：', sum, '条')

    # 共有多少条优惠券的领取记录
    received_count = data['Date_received'].count()
    print('优惠券领取数量：', received_count, '张')

    # 共有多少种不同的优惠券
    diff_kinds = len(data['Coupon_id'].value_counts())
    print('优惠券种类', diff_kinds, '种')

    # 共有多少个用户
    users_num = len(data['User_id'].value_counts())
    print('用户数量', users_num, '位')

    # 共有多少个商家
    merchant_num = len(data['Merchant_id'].value_counts())
    print('商家数量', merchant_num, '家')

    # 最早领券时间
    min_received = str(int(data['Date_received'].min()))
    # 最晚领券时间
    max_received = str(int(data['Date_received'].max()))
    # 最早消费时间
    min_date = str(int(data['Date'].min()))
    # 最晚消费时间
    max_date = str(int(data['Date'].max()))
    #转化为时间模式
    min_received=pd.to_datetime(min_received)
    max_received=pd.to_datetime(max_received)
    min_date=pd.to_datetime(min_date)
    max_date=pd.to_datetime(max_date)

    print('最早领卷', min_received)
    print('最晚领卷', max_received)
    print('最早消费', min_date)
    print('最晚消费', max_date)

    # 数据检查
    columns_to_check = ['Date_received', 'Coupon_id', 'Merchant_id', 'User_id', 'Date', 'Distance', 'Discount_rate']

    for column_to_check in columns_to_check:
        missing_values = data[column_to_check].isnull().sum()
        if missing_values > 0:
            print(f"列 '{column_to_check}' 中有 {missing_values} 个缺失值。")
        else:
            print(f"列 '{column_to_check}' 中没有缺失值。")




def dispose_offline(offline):
    # 复制新的offline，对他进行填充一些数据
    # 填充Distance中的空值
    offline['Distance'].fillna(-1, downcast='infer', inplace=True)
    # 创建新的date_received，date，转化时间显示模式
    offline['date_received'] = pd.to_datetime(offline['Date_received'], format='%Y%m%d')
    offline['date'] = pd.to_datetime(offline['Date'], format='%Y%m%d')
    # 找到折扣率
    offline['discount_rate'] = offline['Discount_rate'].map(lambda x: float(x) if ':' not in str(x) else
    (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))
    # 打标
    offline['label'] = list(map(lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
                                offline['date'],
                                offline['date_received']))
    # 添加满减列：
    offline['is_manjian'] = offline['Discount_rate'].map(lambda x: 1 if ':' in str(x) else 0)
    # 添加领券时间为周几
    offline['weekday_Receive'] = offline['date_received'].apply(lambda x: x.isoweekday())
    #

    return offline  # 返回处理后的数据，以便后续使用

# 每日领取优惠券的分析以及图像绘制
def received_coupon():
    df_1 = offline[offline['Date_received'].notna()]
    tmp = df_1.groupby('Date_received', as_index=False)['Coupon_id'].count()
    bar_1 = Bar(init_opts=opts.InitOpts(width='1500px', height='600px'))
    # 横纵坐标设置
    axis_x = list(tmp['Date_received'])
    axis_y = list(tmp['Coupon_id'])
    # set
    bar_1.add_xaxis(axis_x)
    bar_1.add_yaxis("领取数量", axis_y)
    bar_1.set_series_opts(markline_opts=opts.MarkLineOpts(data=[opts.MarkLineItem(type_="max")]))
    bar_1.set_global_opts(
        title_opts=opts.TitleOpts(title='每天被领券的数量'),  # title
        legend_opts=opts.LegendOpts(is_show=True),  # 显示ToolBox
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=60), interval=1),  # 旋转60度
    )
    bar_1.render('D:\\Data\\bar_1.html')

# 每月核销，领取人数，月消费次数三线合一图：
def line_with_consume_coupon_received_consume_num():
    # 每月核销人数
    print(offline)
    print("每月核销人数")
    offline['received_month'] = offline['date_received'].apply(lambda x: x.month)
    consume_coupon = offline[offline['label'] == 1]['received_month'].value_counts(sort=False)
    consume_coupon.sort_index(inplace=True)
    print(consume_coupon)

    # 每月领取人数
    print("每月领取人数")
    received = offline['received_month'].value_counts(sort=False)
    received.sort_index(inplace=True)
    print(received)

    # 消费月份次数
    print("每月消费次数")
    offline['date_month'] = offline['date'].apply(lambda x: x.month)
    consume_num = offline['date_month'].value_counts(sort=False)
    consume_num.sort_index(inplace=True)

    print(consume_num)
    print('--------------------------------------------')
    print('start painting')
    line_1 = (
        Line()
        .add_xaxis([str(x) for x in range(1, 7)])
        .add_yaxis('核销', list(consume_coupon))
        .add_yaxis('领取', list(received))
        .add_yaxis('消费', list(consume_num))
        .set_global_opts(title_opts={'text': '每月各类消费折线图'})
        .set_series_opts(
            opts.LabelOpts(is_show=False)  # 显示值大小
        )
    )
    line_1 = line_1.render('D:\\Data\\line_1.html')

def get_Distance():
    dis = offline[offline['Distance'] != -1]['Distance'].value_counts()
    dis.sort_index(inplace=True)
    print(dis)
    bar_2 = (
        Bar()
        .add_xaxis([str(x) for x in range(0, 11)])
        .add_yaxis('消费距离', list(dis))
        .set_global_opts(title_opts={'text': '卖家和买家距离条形图'})
        .set_series_opts(
            opts.LabelOpts(is_show=False)  # 显示值大小
        )
    )
    bar_2 = bar_2.render('D:\\Data\\bar_2.html')

def rate_of_use():
    # 核销率：
    # distance=i for i in range(0,11)的时候：
    """sum=offline[offline['Distance']==0]['label'].value_counts().sum()"""
    # 每一次
    rate = [offline[offline['Distance'] == i]['label'].value_counts()[1] /
            offline[offline['Distance'] == i]['label'].value_counts().sum() for i in range(11)]
    # 绘图
    bar_3 = (
        Bar()
        .add_xaxis([str(x) for x in range(0, 11)])
        .add_yaxis('核销率', list(rate))
        .set_global_opts(title_opts={'text': '月核销率'})
        .set_series_opts(
            opts.LabelOpts(is_show=False)
        )
    )
    bar_3 = bar_3.render('D:\\Data\\bar_3.html')

def manjian():
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
    pie_1 = pie_1.render('D:\\Data\\pie_1.html')

    # 核销优惠券数量占比饼图
    # 满足核销：offline['label']==1
    v3 = list(offline[offline['label'] == 1].is_manjian.value_counts(True))
    pie_2 = (
        Pie()
        .add('', [list(v) for v in zip(v1, v3)])
        .set_global_opts(title_opts={'text': '核销优惠券数量占比饼图'})
        .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
    )
    pie_2 = pie_2.render('D:\\Data\\pie_2.html')

def discount_coupon():
    # 统计折扣率和核销数量
    received = (offline[['discount_rate']]
                .assign(cnt=1)
                .groupby('discount_rate')['cnt'].sum()
                .reset_index())

    # 统计领券日期不为空的数据中各种折扣率的优惠券核销数量
    consume_coupon = (offline[offline['label'] == 1][['discount_rate']]
                      .assign(cnt_2=1)
                      .groupby('discount_rate')['cnt_2'].sum()
                      .reset_index())

    data = (received.merge(consume_coupon, on='discount_rate', how='left')
            .fillna(0))

    bar_4 = (
        Bar()
        .add_xaxis([float('%.4f' % x) for x in list(data.discount_rate)])
        .add_yaxis('领取', list(data.cnt))
        .add_yaxis('核销', list(data.cnt_2))
        .set_global_opts(title_opts={'text': '领取与核销的数量'})
        .set_series_opts(
            opts.LabelOpts(is_show=True)
        )
    )
    bar_4.render('D:\\Data\\bar_4.html')

def weekdays_coupon_get():
    # 核销数量
    use_get_weekdays = offline[offline['label'] == 1]['weekday_Receive'].value_counts(sort=False)
    use_get_weekdays.sort_index(inplace=True)
    print('核销数量')
    print(use_get_weekdays)
    # 领取数量
    get_weekdays = offline[offline['weekday_Receive'].notna()]['weekday_Receive'].value_counts()
    get_weekdays.sort_index(inplace=True)
    print('领取数量')
    print(get_weekdays)
    # 绘图
    line_2 = (
        Line()
        .add_xaxis([str(x) for x in range(1, 8)])
        .add_yaxis('周几核销', list(use_get_weekdays))
        .add_yaxis('周几领取', list(get_weekdays))
        .set_global_opts(title_opts={'text': '星期领取日'})
        .set_series_opts(
            opts.LabelOpts(is_show=False)  # 显示值大小
        )
    )
    line_2 = line_2.render('D:\\Data\\line_2.html')

def one_or_zero_paintings():
    v1 = ['正例', '负例']
    v2 = list(offline['label'].value_counts(True))
    """print(v2)"""
    pie_3 = (
        Pie()
        .add('', [list(v) for v in zip(v1, v2)])
        .set_global_opts(title_opts={'text': '正例、负例饼状图'})
        .set_series_opts(label_opts=opts.LabelOpts(formatter='{b}: {c}'))
    )
    pie_3 = pie_3.render('D:\\Data\\pie_3.html')


if __name__ == '__main__':                # 当前文件下
    # 读取数据
    data = pd.read_csv('D:\\Data\\opodata\\tabel3\\ccf_offline_stage1_train.csv')
    # copy函数
    offline = data.copy()
    offline = dispose_offline(offline)
    # date展示：
    print('--------------------------------------')
    show_date()
    print('--------------------------------------')
    received_coupon()
    print('每日领取优惠券的分析以及图像绘制已完成')
    print('--------------------------------------')
    line_with_consume_coupon_received_consume_num()
    print('消费距离')
    get_Distance()
    print('---------------------------------------')
    rate_of_use()
    print('满减类型')
    manjian()
    print('---------------------------------------')
    print('展示领取与核销图')
    discount_coupon()
    print('---------------------------------------')
    weekdays_coupon_get()
    print('---------------------------------------')
    one_or_zero_paintings()


