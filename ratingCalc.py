import pandas as pd

# 读取Sheet1和Sheet2
sheet1 = pd.read_excel('Name.xlsx', sheet_name='Sheet1')
sheet2 = pd.read_excel('Name.xlsx', sheet_name='Sheet2')
sheet1 = sheet1.astype(str)
sheet2['姓名学号'] = sheet2['姓名学号'].astype(str).str.strip()
sheet2['被评海报编号'] = sheet2['被评海报编号'].astype(str).str.strip().replace(r'^0(\d)-', r'\1-', regex=True)
print(sheet1.columns)
# 处理Sheet2的A列，分离姓名和学号
def parse_A_col(a):
    import re
    match = re.search(r'\d+', a)
    try:
        if match:
            student_id = match.group()
            name =  sheet1.loc[sheet1['学号'] == student_id, '姓名'].values[0]
            return {'姓名': name, '学号': student_id}
        else:
            # 如果无法提取学号，则认为是姓名
            name = ''.join(a.split())
            return {'姓名': name, '学号': sheet1.loc[sheet1['姓名'] == name, '学号'].values[0]}
    except:
        return {'姓名': 'NaN', '学号': 'NaN'}

sheet2['姓名'] = sheet2['姓名学号'].apply(lambda x: parse_A_col(x)['姓名'])
sheet2['学号'] = sheet2['姓名学号'].apply(lambda x: parse_A_col(x)['学号'])
# sheet2 = sheet2.drop(columns=['A'])

# 统计每个海报的被评次数和平均分
poster_stats = sheet2.groupby('被评海报编号')['打分'].agg(['count', 'mean']).reset_index()
print(poster_stats)

# 统计每个人评价的海报数量
evaluator_counts = sheet2.groupby(['姓名', '学号'])['被评海报编号'].count().reset_index(name='评价他人海报总次数')
print(evaluator_counts)
# 合并Sheet1和poster_stats
result = sheet1
result = result.merge(poster_stats, left_on='序号', right_on='被评海报编号', how='left')
result = result.drop(columns=['被评海报编号'])
result = result.rename(columns={'count': '被评总次数', 'mean': '被评平均分'})

# # 合并评价次数
result = result.merge(evaluator_counts, on=['姓名', '学号'], how='left')

# # 填充NaN为0
# result['被评总次数'] = result['被评总次数'].fillna(0)
# result['被评平均分'] = result['被评平均分'].fillna(0)
# result['评价他人海报总次数'] = result['评价他人海报总次数'].fillna(0)

# # 输出结果
print(result[['序号', '姓名', '被评总次数', '被评平均分', '评价他人海报总次数']])
result.to_excel('ratingCalc.xlsx', index=False)