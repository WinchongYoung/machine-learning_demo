import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# 1.获取数据
names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
         'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
    names=names)
# 2.基本数据处理
# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data = data.dropna()  # 将数值为NaN的结果删除

# 2.2 确定特征值,目标值
x = data.iloc[:, 1:-1]
y = data["Class"]

# 2.3 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, test_size=0.2)

# 3.特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 4.机器学习(逻辑回归)
estimator = LogisticRegression()
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 准确率
ret = estimator.score(x_test, y_test)
print("准确率:\n", ret)

# 5.2 预测值
y_pre = estimator.predict(x_test)

# 5.3 精确率\召回率指标评价
ret = classification_report(y_test, y_pre, labels=(2, 4), target_names=("良性", "恶性"))
print(ret)

# 5.4 auc指标计算
y_test = np.where(y_test > 3, 1, 0)
roc_auc_score(y_test, y_pre)
