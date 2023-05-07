# coding:utf-8

from sklearn.neighbors import KNeighborsClassifier

# 1.构造数据
x = [[1], [2], [3], [4]]
y = [0, 0, 1, 1]

# 2.训练模型
# 2.1 实例化一个估计器对象
estimator = KNeighborsClassifier(n_neighbors=3)

# 2.2 调用fit方法,进行训练
estimator.fit(x, y)

# 3.数据预测
ret = estimator.predict([[2.51]])
print(ret)
# 可以这样理解, x是特征值, 是dataframe形式理解为二维的[[]],
# y表示的目标值, 可以表示为series, 表示为一维数组[]
ret1 = estimator.predict([[2.52]])
print(ret1)
