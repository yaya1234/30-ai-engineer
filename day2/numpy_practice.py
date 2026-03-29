import numpy as np

# 创建数组
arr = np.arange(12).reshape(3, 4)
print("原始数组：\n", arr)
print("数组形状：", arr.shape)
print("数组维度：", arr.ndim)
print("数组数据类型：", arr.dtype)  

# 数组切片
print("第一行：", arr[0])
print("第一列：", arr[:, 0])
print("子数组：\n", arr[1:3, 1:3])

# 数组运算
print("总和:", arr.sum())
print("每列均值:", arr.mean(axis=0))
print("每行标准差:", arr.std(axis=1))

# 广播机制
arr3 = np.array([1, 2, 3, 4])
print("广播加法：\n", arr + arr3)   
print("广播乘法：\n", arr * arr3)

# 随机数与统计
rand_arr = np.random.normal(loc=0, scale=1, size=1000)
print("随机数组均值:", rand_arr.mean())
print("标准差:", rand_arr.std())

# 矩阵乘法
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
print("矩阵乘法结果:\n", A @ B)