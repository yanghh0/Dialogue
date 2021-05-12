
import numpy as np


a = np.array([3, 1, 2, 4, 6, 1])
# 取出a中元素最大值所对应的索引，此时最大值位6，其对应的位置索引值为4，（索引值默认从0开始）
b = np.argmax(a)
print(b)

a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])

# 对二维矩阵来讲a[0][1]会有两个索引方向，第一个方向为a[0]，默认按列方向搜索最大值
b = np.argmax(a, axis=0)
# a的第一列为1，9，3,最大值为9，所在位置为1，
# a的第一列为5，6，7,最大值为7，所在位置为2，
# 此此类推，因为a有4列，所以得到的b为1行4列，
print(b)  # [1 2 2 1]

c = np.argmax(a, axis=1)  # 现在按照a[0][1]中的a[1]方向，即行方向搜索最大值，
# a的第一行为1，5，5，2,最大值为5（虽然有2个5，但取第一个5所在的位置），索引值为1，
# a的第2行为9，6，2，8,最大值为9，索引值为0，
# 因为a有3行，所以得到的c有3个值，即为1行3列
print(c)