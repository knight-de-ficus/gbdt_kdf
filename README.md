# 基于GBDT的网络攻击检测与识别

## 现在要做的工作

软件的封装和图形化，论文写说明书
做特征选择来减小时间
对于某些不平衡标签，提高权重
不同的数据集的效果
十分训练集的测试？

## 要实现的图片

哈夫曼 vs 独热：
   卡方检验中，哈夫曼输出的列能够比独热的列更靠前

针对特征选择：
   折线图，理想状况下，训练时间能够随着选择的特征列数量的增加而减少
   另一方面，准确率能够呈现一种弧形（你懂得

针对小众标签的优化：
   混淆矩阵的对比图，如果成功，优化后的混淆矩阵能够取得u2r的比较高的精确度和召回率

## 对比实验

哈夫曼编码 vs 独热 （√）
特征选择的参数 训练时间 准确度   的折线图
不同模型的效果
传统模型，GBDT，XGBoost

## 论文的目的

做当前的所有工作，不包括所谓算法改进，去做到最好的网络攻击检测与识别

kdd_f {'dos': 0, 'normal': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}

unsw_f {'Normal': 0, 'Backdoor': 1, 'Analysis': 2, 'Fuzzers': 3, 'Shellcode': 4, 'Reconnaissance': 5, 'Exploits': 6, 'DoS': 7, 'Worms': 8, 'Generic': 9}
