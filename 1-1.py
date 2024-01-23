# 引入鸢尾花数据集
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 加载数据
iris = load_iris()
X = iris.data #特征
Y = iris.target #类别
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=66) # 数据划分

# pandas将数据制作成dataframe格式，使得输出更直观，规范
df_iris=pd.DataFrame(iris.data, columns=iris.feature_names)#将data加入到数据框中
class_mapping = {0:'setosa',1:'versicolor',2:'virginica'}
df_iris['target']=iris.target#将target加入到数据框中
df_iris['target'] = df_iris['target'].map(class_mapping)#将分类名由数字修改为直观的文字
print(df_iris)
sns.lmplot(data=df_iris,x="petal width (cm)",y="sepal length (cm)",hue="target",fit_reg=False)#hue为可分的类别，fit_reg为是否对图进行线性回归
plt.show()