hog.py尝试了hog特征

pca.py直接对图片数据做pca

最后都使用k-means算法聚类，特征提取部分没有使用label信息

结果如下，每一行为原始的一个类，每一列为聚出的一类

hog: entropy -> 2.0691

![hog](hog.png)

pca: entropy -> 2.12![pca](pca.png)