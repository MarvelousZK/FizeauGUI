# Zernike 拟合、Noll 编号与本项目实现

## 1. 先说结论

本项目的 zStd 使用的是 **Noll Zernike 顺序和 Noll 归一化**。它不是递推算法：径向多项式由阶乘系数的有限求和式直接计算，然后在有效圆孔径内用最小二乘拟合系数。

原始 MATLAB 文件 zStd.m 留有“光学车间检测 383 页，公式 (13.47)”的注释。这个页码不是 180 多页，而且原文件没有记录中文书的版次、译者和 ISBN，因此不能把“第 383 页”当作所有版本通用的页码。可公开核验的英文第三版是 Mahajan 所写的第 13 章 *Zernike Polynomial and Wavefront Fitting*，范围为第 498–546 页。

## 2. Noll 编号在代码中怎样体现

Noll 的表 I 使用单一序号 $j=1,2,\ldots$ 排列二维指标 $(n,m)$：

- 先按径向阶数 $n$ 从低到高排列；
- 同一 $n$ 内，较小的方位频率 $m$ 先出现；
- $m\ne0$ 时，偶数 $j$ 使用 $\cos(m\theta)$，奇数 $j$ 使用 $\sin(m\theta)$；
- $m=0$ 时只有一个纯径向项。

项目中的 mVec 保存的是非负的 $|m|$；正弦/余弦分支由 $j$ 的奇偶性决定。因此报告中显示 |m|，而不是容易让人误以为带符号的 m。

前 15 项的实际对应关系如下。

| Noll $j$ | $n$ | $|m|$ | 角向项 | 本项目名称 |
|---:|---:|---:|---|---|
| 1 | 0 | 0 | 1 | 活塞（常数项） |
| 2 | 1 | 1 | $\cos\theta$ | X 倾斜 |
| 3 | 1 | 1 | $\sin\theta$ | Y 倾斜 |
| 4 | 2 | 0 | 1 | 离焦 |
| 5 | 2 | 2 | $\sin2\theta$ | 45°像散 |
| 6 | 2 | 2 | $\cos2\theta$ | 0°像散 |
| 7 | 3 | 1 | $\sin\theta$ | Y 彗差 |
| 8 | 3 | 1 | $\cos\theta$ | X 彗差 |
| 9 | 3 | 3 | $\sin3\theta$ | 三叶像差 30° |
| 10 | 3 | 3 | $\cos3\theta$ | 三叶像差 0° |
| 11 | 4 | 0 | 1 | 初级球差 |
| 12 | 4 | 2 | $\cos2\theta$ | 二级像散 0° |
| 13 | 4 | 2 | $\sin2\theta$ | 二级像散 45° |
| 14 | 4 | 4 | $\cos4\theta$ | 四叶像差 0° |
| 15 | 4 | 4 | $\sin4\theta$ | 四叶像差 22.5° |

这里顺便修正了旧界面的命名问题：Z7–Z8 的 X/Y 方向曾被写反，Z9–Z10 现按角向公式写为 30°/0° 三叶像差，Z14–Z15 曾被误写成二级彗差。按照当前基函数，Z14–Z15 实际是四叶像差；二级彗差从 $n=5,|m|=1$ 的后续项开始。

## 3. 多项式公式和归一化

径向多项式采用直接有限和

$$
R_n^m(\rho)=
\sum_{s=0}^{(n-m)/2}
(-1)^s
\frac{(n-s)!}
{s!\left(\frac{n+m}{2}-s\right)!
\left(\frac{n-m}{2}-s\right)!}
\rho^{n-2s},
$$

其中 $0\le\rho\le1$，且 $n-m$ 必须为偶数。再按 Noll 归一化构造

$$
Z_j(\rho,\theta)=
\begin{cases}
\sqrt{n+1}\,R_n^0(\rho), & m=0,\\[4pt]
\sqrt{2(n+1)}\,R_n^m(\rho)\cos(m\theta), & m\ne0,\ j\text{ 为偶数},\\[4pt]
\sqrt{2(n+1)}\,R_n^m(\rho)\sin(m\theta), & m\ne0,\ j\text{ 为奇数}.
\end{cases}
$$

该归一化满足单位圆上的正交关系

$$
\int_0^{2\pi}\int_0^1 Z_jZ_k\,\rho\,d\rho\,d\theta
=\pi\,\delta_{jk}.
$$

### 它是不是递推法？

不是。当前实现中的两重循环分别完成两件事：枚举 $j\rightarrow(n,|m|)$，以及对上式的 $s$ 项求和。代码没有用 $R_n^m$ 与低阶径向多项式之间的递推关系。按 $j$ 从小到大计算也不等于“递推计算”。

对于本项目默认的 80 项，最高径向阶数不高，直接有限和简单且能与原 MATLAB 结果逐项一致。如果将来扩展到数百项，阶乘求和的数值稳定性会变差，才有必要换成稳定递推或其他高阶算法。

## 4. 系数怎样拟合

软件只取圆形有效掩膜内的 $N$ 个像素，在每个像素计算前 $J$ 项 Zernike，组成设计矩阵

$$
\mathbf Z\in\mathbb R^{N\times J}.
$$

对相位差向量 $\boldsymbol\phi$ 求线性最小二乘

$$
\hat{\mathbf c}
=\arg\min_{\mathbf c}\|\mathbf Z\mathbf c-\boldsymbol\phi\|_2^2.
$$

代码使用 numpy.linalg.lstsq，而不是简单利用连续积分正交性逐项内积。这样可以适应离散像素采样和圆周边界处的不完整像素分布。

拟合得到的 $c_j$ 最初单位为弧度。面形高度按

$$
h_j=c_j\frac{\lambda}{2D\pi}
$$

换算为 nm，其中 $D$ 是软件中的双程因子；反射式菲索双程测量取 $D=2$，于是比例为 $\lambda/(4\pi)$。

“去前 $K$ 项”表示计算

$$
W_K=\phi-\sum_{j=1}^{K}\hat c_j Z_j.
$$

所以去前 1 项只是去掉活塞常数，去前 3 项去掉活塞和两个倾斜，去前 4 项再去掉离焦。

## 5. 坐标和跨软件交换时的注意事项

本项目定义 $\theta=\operatorname{atan2}(Y,X)$，其中图像列方向是 $X$，图像行方向是 $Y$；图像坐标的 $Y$ 向下增加。Noll 序号和基函数公式不受影响，但与采用笛卡尔坐标 $Y$ 向上的光学软件交换系数时，所有含正弦角向因子的项可能出现符号翻转。

另外，Noll、OSA/ANSI、Fringe/Wyant 等体系的序号顺序并不相同。导出到其他软件前必须同时核对：编号、归一化、角度零点、角度正方向、表面/波前单位以及反射双程因子；不能只看“Z4”这样的序号。

## 6. 文献与页码说明

1. F. Zernike, “Beugungstheorie des Schneidenverfahrens und seiner verbesserten Form, der Phasenkontrastmethode,” *Physica* **1**, 689–704 (1934). [DOI: 10.1016/S0031-8914(34)80259-5](https://doi.org/10.1016/S0031-8914(34)80259-5)  
   Zernike 圆多项式的原始来源。

2. R. J. Noll, “Zernike polynomials and atmospheric turbulence,” *Journal of the Optical Society of America* **66**(3), 207–211 (1976). [DOI: 10.1364/JOSA.66.000207](https://doi.org/10.1364/JOSA.66.000207)  
   本项目所用单序号排列、奇偶正弦/余弦分支和归一化约定的直接来源；其表 I 与代码前若干项逐项一致。

3. V. N. Mahajan, “Zernike Polynomial and Wavefront Fitting,” in D. Malacara (ed.), *Optical Shop Testing*, 3rd ed., pp. 498–546, Wiley (2007). [DOI: 10.1002/9780470135976.ch13](https://doi.org/10.1002/9780470135976.ch13)  
   系统讲述 Zernike 多项式、像差和波前拟合。Wiley 的第三版目录确认该章页码为 498–546。

4. D. Malacara (ed.), *Optical Shop Testing*, 3rd ed., Wiley (2007). [DOI: 10.1002/9780470135976](https://doi.org/10.1002/9780470135976)  
   中文常称《光学车间检测》或《光学车间检验》。本地原 zStd.m 的“第 383 页，式 (13.47)”只应视为对其所用中文版本的内部溯源线索；在未确认版次前，不把这个页码套用到英文第三版，也不采用无法从源码或目录支持的“180 多页”。

## 7. 本项目的核验结论

- Noll 论文、Wiley 书籍及章节的 DOI、题名、年份由出版社页面和 Crossref 元数据交叉核验；
- zStd 前 15 项的 $(n,|m|)$、正弦/余弦分支和归一化已按代码实际输出逐项检查；
- Semantic Scholar API 在本次核验时返回限流状态，因此未将其作为证据来源；
- 资料核验日期：2026-08-24。
