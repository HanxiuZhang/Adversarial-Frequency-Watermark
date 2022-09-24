## 一种通过频域水印施加梯度攻击的方法
---
在前期的实验中发现：
- 直接对添加电子水印的图像施加梯度干扰，很可能会对水印的提取效果产生无法估量的干扰
- 基于优化的攻击方法由于对图像质量的限制，能够减轻上述问题，但攻击效果不如梯度攻击

提出了一种通过频域水印施加梯度攻击的方法，该方法的**创新点**体现在：
- 不直接对原图像施加梯度攻击，而是**改变图像上所施加的电子水印**
- 可以用于无法改变原始图像，但**能够对电子水印进行修改**的攻击情景
- **对水印的提取效果进行控制**的同时，能够达到与梯度攻击相似的攻击效果
- 对于任意给定的基于梯度的攻击方法，可以**即插即用**
---
### 水印的添加流程
原始图像为$I$,原始水印为等尺寸的$W$
1. 对于每个channel，将$I$按照$n\times n$的block分块
2. 对于$I$的每个block进行离散余弦变换, 得到频域图像$I'$
$$I' = F_{DCT}(I)$$
3. 在频域图像上按照强度$\alpha$叠加水印$W$,得到叠加后的频域图像$I'_{W}$
$$I'_{W} = I'+\alpha W$$
4. 对于$I'_{W}$进行分块与反DCT变换，得到施加频域水印后的图像$I_W$：
$$I_W = F_{IDCT}(I'_W)$$

记上述过程为
$$I_W = F(I,W)$$
---
### 模型推导
对于添加水印后直接进行梯度攻击的方法，可以表示为：
$$I_{per} = I_W + \beta P$$
$P$为按照梯度攻击的方法（FGSM/PGD等）计算出的梯度\
令$\hat{I_{per}}$为在水印上施加干扰$P_W$后所产生的对抗样本,则
$$\hat{I_{per}} = F(I,W+P_W)$$
为使$\hat{I_{per}}$具有对抗效果，令
$$\hat{I_{per}} = I_{per}$$
即
$$F(I,W+P_W) = F(I,W)+\beta P$$
$$F_{IDCT}(F_{DCT}(I)+\alpha (W+P_W))=F_{IDCT}(F_{DCT}(I)+\alpha W)+\beta P$$
已知
$$F_{DCT}(F_{IDCT}(x))=x$$
且傅里叶变换具有线性性\
将等式两边同时作为$F_{DCT}(x)$的自变量，得到
$$F_{DCT}(I)+\alpha W+ \alpha P_W = F_{DCT}(I)+\alpha W+\beta F_{DCT}(P)$$
$$P_W = \frac{\beta}{\alpha}F_{DCT}(P)$$
可以看出，当$\alpha \gg \beta$时，能够保证施加在水印上的扰动$P_W$足够小
---
### 方法流程
对于给定模型$M$，给定图像$I$以及对应标签$T$（对于指定标签攻击，给定攻击标签$T'$）
1. 利用基于梯度的攻击方法生成梯度干扰$P$
2. 确定参数$\beta$使得$M(clip(I+\beta P))\ne T$\
  对于指定标签攻击：$M(clip(I+\beta P)) = T'$
3. 对$P$进行分块离散余弦变换得到$F_{DCT}(P)$
4. 确定参数$\alpha$，使得$\frac{\beta}{\alpha}$足够小，从而保证对水印添加的扰动$P_W$足够小，且$\alpha$也需要足够小从而确保添加水印后的图像质量，得到施加在电子水印上的对抗扰动
$$P_W=\frac{\beta}{\alpha}F_{DCT}(P)$$
5. 计算得出生成的电子水印对抗样本
$$W_{per} = clip(W+P_W)$$
---
上述问题转化为对于$\alpha$和$\beta$两个参数的优化问题\
目前：手动指定\
由于裁剪操作以及精度误差，$\beta$满足$M(clip(I+\beta P))\ne T$时，不一定满足$M(clip(F(I,clip(W+P_W))))\ne T$,
故约束条件为
$$M(F(I,clip(W+P_W)))\ne T$$
宿主图像的质量损失定为$L_I=\lVert I_W-I \rVert_2^2$\
水印图像的质量损失定为$L_W=\lVert P_W \rVert_2^2$\
损失函数定为$L=\lambda_1 L_I+\lambda_2 L_W$\
则该优化问题为
$$\min_{\alpha,\beta \in D} L$$
subject to $$ M(clip(F(I,clip(W+P_W))))\ne T$$
---
# 2022.9.15
## 基于梯度下降的优化方法
---
for $n=0$ to $n = N-1$\
    $$\alpha_{n+1}=\max \{ \alpha_n-s_\alpha\cdot\frac{\mathrm{d}(L)},{\mathrm{d}(\alpha_n)}, 0 \}$$
    $$\beta_{n+1}=\max \{ \beta_n-s_\beta\cdot\frac{\mathrm{d}(L)}{\mathrm{d}(\beta_n)},0\}$$
use $\alpha_{n+1}$ and $\beta_{n+1}$ if $M(clip(F(I,clip(W+P_W))))\ne T$, continue\
else $\alpha = \alpha_{n}$, $\beta = \beta_{n}$, break\

Note that\
$$I_W-I = \alpha F_{IDCT}(W) + \beta P$$
