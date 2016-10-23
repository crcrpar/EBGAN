# [WORK IN PROGRESS]

# Energy-Based Generative Adversarial Network
## Ref.
Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based Generative Adversarial Network, 1–15. Retrieved from http://arxiv.org/abs/1609.03126

## 0. Environment
- Python 3.5.2
- chainer==1.16.0

## 1. Loss Function
### 1.1. Discriminator
$$
\begin{eqnarray}
f_D(x, z) &=& D(x) + [m - D(G(z))]^+ \\
&=& || Dec\left(Enc(x)\right) - x || + [m - \left|\left|Dec\left(Enc(G(z))\right) - G(z)\right|\right|]^+
\end{eqnarray}
$$

### 1.2. Generator
$${
\begin{eqnarray}
f_G(z) &=& ||D(G(z))|| \\
&=& ||Dec \left(Enc(G(z))\right) - G(z) ||
\end{eqnarray}
$$

## 2. Regularizer
**Pull-away Term**
$$
\begin{eqnarray}
f_{\rm PT}(S) = \dfrac{1}{{\rm batch\_size}({\rm batch\_size}-1)} \sum_i \sum_{j \ne i} \left( \dfrac{S_i^{\top}S_j}{||S_i|| ||S_j||} \right)^2
\end{eqnarray}
$$
