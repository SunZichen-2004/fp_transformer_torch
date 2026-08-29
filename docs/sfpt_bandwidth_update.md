# sFPT Bandwidth 更新策略

实现位置：`src/fp_transformer_torch/kernel/gaussian.py`  
（`PerReplicaRescaleConfig` + `GaussianKernel.update_bandwidth`）

生产实验（真空 / 水相 dialanine、Claisen）统一使用 **per-replica rescale**，基于**未归一化** Gaussian 的 raw row-sum。

---

## 1. 更新时机

每隔 `update_pace` 个 MD 步更新一次（生产默认 **20**）。

在 BAOAB 一步开始前：先用当前全部 replica 的 CV 更新 \(\{\sigma_i\}\)，再用新的 bandwidth 计算 Fokker–Planck bias。

---

## 2. 初始化 \(\sigma_0\)

对初始 CV 计算两两距离 \(D_{ij}\)（二面角用 torus 距离），取上三角正值的中位数：

\[
\sigma_0 = \operatorname{median}\{D_{ij}\}_{i < j,\; D_{ij} > 0}.
\]

若无有效距离，回退到 \(0.5\)。随后广播到所有 replica：

\[
\sigma_i \leftarrow \sigma_0, \qquad i = 1,\ldots,B.
\]

其中 \(B\) 为 replica 数。

---

## 3. 未归一化 Gaussian 与监控指标

生产配置：

```text
use_normalized_kernel = False
normalize_row_sum_by_replicas = False
```

未归一化 kernel（\(\sigma_i\) 挂在**中心** replica \(i\) 上，列索引）：

\[
\tilde{K}_{ji}
  = \exp\!\left(-\frac{1}{2}\,\frac{D_{ji}}{\sigma_i^{2}}\right).
\]

监控指标为每行的 **raw row-sum**（**不除以 \(B\)**）：

\[
m_j = \sum_{i=1}^{B} \tilde{K}_{ji}.
\]

- \(m_j\) 过大：邻居过密 / \(\sigma\) 偏大  
- \(m_j\) 过小：邻居过稀 / \(\sigma\) 偏小  

> 代码中另有归一化路径 \(m_j = (\sum_i K_{ji})/B\)，但上述生产体系均关闭该路径。

---

## 4. 自适应目标带（adaptive quantile）

每次更新时，先取全体 \(\{m_j\}\) 的分位数作为参考：

\[
\operatorname{ref}
  = \operatorname{quantile}(m,\, q),
  \qquad q = 0.5
  \quad\text{（中位数）}.
\]

相对目标带（默认 \(\pm 20\%\)）：

\[
[\ell,\, h]
  = [\operatorname{ref}\cdot\alpha_{\mathrm{lo}},\;
     \operatorname{ref}\cdot\alpha_{\mathrm{hi}}],
  \qquad
  \alpha_{\mathrm{lo}} = 0.8,\;
  \alpha_{\mathrm{hi}} = 1.2.
\]

再将目标带夹到绝对窗口（legacy，因 raw row-sum 与 \(B\) 耦合）：

\[
[\ell,\, h]
  \leftarrow
  \operatorname{clip}\!\big([\ell,\, h],\; [m_{\min},\, m_{\max}]\big).
\]

| 体系 | \(m_{\min}\) | \(m_{\max}\) |
|------|--------------|--------------|
| dialanine（真空 / 水） | \(2\) | \(4\) |
| Claisen（水） | \(2\) | \(5\) |

即：希望每个 replica 的有效邻居数落在「全体中位数的 \(80\%\)–\(120\%\)」附近，且绝对落在 \([2,4]\)（Claisen 为 \([2,5]\)）。

---

## 5. Per-replica 对数空间迭代修正

对越界的 replica \(j\) 定义对数误差：

\[
e_j =
\begin{cases}
\log h - \log m_j, & m_j > h, \\[4pt]
\log \ell - \log m_j, & m_j < \ell, \\[4pt]
0, & \text{otherwise},
\end{cases}
\qquad
e_j \leftarrow \operatorname{clip}(e_j,\, [-c,\, c]).
\]

其中误差裁剪 \(c = 0.5\)（`log_error_clip`）。

在 \(\log\sigma\) 上做小步更新，并加入 **2% 全局 pull**：

\[
\begin{aligned}
\log\sigma
  &\leftarrow \log\sigma + \eta\, e,
  \qquad \eta = 0.05, \\[6pt]
\log\sigma
  &\leftarrow (1-\rho)\,\log\sigma
           + \rho\,\overline{\log\sigma},
  \qquad \rho = 0.02, \\[6pt]
\sigma
  &\leftarrow \operatorname{clip}\!\big(
       \exp(\log\sigma),\;
       [\sigma_{\min},\, \sigma_{\max}]
     \big).
\end{aligned}
\]

其中 \(\overline{\log\sigma}\) 为全体 replica 的 \(\log\sigma\) 均值。

每步迭代后**重算** \(m\)（因为 \(\sigma\) 非线性进入 \(\tilde{K}\)）。最多迭代 `max_rescale_iterations = 128` 次；若全体已进入目标带，或 \(\sigma\) 不再变化，则提前停止。

\(\sigma\) 硬裁剪：

| 体系 | \(\sigma_{\min}\) | \(\sigma_{\max}\) |
|------|-------------------|-------------------|
| dialanine（真空 / 水） | \(10^{-3}\) | \(8\) |
| Claisen（水） | \(10^{-3}\) | \(4\) |

---

## 6. 策略一句话

> **每 20 步**，用当前 CV 重算未归一化 Gaussian 的 **per-replica row-sum**；以全体中位数的 **\(\pm 20\%\)** 为相对目标，并夹在绝对 **\([2,4]\)**（Claisen **\([2,5]\)**）内；对越界 replica 在 \(\log\sigma\) 上做小步修正，并加 **2% 全局 pull**，把各 replica 的有效邻居数拉回同一尺度。

目的：防止个别 replica 的 \(\sigma\) 塌缩（FP 力噪声大）或胀开（kernel 过平、驱动变弱），同时避免全体 \(\sigma\) 漂到极端值。

---

## 7. 超参数汇总

| 符号 / 参数 | 默认 / 生产值 | 含义 |
|-------------|---------------|------|
| `update_pace` | \(20\) | 每隔多少 MD 步更新一次 |
| \(q\) (`adaptive_ref_quantile`) | \(0.5\) | 参考分位数（中位数） |
| \(\alpha_{\mathrm{lo}}\) | \(0.8\) | 相对带下界 |
| \(\alpha_{\mathrm{hi}}\) | \(1.2\) | 相对带上界 |
| \([m_{\min}, m_{\max}]\) | \([2,4]\) / Claisen \([2,5]\) | 绝对目标带 |
| \(\eta\) (`log_step_size`) | \(0.05\) | \(\log\sigma\) 步长 |
| \(c\) (`log_error_clip`) | \(0.5\) | 单步误差上限 |
| \(\rho\) (`global_pull`) | \(0.02\) | 向全局均值拉回比例 |
| max iterations | \(128\) | 单次更新最多迭代次数 |
| \([\sigma_{\min}, \sigma_{\max}]\) | \([10^{-3}, 8]\) / Claisen \([10^{-3}, 4]\) | bandwidth 硬裁剪 |

---

## 8. 与「关掉 per-replica」的对比

若 `enabled = False`，只做一次全局缩放：

\[
\sigma
  \leftarrow
  \sigma \cdot
  \sqrt{\frac{\operatorname{median}(m)}{\operatorname{mean}(m)}},
\]

再 clip 到 \([\sigma_{\min}, \sigma_{\max}]\)。生产默认**不走**这条，而是第 5 节的 per-replica 迭代。
