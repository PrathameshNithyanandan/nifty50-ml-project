# Mathematical model for Units 2 and 3

This document states each problem mathematically and derives the update each algorithm uses. Every algorithm is implemented in `unit2.py` or `unit3.py`, with the matrix routines written by hand in `matrix.py`. Equation numbers appear in the code's docstrings.

## 1. Data

For each stock, let $C_t$ be the closing price on trading day $t$. Prices are not adjusted for splits, so a day is flagged as a corporate action when

$$H_t < 0.80\,C_{t-1}\ \text{ or }\ L_t > 1.25\,C_{t-1}\ \text{ or }\ O_t < 0.75\,C_{t-1}\ \text{ or }\ O_t > 1.333\,C_{t-1}. \tag{1.1}$$

Here $O_t, H_t, L_t$ are the open, high and low. The daily log-return is

$$r_t = \ln\frac{C_t}{C_{t-1}}, \tag{1.2}$$

and it is treated as missing on flagged days and across gaps longer than 20 calendar days. The back-adjusted price is rebuilt from the kept returns, ending at the last quoted close:

$$A_t = C_T \exp\Big(\sum_{s \le t} r_s - \sum_{s \le T} r_s\Big). \tag{1.3}$$

The design uses 21 features $\mathbf{x}_t \in \mathbb{R}^{21}$ built from data up to day $t$ only: five lagged returns, rolling means and standard deviations, momentum, distance from the 50-day average, high–low range, open–close return, two volume measures, the market return and weekday indicators.

Columns are standardised with the training mean and standard deviation only:

$$z_{tj} = \frac{x_{tj} - \bar{x}_j}{s_j}. \tag{1.4}$$

The split is chronological: RELIANCE trains on days before 2018-01-01 and tests on the rest. Tuning uses five expanding-window folds inside the training days, so a model never sees a day later than the one it predicts.

---

# Unit 2

## 2. Next-day closing price

Target B is the next-day return $t_n = r_{n+1}$, and the price forecast is $\hat{A}_{n+1} = A_n e^{\hat{t}_n}$. Target A is the next-day price itself, $t_n = A_{n+1}$. The baseline is persistence, $\hat{A}_{n+1} = A_n$.

### Least squares and maximum likelihood

The model is linear with Gaussian noise of precision $\beta$:

$$t_n = \mathbf{w}^\top\boldsymbol{\phi}_n + \varepsilon_n,\qquad \varepsilon_n \sim \mathcal{N}(0, \beta^{-1}), \tag{2.1}$$

with $\boldsymbol{\phi}_n = (1, \mathbf{x}_n^\top)^\top$. The log-likelihood is

$$\ln p(\mathbf{t}\mid\mathbf{w},\beta) = \frac{N}{2}\ln\beta - \frac{N}{2}\ln 2\pi - \frac{\beta}{2}\sum_n \big(t_n - \mathbf{w}^\top\boldsymbol{\phi}_n\big)^2. \tag{2.2}$$

Setting the gradient with respect to $\mathbf{w}$ to zero gives the normal equations, and setting the derivative with respect to $\beta$ to zero gives the noise estimate:

$$\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\,\mathbf{w} = \boldsymbol{\Phi}^\top\mathbf{t},\qquad \frac{1}{\beta} = \frac{1}{N}\sum_n\big(t_n - \mathbf{w}^\top\boldsymbol{\phi}_n\big)^2. \tag{2.3}$$

The script solves (2.3) by Cholesky factorisation. To show that least squares and maximum likelihood are the same, it also maximises (2.2) by gradient ascent and compares the weights:

$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \eta\,\frac{1}{N}\big(\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\,\mathbf{w}^{(k)} - \boldsymbol{\Phi}^\top\mathbf{t}\big),\qquad \eta = \frac{1}{\lambda_{\max}\big(\tfrac{1}{N}\boldsymbol{\Phi}^\top\boldsymbol{\Phi}\big)}. \tag{2.4}$$

### Robust regression (Huber)

Squared error lets a few extreme days dominate the fit. The Huber objective, with a jointly estimated scale $\sigma$ and residuals $e_n = t_n - \mathbf{w}^\top\boldsymbol{\phi}_n$, is

$$\min_{\mathbf{w},\,\sigma}\ N\sigma + \sum_n \sigma\,H_\epsilon\!\Big(\frac{e_n}{\sigma}\Big) + \alpha\lVert\mathbf{w}\rVert^2,\qquad H_\epsilon(z) = \begin{cases} z^2 & |z| \le \epsilon \\ 2\epsilon|z| - \epsilon^2 & |z| > \epsilon. \end{cases} \tag{2.5}$$

For fixed $\sigma$ this is a weighted least-squares problem. With $v_n = 1$ when $|e_n| \le \epsilon\sigma$ and $v_n = \epsilon\sigma/|e_n|$ otherwise,

$$\big(\boldsymbol{\Phi}^\top\mathbf{V}\boldsymbol{\Phi} + \sigma\alpha\mathbf{I}'\big)\,\mathbf{w} = \boldsymbol{\Phi}^\top\mathbf{V}\mathbf{t}, \tag{2.6}$$

where $\mathbf{I}'$ is the identity with a zero for the intercept. For fixed $\mathbf{w}$, setting the derivative with respect to $\sigma$ to zero gives

$$\sigma^2 = \frac{\sum_{\text{inliers}} e_n^2}{N - \epsilon^2 N_{\text{out}}}. \tag{2.7}$$

The script alternates (2.6) and (2.7) until both stop changing.

### Ridge regression

A Gaussian prior on the weights gives a penalised objective. On centred data, the intercept is left unpenalised:

$$\min_{\mathbf{w}}\ \lVert\mathbf{t} - \mathbf{X}\mathbf{w}\rVert^2 + \alpha\lVert\mathbf{w}\rVert^2 \quad\Rightarrow\quad \big(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I}\big)\mathbf{w} = \mathbf{X}^\top\mathbf{t}. \tag{2.8}$$

$\alpha$ is chosen by mean squared error on the time-ordered folds.

### Bayesian linear regression

With the prior

$$p(\mathbf{w}\mid\alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I}), \tag{2.9}$$

the posterior is Gaussian:

$$p(\mathbf{w}\mid\mathbf{t}) = \mathcal{N}(\mathbf{m}, \mathbf{S}),\qquad \mathbf{S}^{-1} = \alpha\mathbf{I} + \beta\mathbf{X}^\top\mathbf{X},\qquad \mathbf{m} = \beta\mathbf{S}\mathbf{X}^\top\mathbf{t}. \tag{2.10}$$

Instead of cross-validation, $\alpha$ and $\beta$ maximise the marginal likelihood. Let $\lambda_i$ be the eigenvalues of $\beta\mathbf{X}^\top\mathbf{X}$, found here by Jacobi rotations. The effective number of parameters is

$$\gamma = \sum_i \frac{\lambda_i}{\alpha + \lambda_i}, \tag{2.11}$$

and the fixed-point updates are

$$\alpha = \frac{\gamma}{\mathbf{m}^\top\mathbf{m}},\qquad \frac{1}{\beta} = \frac{\lVert\mathbf{t} - \mathbf{X}\mathbf{m}\rVert^2}{N - \gamma}. \tag{2.12}$$

The predictive distribution gives a forecast and a 95% interval:

$$p(t\mid\mathbf{x}) = \mathcal{N}\big(\mathbf{m}^\top\mathbf{x},\ \beta^{-1} + \mathbf{x}^\top\mathbf{S}\mathbf{x}\big). \tag{2.13}$$

## 3. Up or down

The target is $y_n = 1$ if $r_{n+1} > 0$ and $0$ otherwise. There are two baselines: always predict the majority class of the training days, and predict the same direction as today.

### Discriminant function and generative models

LDA assumes Gaussian classes with a shared covariance:

$$p(\mathbf{x}\mid y=k) = \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}),\qquad \boldsymbol{\Sigma} = \frac{1}{N}\sum_k\sum_{n\in k}(\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^\top. \tag{2.14}$$

The log-odds are then linear in $\mathbf{x}$, which gives the discriminant:

$$y(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + w_0,\qquad \mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0),\qquad w_0 = -\tfrac12(\boldsymbol{\mu}_1 + \boldsymbol{\mu}_0)^\top\mathbf{w} + \ln\frac{\pi_1}{\pi_0}. \tag{2.15}$$

Gaussian naive Bayes assumes the features are independent within a class:

$$\ln p(y=k\mid\mathbf{x}) = \ln\pi_k - \frac12\sum_j\Big[\ln 2\pi\sigma_{kj}^2 + \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2}\Big] + \text{const}. \tag{2.16}$$

### Logistic regression and the Laplace approximation

The discriminative model is $p(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\boldsymbol{\phi})$, with $\sigma(a) = 1/(1 + e^{-a})$. With a Gaussian prior on every weight except the intercept, the negative log-posterior is

$$\Psi(\mathbf{w}) = -\sum_n\big[y_n\ln\sigma_n + (1 - y_n)\ln(1 - \sigma_n)\big] + \frac{\alpha}{2}\mathbf{w}^\top\mathbf{I}'\mathbf{w}. \tag{2.17}$$

Its gradient and Hessian give Newton's method, also called iteratively reweighted least squares:

$$\mathbf{g} = \boldsymbol{\Phi}^\top(\boldsymbol{\sigma} - \mathbf{y}) + \alpha\mathbf{I}'\mathbf{w},\qquad \mathbf{H} = \boldsymbol{\Phi}^\top\mathbf{R}\boldsymbol{\Phi} + \alpha\mathbf{I}',\qquad \mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}\mathbf{g}, \tag{2.18}$$

with $\mathbf{R} = \mathrm{diag}\big(\sigma_n(1 - \sigma_n)\big)$. Plain logistic regression stops at $\mathbf{w}_{\text{MAP}}$, with $\alpha = 1/C$.

The Bayesian version replaces the posterior with a Gaussian centred at the optimum, $q(\mathbf{w}) = \mathcal{N}(\mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})$. The activation $a = \mathbf{w}^\top\boldsymbol{\phi}$ then has mean $\mu_a = \mathbf{w}_{\text{MAP}}^\top\boldsymbol{\phi}$ and variance $\sigma_a^2 = \boldsymbol{\phi}^\top\mathbf{H}^{-1}\boldsymbol{\phi}$, and the probit approximation gives the predictive probability:

$$p(y=1\mid\mathbf{x}) \approx \sigma\big(\kappa(\sigma_a^2)\,\mu_a\big),\qquad \kappa(\sigma_a^2) = \big(1 + \pi\sigma_a^2/8\big)^{-1/2}. \tag{2.19}$$

### Comparing forecasts

Two price forecasts are compared with the Diebold–Mariano test. With loss differential $d_n = e_{1n}^2 - e_{0n}^2$,

$$\mathrm{DM} = \frac{\bar{d}}{\sqrt{\hat{v}/N}},\qquad \hat{v} = \hat\gamma_0 + 2\sum_{l=1}^{5}\Big(1 - \frac{l}{6}\Big)\hat\gamma_l, \tag{2.20}$$

where $\hat\gamma_l$ is the sample autocovariance of $d_n$ at lag $l$.

### Kernels and the support vector machine

A kernel computes an inner product in a feature space without building it. For $\mathbf{x}, \mathbf{z} \in \mathbb{R}^2$, the feature map $\boldsymbol{\varphi}(\mathbf{x}) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)$ satisfies $\boldsymbol{\varphi}(\mathbf{x})^\top\boldsymbol{\varphi}(\mathbf{z}) = (\mathbf{x}^\top\mathbf{z})^2$, and `unit2.py` prints a worked example. The RBF kernel is

$$k(\mathbf{x}, \mathbf{z}) = \exp\big(-\gamma\lVert\mathbf{x} - \mathbf{z}\rVert^2\big). \tag{2.21}$$

With labels $s_n \in \{-1, +1\}$, the soft-margin SVM is solved in its dual:

$$\min_{\mathbf{a}}\ \frac12\sum_{n,m}a_na_ms_ns_mk(\mathbf{x}_n,\mathbf{x}_m) - \sum_n a_n\quad\text{subject to}\quad 0 \le a_n \le C,\ \ \sum_n a_ns_n = 0. \tag{2.22}$$

Sequential minimal optimisation changes two multipliers at a time, which keeps the equality constraint satisfied. With gradient $G_n$ of the objective, choose $i$ as the most violating multiplier and $j$ by the largest guaranteed decrease. Then, with $b = -s_iG_i + s_jG_j$ and $q = k_{ii} + k_{jj} - 2k_{ij}$,

$$a_i \leftarrow a_i + s_i\frac{b}{q},\qquad a_j \leftarrow a_j - s_j\frac{b}{q}, \tag{2.23}$$

followed by clipping to $[0, C]$. The loop stops when the largest violation falls below $10^{-3}$. The decision function is $f(\mathbf{x}) = \sum_n a_ns_nk(\mathbf{x}_n, \mathbf{x}) - \rho$.

The linear SVM minimises $\tfrac12\lVert\mathbf{w}\rVert^2 + C\sum_n\max(0, 1 - s_n\mathbf{w}^\top\mathbf{x}_n)$, with the bias added as one more feature. Dual coordinate descent updates one multiplier at a time:

$$a_n \leftarrow \mathrm{clip}_{[0,C]}\Big(a_n - \frac{s_n\mathbf{w}^\top\mathbf{x}_n - 1}{\mathbf{x}_n^\top\mathbf{x}_n}\Big),\qquad \mathbf{w} \leftarrow \mathbf{w} + \Delta a_n\,s_n\mathbf{x}_n. \tag{2.24}$$

### Evaluation

Accuracy on the test days is compared with the majority rate $p_0$ by the exact binomial test, $P(X \ge k)$ for $X \sim \mathrm{Bin}(N, p_0)$. AUC is computed from ranks:

$$\mathrm{AUC} = \frac{\sum_{n: y_n=1}\mathrm{rank}_n - N_1(N_1 + 1)/2}{N_1N_0}. \tag{2.25}$$

---

# Unit 3

## 4. Clustering stocks

Each stock $i$ is described by

$$\mathbf{x}_i = (\text{annual return},\ \text{annual volatility},\ \beta_i,\ \text{skewness},\ \text{excess kurtosis},\ \text{maximum drawdown}), \tag{3.1}$$

standardised across the 49 stocks. A second space uses return and volatility only. Here $\beta_i = \mathrm{cov}(r_i, r_m)/\mathrm{var}(r_m)$, where $r_m$ is the equal-weighted market return.

### K-means

K-means minimises $J = \sum_i\sum_k h_{ik}\lVert\mathbf{x}_i - \boldsymbol{\mu}_k\rVert^2$ with hard assignments $h_{ik} \in \{0, 1\}$. Each step can only lower $J$, so the loop ends:

$$h_{ik} = 1 \iff k = \arg\min_j\lVert\mathbf{x}_i - \boldsymbol{\mu}_j\rVert^2, \tag{3.2}$$

$$\boldsymbol{\mu}_k = \frac{\sum_i h_{ik}\mathbf{x}_i}{\sum_i h_{ik}}. \tag{3.3}$$

It can stop at a local minimum, so the script restarts many times from k-means++ starting points and keeps the lowest $J$.

### Gaussian mixtures and EM

$$p(\mathbf{x}) = \sum_k\pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k). \tag{3.4}$$

**E-step.** The responsibilities are the posterior probabilities of the latent component:

$$\rho_{ik} = \frac{\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j\pi_j\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}, \tag{3.5}$$

$$\ln p(\mathbf{X}) = \sum_i\ln\sum_k\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k). \tag{3.6}$$

**M-step.** With $N_k = \sum_i\rho_{ik}$,

$$\pi_k = \frac{N_k}{N}, \tag{3.7}$$

$$\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_i\rho_{ik}\mathbf{x}_i, \tag{3.8}$$

$$\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_i\rho_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^\top + \varepsilon\mathbf{I}. \tag{3.9}$$

Four covariance families restrict (3.9): full, tied (one shared matrix), diagonal, and spherical.

**The alternative view.** For any distribution $q(\mathbf{Z})$ over the latent assignments,

$$\ln p(\mathbf{X}\mid\boldsymbol\theta) = \mathcal{L}(q, \boldsymbol\theta) + \mathrm{KL}\big(q\,\Vert\,p(\mathbf{Z}\mid\mathbf{X}, \boldsymbol\theta)\big),\qquad \mathcal{L}(q, \boldsymbol\theta) = \sum_{\mathbf{Z}} q(\mathbf{Z})\ln\frac{p(\mathbf{X}, \mathbf{Z}\mid\boldsymbol\theta)}{q(\mathbf{Z})}. \tag{3.10}$$

The E-step sets $q$ to the posterior, which makes the KL term zero and the bound equal to the log-likelihood. The M-step raises the bound with $q$ fixed. So the log-likelihood never falls, and the script checks both facts.

**K-means as a limit.** Take equal weights and a fixed covariance $\boldsymbol\Sigma_k = \varepsilon\mathbf{I}$. Then

$$\rho_{ik} = \frac{\exp(-\lVert\mathbf{x}_i - \boldsymbol\mu_k\rVert^2/2\varepsilon)}{\sum_j\exp(-\lVert\mathbf{x}_i - \boldsymbol\mu_j\rVert^2/2\varepsilon)} \xrightarrow{\ \varepsilon\to0\ } h_{ik}. \tag{3.11}$$

**Choosing the model.** With $p$ free parameters and maximised likelihood $\hat{L}$,

$$\mathrm{BIC} = -2\ln\hat{L} + p\ln N,\qquad \mathrm{AIC} = -2\ln\hat{L} + 2p. \tag{3.12}$$

### Hierarchical clustering

Two dissimilarity measures are used, Euclidean distance on the features and correlation distance on the return series:

$$d_{ij} = \lVert\mathbf{x}_i - \mathbf{x}_j\rVert,\qquad d_{ij} = \sqrt{2(1 - \rho_{ij})}. \tag{3.13}$$

Starting from single stocks, the closest pair of clusters $a, b$ is merged. The distance from the new cluster to any other cluster $c$ follows the Lance–Williams updates. For complete and average linkage,

$$d_{(ab)c} = \max(d_{ac}, d_{bc}),\qquad d_{(ab)c} = \frac{n_ad_{ac} + n_bd_{bc}}{n_a + n_b}. \tag{3.14}$$

For Ward linkage,

$$d_{(ab)c} = \sqrt{\frac{(n_a + n_c)d_{ac}^2 + (n_b + n_c)d_{bc}^2 - n_cd_{ab}^2}{n_a + n_b + n_c}}. \tag{3.15}$$

How well the tree keeps the original distances is measured by the cophenetic correlation, where $h_{ij}$ is the height at which $i$ and $j$ first join:

$$c = \mathrm{corr}\big(d_{ij},\ h_{ij}\big). \tag{3.16}$$

### Cluster evaluation

Let $a_i$ be the mean distance from stock $i$ to its own cluster and $b_i$ the smallest mean distance to another cluster. The silhouette is

$$s = \frac{1}{N}\sum_i\frac{b_i - a_i}{\max(a_i, b_i)}. \tag{3.21}$$

With $S_k$ the mean distance of a cluster's stocks to its centre, the Davies–Bouldin index is

$$\mathrm{DB} = \frac{1}{K}\sum_k\max_{j\ne k}\frac{S_k + S_j}{\lVert\boldsymbol\mu_k - \boldsymbol\mu_j\rVert}. \tag{3.22}$$

Agreement with the industry labels uses the adjusted Rand index on a contingency table $n_{ij}$ with margins $a_i, b_j$:

$$\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - E}{\tfrac12\big[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\big] - E},\qquad E = \frac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{N}{2}}. \tag{3.23}$$

## 5. Market factors

### Principal component analysis

Let $\mathbf{X}$ hold the standardised returns of all $P$ stocks over the days when all of them traded. PCA finds orthogonal directions of largest variance, which are the eigenvectors of the sample covariance:

$$\mathbf{S} = \frac{1}{N-1}\mathbf{X}^\top\mathbf{X},\qquad \mathbf{S}\mathbf{u}_j = \lambda_j\mathbf{u}_j. \tag{3.17}$$

The eigenvalues come from Jacobi rotations in `matrix.py`. Component $j$ explains a share $\lambda_j/\sum_k\lambda_k$ of the variance, and the loading of stock $i$ on it is $u_{ij}\sqrt{\lambda_j}$.

The number of latent dimensions is examined by four rules: 80% of the variance; the Kaiser rule $\lambda_j > 1$; parallel analysis, which keeps components above the 95th percentile of eigenvalues from data with each column shuffled; and the broken stick, which compares each share with $b_k = \tfrac1P\sum_{m=k}^P\tfrac1m$.

Probabilistic PCA treats the discarded directions as isotropic noise, with the maximum-likelihood solution

$$\sigma^2 = \frac{1}{P - K}\sum_{j>K}\lambda_j,\qquad \boldsymbol\Lambda = \mathbf{U}_K\big(\mathbf{L}_K - \sigma^2\mathbf{I}\big)^{1/2}. \tag{3.18}$$

### Factor analysis

Factor analysis gives each stock its own noise variance:

$$\mathbf{x} = \boldsymbol\Lambda\mathbf{z} + \boldsymbol\mu + \boldsymbol\varepsilon,\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0}, \mathbf{I}),\quad \boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0}, \boldsymbol\Psi),\quad \boldsymbol\Psi\text{ diagonal},\quad \mathbf{C} = \boldsymbol\Lambda\boldsymbol\Lambda^\top + \boldsymbol\Psi. \tag{3.19}$$

EM alternates between the posterior of $\mathbf{z}$ and the parameters. With $\mathbf{G} = (\mathbf{I} + \boldsymbol\Lambda^\top\boldsymbol\Psi^{-1}\boldsymbol\Lambda)^{-1}$ and $\mathbf{B} = \mathbf{G}\boldsymbol\Lambda^\top\boldsymbol\Psi^{-1}$,

$$\mathbb{E}[\mathbf{z}\mathbf{z}^\top] = \mathbf{G} + \mathbf{B}\mathbf{S}\mathbf{B}^\top,\qquad \boldsymbol\Lambda \leftarrow \mathbf{S}\mathbf{B}^\top\,\mathbb{E}[\mathbf{z}\mathbf{z}^\top]^{-1},\qquad \boldsymbol\Psi \leftarrow \mathrm{diag}\big(\mathbf{S} - \boldsymbol\Lambda\mathbf{B}\mathbf{S}\big). \tag{3.20}$$

The log-likelihood uses the determinant lemma and the Woodbury identity, so only a $K \times K$ matrix has to be factorised.

---

## 6. Where each algorithm is implemented

| Algorithm | Equations | Function |
|---|---|---|
| Least squares, ridge | 2.3, 2.8 | `unit2.fit_ridge` |
| Least squares as maximum likelihood | 2.2, 2.4 | `unit2.ols_as_mle` |
| Huber regression | 2.5–2.7 | `unit2.fit_huber` |
| Bayesian linear regression | 2.9–2.13 | `unit2.blr_evidence`, `unit2.blr_predict` |
| LDA | 2.14, 2.15 | `unit2.fit_lda` |
| Gaussian naive Bayes | 2.16 | `unit2.fit_gnb`, `unit2.gnb_prob` |
| Logistic regression, Laplace approximation | 2.17–2.19 | `unit2.laplace_fit`, `unit2.laplace_pred` |
| RBF kernel SVM | 2.21–2.23 | `unit2.rbf_row`, `unit2.fit_rbf_svm` |
| Linear SVM | 2.24 | `unit2.fit_linear_svm` |
| K-means | 3.2, 3.3 | `unit3.lloyd`, `unit3.kmeans_pp` |
| GMM by EM | 3.4–3.10 | `unit3.e_step`, `unit3.m_step`, `unit3.em_gmm` |
| K-means as the EM limit | 3.11 | `unit3.soft_kmeans` |
| Hierarchical clustering | 3.13–3.16 | `unit3.linkage`, `unit3.cut`, `unit3.cophenetic` |
| Silhouette, Davies–Bouldin, ARI | 3.21–3.23 | `unit3.silhouette`, `unit3.davies_bouldin`, `unit3.adjusted_rand` |
| PCA | 3.17 | `matrix.eigh`, `unit3.main` |
| Probabilistic PCA, factor analysis | 3.18–3.20 | `unit3.main`, `unit3.factor_analysis` |

`COMPARISON.md` lists the results of these scripts next to the notebook's scikit-learn results.
