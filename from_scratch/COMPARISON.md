# No-library results compared with the notebook

Written by `from_scratch/check.py`. The notebook uses scikit-learn and SciPy; the scripts here use neither, nor NumPy or pandas.

**177 of 191 quantities agree.** Modules imported: `array`, `ast`, `csv`, `datetime`, `json`, `math`, `multiprocessing`, `operator`, `os`, `random`, `sys`, `time`. All come with Python.

| Part | Quantities | Agree |
|---|---:|---:|
| Unit 2: next-day close | 44 | 44 |
| Unit 2: up or down | 35 | 35 |
| Unit 3: K-means | 44 | 32 |
| Unit 3: GMM selection | 10 | 8 |
| Unit 3: hierarchical | 39 | 39 |
| Unit 3: PCA and factor analysis | 19 | 19 |

## Where they differ

K-means and EM start from random points and can stop at different local optima, so runs with different random numbers can end in different clusterings. This shows up when 49 stocks are split into many clusters, or when AIC compares models with many parameters. A lower inertia or AIC means that run found the better optimum, and either run can be the one that does. The closed-form and convex models have no such randomness.

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| K-means 6d k=5 inertia | 116.986 | 116.301 | 6.9e-01 |
| K-means 6d k=5 silhouette | 0.251754 | 0.254155 | 2.4e-03 |
| K-means 6d k=5 Davies–Bouldin | 1.09861 | 1.10012 | 1.5e-03 |
| K-means 6d k=6 inertia | 102.531 | 102.838 | 3.1e-01 |
| K-means 6d k=6 silhouette | 0.235527 | 0.23139 | 4.1e-03 |
| K-means 6d k=6 Davies–Bouldin | 1.21433 | 1.21666 | 2.3e-03 |
| K-means 6d k=7 inertia | 92.6779 | 91.6277 | 1.1e+00 |
| K-means 6d k=7 silhouette | 0.240217 | 0.229459 | 1.1e-02 |
| K-means 6d k=7 Davies–Bouldin | 1.18584 | 1.23252 | 4.7e-02 |
| K-means 6d k=8 inertia | 81.4802 | 80.5019 | 9.8e-01 |
| K-means 6d k=8 silhouette | 0.238935 | 0.241734 | 2.8e-03 |
| K-means 6d k=8 Davies–Bouldin | 1.18586 | 1.15979 | 2.6e-02 |
| AIC2 COV | diag | spherical | differs |
| AIC2 K | 5 | 8 | differs |

## Unit 2: next-day close

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| Persistence [baseline|none] RMSE | 30.3051 | 30.3051 | 7.6e-10 |
| Persistence [baseline|none] MAE | 20.8155 | 20.8155 | 1.1e-09 |
| Persistence [baseline|none] MAPE | 1.46929 | 1.46929 | 3.0e-11 |
| OLS [A|RAW] RMSE | 30.3349 | 30.3349 | 4.1e-10 |
| OLS [A|RAW] MAE | 20.8622 | 20.8622 | 4.4e-09 |
| OLS [A|RAW] MAPE | 1.47268 | 1.47268 | 3.8e-10 |
| OLS [B|RAW-returns] RMSE | 30.3619 | 30.3619 | 5.0e-09 |
| OLS [B|RAW-returns] MAE | 20.862 | 20.862 | 3.6e-09 |
| OLS [B|RAW-returns] MAPE | 1.47431 | 1.47431 | 3.4e-10 |
| OLS [B|ENGINEERED] RMSE | 30.3437 | 30.3437 | 2.3e-09 |
| OLS [B|ENGINEERED] MAE | 20.8418 | 20.8418 | 4.6e-09 |
| OLS [B|ENGINEERED] MAPE | 1.47415 | 1.47415 | 3.7e-10 |
| Huber [A|RAW] RMSE | 30.3146 | 30.3147 | 7.6e-05 |
| Huber [A|RAW] MAE | 20.8511 | 20.8512 | 5.6e-05 |
| Huber [A|RAW] MAPE | 1.47221 | 1.47222 | 5.0e-06 |
| Huber [B|ENGINEERED] RMSE | 30.3658 | 30.3658 | 1.9e-05 |
| Huber [B|ENGINEERED] MAE | 20.8597 | 20.8597 | 3.1e-05 |
| Huber [B|ENGINEERED] MAPE | 1.47629 | 1.47629 | 2.1e-06 |
| Ridge [A|RAW] RMSE | 30.3349 | 30.3349 | 4.8e-09 |
| Ridge [A|RAW] MAE | 20.8622 | 20.8622 | 4.7e-09 |
| Ridge [A|RAW] MAPE | 1.47268 | 1.47268 | 2.4e-10 |
| Ridge [B|RAW-returns] RMSE | 30.3068 | 30.3068 | 2.9e-09 |
| Ridge [B|RAW-returns] MAE | 20.813 | 20.813 | 1.6e-10 |
| Ridge [B|RAW-returns] MAPE | 1.47042 | 1.47042 | 1.2e-10 |
| Ridge [B|ENGINEERED] RMSE | 30.2822 | 30.2822 | 2.7e-09 |
| Ridge [B|ENGINEERED] MAE | 20.7857 | 20.7857 | 1.7e-09 |
| Ridge [B|ENGINEERED] MAPE | 1.46882 | 1.46882 | 4.3e-10 |
| Bayesian LR [B|ENGINEERED] RMSE | 30.2915 | 30.2915 | 5.0e-09 |
| Bayesian LR [B|ENGINEERED] MAE | 20.7921 | 20.7921 | 2.8e-09 |
| Bayesian LR [B|ENGINEERED] MAPE | 1.46982 | 1.46982 | 2.3e-11 |
| Bayesian LR [A|RAW] RMSE | 30.3327 | 30.3327 | 1.0e-09 |
| Bayesian LR [A|RAW] MAE | 20.8601 | 20.8601 | 3.3e-09 |
| Bayesian LR [A|RAW] MAPE | 1.47253 | 1.47253 | 1.2e-11 |
| Bayesian LR α | 7.79737e+06 | 7.79737e+06 | 1.0e-04 |
| Bayesian LR β | 2229.34 | 2229.34 | 4.8e-07 |
| Bayesian LR effective parameters γ | 9.53401 | 9.53401 | 3.4e-10 |
| OLS noise precision β | 2239.35 | 2239.35 | 1.6e-08 |
| huber_a tuned ε | 2.5 | 2.5 |  |
| huber_a outlier days | 225 | 225 | 0.0e+00 |
| huber_b_eng tuned ε | 1.35 | 1.35 |  |
| huber_b_eng outlier days | 1629 | 1629 | 0.0e+00 |
| ridge_a tuned α | 1e-05 | 1e-05 | 0.0e+00 |
| ridge_b_eng tuned α | 10000 | 10000 | 0.0e+00 |
| ridge_b_raw tuned α | 3162.28 | 3162.28 | 1.7e-07 |

## Unit 2: up or down

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| Majority class accuracy | 0.519465 | 0.519465 | 5.4e-12 |
| Majority class F1 | 0.683747 | 0.683747 | 1.9e-12 |
| Majority class log-loss | 0.692416 | 0.692416 | 3.1e-11 |
| Same as today accuracy | 0.508516 | 0.508516 | 1.5e-11 |
| Same as today F1 | 0.526932 | 0.526932 | 9.1e-12 |
| Same as today AUC | 0.50777 | 0.50777 | 2.3e-11 |
| LDA accuracy | 0.521898 | 0.521898 | 1.9e-11 |
| LDA F1 | 0.605025 | 0.605025 | 2.8e-11 |
| LDA AUC | 0.506122 | 0.506122 | 7.9e-12 |
| LDA log-loss | 0.695803 | 0.695803 | 1.4e-12 |
| Gaussian NB accuracy | 0.508516 | 0.508516 | 1.5e-11 |
| Gaussian NB F1 | 0.488608 | 0.488608 | 3.7e-11 |
| Gaussian NB AUC | 0.510592 | 0.510592 | 2.7e-12 |
| Gaussian NB log-loss | 0.732139 | 0.732139 | 1.1e-11 |
| Logistic regression accuracy | 0.519465 | 0.519465 | 5.4e-12 |
| Logistic regression F1 | 0.683747 | 0.683747 | 1.9e-12 |
| Logistic regression AUC | 0.502493 | 0.502428 | 6.5e-05 |
| Logistic regression log-loss | 0.692419 | 0.692419 | 5.7e-08 |
| Bayesian logistic (Laplace) accuracy | 0.519465 | 0.519465 | 5.4e-12 |
| Bayesian logistic (Laplace) F1 | 0.683747 | 0.683747 | 1.9e-12 |
| Bayesian logistic (Laplace) AUC | 0.502452 | 0.502452 | 4.7e-11 |
| Bayesian logistic (Laplace) log-loss | 0.692419 | 0.692419 | 2.3e-11 |
| Linear SVM accuracy | 0.519465 | 0.519465 | 5.4e-12 |
| Linear SVM F1 | 0.553672 | 0.553672 | 1.6e-11 |
| Linear SVM AUC | 0.502357 | 0.502357 | 4.4e-11 |
| RBF SVM accuracy | 0.519465 | 0.519465 | 5.4e-12 |
| RBF SVM F1 | 0.683747 | 0.683747 | 1.9e-12 |
| RBF SVM AUC | 0.504965 | 0.504853 | 1.1e-04 |
| Logistic tuned C | 1e-06 | 1e-06 | 0.0e+00 |
| Bayesian logistic tuned α | 1e+06 | 1e+06 | 0.0e+00 |
| Linear SVM tuned C | 0.0001 | 0.0001 | 0.0e+00 |
| RBF SVM C | 0.1 | 0.1 |  |
| RBF SVM γ | 0.01 | 0.01 |  |
| RBF SVM support vectors | 4286 | 4287 | 1.0e+00 |
| Models significantly above the majority rate | 0 | 0 |  |

## Unit 3: K-means

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| K-means 6d k=2 inertia | 205.843 | 205.843 | 3.5e-08 |
| K-means 6d k=2 silhouette | 0.258498 | 0.258498 | 4.4e-11 |
| K-means 6d k=2 Davies–Bouldin | 1.4271 | 1.4271 | 1.8e-10 |
| K-means 6d k=3 inertia | 165.868 | 165.868 | 4.3e-08 |
| K-means 6d k=3 silhouette | 0.264293 | 0.264293 | 1.5e-11 |
| K-means 6d k=3 Davies–Bouldin | 1.18891 | 1.18891 | 3.4e-10 |
| K-means 6d k=4 inertia | 136.576 | 136.576 | 1.2e-08 |
| K-means 6d k=4 silhouette | 0.252706 | 0.252706 | 4.0e-11 |
| K-means 6d k=4 Davies–Bouldin | 1.15618 | 1.15618 | 2.2e-10 |
| K-means 6d k=5 inertia | 116.986 | 116.301 | 6.9e-01 |
| K-means 6d k=5 silhouette | 0.251754 | 0.254155 | 2.4e-03 |
| K-means 6d k=5 Davies–Bouldin | 1.09861 | 1.10012 | 1.5e-03 |
| K-means 6d k=6 inertia | 102.531 | 102.838 | 3.1e-01 |
| K-means 6d k=6 silhouette | 0.235527 | 0.23139 | 4.1e-03 |
| K-means 6d k=6 Davies–Bouldin | 1.21433 | 1.21666 | 2.3e-03 |
| K-means 6d k=7 inertia | 92.6779 | 91.6277 | 1.1e+00 |
| K-means 6d k=7 silhouette | 0.240217 | 0.229459 | 1.1e-02 |
| K-means 6d k=7 Davies–Bouldin | 1.18584 | 1.23252 | 4.7e-02 |
| K-means 6d k=8 inertia | 81.4802 | 80.5019 | 9.8e-01 |
| K-means 6d k=8 silhouette | 0.238935 | 0.241734 | 2.8e-03 |
| K-means 6d k=8 Davies–Bouldin | 1.18586 | 1.15979 | 2.6e-02 |
| K-means 2d k=2 inertia | 60.7083 | 60.7083 | 1.2e-09 |
| K-means 2d k=2 silhouette | 0.354886 | 0.354886 | 3.6e-11 |
| K-means 2d k=2 Davies–Bouldin | 1.11809 | 1.11809 | 1.9e-10 |
| K-means 2d k=3 inertia | 39.9142 | 39.9142 | 4.5e-09 |
| K-means 2d k=3 silhouette | 0.384353 | 0.384353 | 3.6e-11 |
| K-means 2d k=3 Davies–Bouldin | 0.919705 | 0.919705 | 3.3e-11 |
| K-means 2d k=4 inertia | 28.9285 | 28.9285 | 2.6e-09 |
| K-means 2d k=4 silhouette | 0.409431 | 0.409431 | 3.7e-11 |
| K-means 2d k=4 Davies–Bouldin | 0.748478 | 0.748478 | 7.0e-12 |
| K-means 2d k=5 inertia | 22.8156 | 22.8156 | 1.0e-09 |
| K-means 2d k=5 silhouette | 0.381009 | 0.381009 | 4.4e-11 |
| K-means 2d k=5 Davies–Bouldin | 0.786724 | 0.786724 | 2.7e-11 |
| K-means 2d k=6 inertia | 18.4199 | 18.4199 | 4.2e-09 |
| K-means 2d k=6 silhouette | 0.370888 | 0.370888 | 3.0e-11 |
| K-means 2d k=6 Davies–Bouldin | 0.693977 | 0.693977 | 2.1e-11 |
| K-means 2d k=7 inertia | 14.3169 | 14.3169 | 2.6e-09 |
| K-means 2d k=7 silhouette | 0.378558 | 0.378558 | 2.6e-11 |
| K-means 2d k=7 Davies–Bouldin | 0.708303 | 0.708303 | 2.0e-11 |
| K-means 2d k=8 inertia | 11.1945 | 11.1945 | 1.9e-10 |
| K-means 2d k=8 silhouette | 0.398529 | 0.398529 | 1.7e-11 |
| K-means 2d k=8 Davies–Bouldin | 0.65908 | 0.65908 | 3.8e-11 |
| Chosen k, 6 features | 3 | 3 |  |
| Chosen k, 2 features | 4 | 4 |  |

## Unit 3: GMM selection

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| BIC COV | tied | tied |  |
| BIC K | 1 | 1 |  |
| AIC COV | full | full |  |
| AIC K | 8 | 8 |  |
| BIC2 COV | spherical | spherical |  |
| BIC2 K | 1 | 1 |  |
| AIC2 COV | diag | spherical | differs |
| AIC2 K | 5 | 8 | differs |
| BIC of the chosen model | 796.862 | 796.862 | 1.9e-09 |
| EM log-likelihood never falls | True | True |  |

## Unit 3: hierarchical

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| euclidean ward cophenetic | 0.488416 | 0.488416 | 1.8e-11 |
| euclidean complete cophenetic | 0.568401 | 0.568401 | 4.1e-11 |
| euclidean average cophenetic | 0.671874 | 0.671874 | 3.2e-11 |
| correlation ward cophenetic | 0.612019 | 0.612019 | 1.2e-11 |
| correlation complete cophenetic | 0.701834 | 0.701834 | 4.3e-11 |
| correlation average cophenetic | 0.86664 | 0.86664 | 2.9e-11 |
| Ward 6d k=2 silhouette | 0.195005 | 0.195005 | 2.1e-11 |
| Ward 6d k=2 ARI with industry | 0.052586 | 0.052586 | 3.0e-12 |
| Ward 6d k=3 silhouette | 0.195356 | 0.195356 | 2.5e-11 |
| Ward 6d k=3 ARI with industry | 0.0658355 | 0.0658355 | 1.8e-12 |
| Ward 6d k=4 silhouette | 0.234364 | 0.234364 | 4.6e-11 |
| Ward 6d k=4 ARI with industry | 0.0631927 | 0.0631927 | 3.0e-12 |
| Ward 6d k=5 silhouette | 0.203817 | 0.203817 | 4.7e-11 |
| Ward 6d k=5 ARI with industry | 0.0824109 | 0.0824109 | 2.2e-12 |
| Ward 6d k=6 silhouette | 0.204973 | 0.204973 | 2.2e-11 |
| Ward 6d k=6 ARI with industry | 0.0756769 | 0.0756769 | 2.8e-12 |
| Ward 6d k=7 silhouette | 0.214905 | 0.214905 | 3.2e-11 |
| Ward 6d k=7 ARI with industry | 0.0541494 | 0.0541494 | 4.9e-12 |
| Ward 6d k=8 silhouette | 0.230692 | 0.230692 | 2.7e-11 |
| Ward 6d k=8 ARI with industry | 0.0470994 | 0.0470994 | 7.9e-13 |
| Ward 2d k=2 silhouette | 0.385183 | 0.385183 | 7.6e-13 |
| Ward 2d k=2 ARI with industry | 0.0114843 | 0.0114843 | 3.2e-12 |
| Ward 2d k=3 silhouette | 0.35706 | 0.35706 | 4.8e-11 |
| Ward 2d k=3 ARI with industry | 0.0449091 | 0.0449091 | 3.2e-13 |
| Ward 2d k=4 silhouette | 0.398815 | 0.398815 | 3.1e-11 |
| Ward 2d k=4 ARI with industry | 0.0407874 | 0.0407874 | 3.6e-13 |
| Ward 2d k=5 silhouette | 0.37736 | 0.37736 | 1.5e-11 |
| Ward 2d k=5 ARI with industry | 0.052876 | 0.052876 | 4.4e-12 |
| Ward 2d k=6 silhouette | 0.385443 | 0.385443 | 3.4e-11 |
| Ward 2d k=6 ARI with industry | 0.0129828 | 0.0129828 | 3.9e-13 |
| Ward 2d k=7 silhouette | 0.381348 | 0.381348 | 2.0e-11 |
| Ward 2d k=7 ARI with industry | 0.0167286 | 0.0167286 | 3.7e-14 |
| Ward 2d k=8 silhouette | 0.38552 | 0.38552 | 4.9e-11 |
| Ward 2d k=8 ARI with industry | 0.0221646 | 0.0221646 | 1.4e-12 |
| ARI with industry, km_6d | 0.0289513 | 0.0289513 | 2.9e-12 |
| ARI with industry, hier_6d | 0.0658355 | 0.0658355 | 1.8e-12 |
| ARI with industry, km_2d | 0.0783183 | 0.0783183 | 3.8e-12 |
| ARI with industry, corr_ward | 0.134267 | 0.134267 | 3.7e-12 |
| ARI with industry, corr_average | 0.039314 | 0.039314 | 1.6e-12 |

## Unit 3: PCA and factor analysis

| Quantity | Notebook | No library | Difference |
|---|---:|---:|---:|
| days | 2557 | 2557 |  |
| stocks | 49 | 49 |  |
| n80 | 27 | 27 |  |
| n_kaiser | 8 | 8 |  |
| n_parallel | 6 | 6 |  |
| n_broken | 1 | 1 |  |
| start | 2010-11-05 | 2010-11-05 |  |
| end | 2021-04-30 | 2021-04-30 |  |
| pc1 | 0.3018 | 0.3018 | 2.1e-11 |
| pc2 | 0.0499976 | 0.0499976 | 9.1e-13 |
| pc3 | 0.0355943 | 0.0355943 | 2.4e-12 |
| ev1 | 14.794 | 14.794 | 3.3e-09 |
| pc1_corr | 0.99837 | 0.99837 | 5.0e-11 |
| ll_fa | -155008 | -155008 | 8.1e-05 |
| ll_ppca | -156422 | -156422 | 3.3e-05 |
| sigma2_ppca | 0.602818 | 0.602818 | 3.3e-11 |
| psi_min | 0.279713 | 0.279744 | 3.1e-05 |
| psi_max | 0.829041 | 0.829042 | 9.7e-07 |
| psi_mean | 0.596947 | 0.596945 | 2.6e-06 |
