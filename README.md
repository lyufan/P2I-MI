# P2I-MI
[ECCV 2024] "Prediction Exposes Your Face: Black-box Model Inversion via Prediction Alignment"

Arxiv: [Prediction Exposes Your Face: Black-box Model Inversion via Prediction Alignment](https://arxiv.org/abs/2407.08127)

> **Prediction Exposes Your Face: Black-box Model Inversion via Prediction Alignment**

> Yufan Liu, Wanqian Zhang, Dayan Wu, Zheng Lin, Jingzi Gu, Weiping Wang
> ECCV 2024
>
> **Abstract:**
> Model inversion (MI) attack reconstructs the private training data of a target model given its output, posing a significant threat to deep learning models and data privacy. On one hand, most of existing MI methods focus on searching for latent codes to represent the target identity, yet this iterative optimization-based scheme consumes a huge number of queries to the target model, making it unrealistic especially in black-box scenario. On the other hand, some training-based methods launch an attack through a single forward inference, whereas failing to directly learn high-level mappings from prediction vectors to images. Addressing these limitations, we propose a novel Prediction-to-Image (P2I) method for black-box MI attack. Specifically, we introduce the Prediction Alignment Encoder to map the target model's output prediction into the latent code of StyleGAN. In this way, prediction vector space can be well aligned with the more disentangled latent space, thus establishing a connection between prediction vectors and the semantic facial features. During the attack phase, we further design the Aligned Ensemble Attack scheme to integrate complementary facial attributes of target identity for better reconstruction. Experimental results show that our method outperforms other SOTAs, e.g.,compared with RLB-MI, our method improves attack accuracy by 8.5% and reduces query numbers by 99% on dataset CelebA.

![](https://img.shields.io/badge/arXiv-2407.08127-AE2525)

![](https://github.com/lyufan/P2I-MI/blob/main/assets/figure2.pdf)
![](https://github.com/lyufan/P2I-MI/blob/main/assets/figure3.pdf)

