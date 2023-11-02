This code is the implementation of the paper "Imperceptible Adversarial Attack on S Channel of HSV Colorspace".

[Paper Link](https://ieeexplore.ieee.org/abstract/document/10191049)

## Abstract
Deep neural network models are vulnerable to subtle but adversarial perturbations that alter the model. Adversarial perturbations are typically computed for RGB images and, therefore, are evenly distributed among RGB channels. Compared with RGB images, HSV images can express the Hue, saturation, and brightness more intuitively. We find that the adversarial perturbation in the S-channel ensures a high attack success rate, while the perturbation is small, and the visual quality of the adversarial examples is good. Using this finding, we propose an attack method, SPGD, to improve the visual quality of adversarial examples by generating perturbations on the S-channel. Based on the attack principle of the PGD method, the RGB image was converted into an HSV image. The gradient calculated by the model on the S channel was superimposed on the S channel and then combined with the non-interference H and V channels to convert back to the RGB image. The iteration stops until the attack succeed. We compare the SPGD method with the existing state-of-the-art attack methods. The results show that SPGD minimizes pixel perturbation while maintaining a high attack success rate and achieves the best results in terms of structural similarity, imperceptibility, the minimum number of iterations, and the shortest run time.

## 摘要
深度神经网络模型很容易受到微妙但对抗性扰动的影响，从而改变模型。 对抗性扰动通常针对 RGB 图像进行计算，因此均匀分布在 RGB 通道中。 与RGB图像相比，HSV图像可以更直观地表达色相、饱和度和亮度。 我们发现S通道中的对抗性扰动保证了较高的攻击成功率，同时扰动较小，并且对抗性示例的视觉质量良好。 利用这一发现，我们提出了一种攻击方法 SPGD，通过在 S 通道上生成扰动来提高对抗性示例的视觉质量。 基于PGD方法的攻击原理，将RGB图像转换为HSV图像。 将模型在S通道上计算出的梯度叠加在S通道上，然后与无干扰的H和V通道结合起来转换回RGB图像。 迭代停止，直到攻击成功。 我们将 SPGD 方法与现有最先进的攻击方法进行比较。 结果表明，SPGD在保持较高攻击成功率的同时最大限度地减少了像素扰动，并在结构相似性、不可感知性、最少迭代次数和最短运行时间方面取得了最佳效果。

## How to cite our paper
    @inproceedings{zhu2023imperceptible,
      title={Imperceptible Adversarial Attack on S Channel of HSV Colorspace},
      author={Zhu, Tong and Yin, Zhaoxia and Lyu, Wanli and Zhang, Jiefei and Luo, Bin},
      booktitle={2023 International Joint Conference on Neural Networks (IJCNN)},
      pages={1--7},
      year={2023},
      organization={IEEE}
    }
