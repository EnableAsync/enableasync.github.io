---
title: 小样本微调研究综述
date: 2025-04-16 11:00:00
tags:
---

# 小样本微调领域（2022年至今）的代表性论文

以下列出几篇2022年以来在计算机视觉领域关于**小样本微调（Few-Shot Fine-Tuning）**的代表性论文。每篇包括论文标题、作者与年份、主要贡献摘要、公开源码情况以及发表的会议或期刊，涵盖图像分类、目标检测、语义分割等方向：

1. **论文标题**: *Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference*  
   **作者与年份**: Shell Xu Hu，Da Li，Jan Stühmer，Minyoung Kim，Timothy M. Hospedales，2022年  
   **摘要**: 提出利用一个简单但有效的三阶段小样本学习流程，包括**在外部数据上预训练**、**小样本任务的元训练**以及**特定任务的微调**。作者探讨了：(1) 外部数据预训练如何提升小样本学习性能，(2) 利用最新Transformer架构的效果，以及 (3) 如何充分利用微调步骤。实验表明，一个基于Transformer的简单管线在Mini-ImageNet、CIFAR-FS、CDFSL和Meta-Dataset等标准小样本分类基准上取得了惊人的性能。这一结果强调了**将大规模外部数据与微调相结合**的策略对提升小样本学习效果的重要作用。**代码**: 提供了开源代码 (GitHub: `hushell/pmf_cvpr22`)。  
   **发表**: CVPR 2022

2. **论文标题**: *Conditional Prompt Learning for Vision-Language Models*  
   **作者与年份**: Kaiyang Zhou，Jingkang Yang，Chen Change Loy，Ziwei Liu，2022年  
   **摘要**: 针对CLIP等大型视觉-语言模型的小样本适应问题，作者提出了**条件上下文优化（CoCoOp）**方法。此前的方法CoOp通过学习一组固定的可调节提示（prompt）在小样本下取得了显著效果，但存在**对训练类别过拟合、无法泛化到新类别**的问题。CoCoOp通过引入一个轻量级网络为每张图像**动态生成输入相关的提示向量**，使提示根据图像自适应变化，从而减轻了类别移位带来的影响。实验表明，相比静态提示，**动态提示的CoCoOp在未见类别上的泛化性能明显优于CoOp**，在跨数据集迁移和域泛化上也更出色。**代码**: 提供了开源代码 (GitHub: `KaiyangZhou/CoOp`)。  
   **发表**: CVPR 2022

3. **论文标题**: *Singular Value Fine-tuning: Few-shot Segmentation requires Few-parameters Fine-tuning*  
   **作者与年份**: Yanpeng Sun，Qiang Chen，Xiangyu He 等，2022年  
   **摘要**: 提出了一种针对小样本语义分割的**奇异值微调（SVF）**方法。传统方法为了避免过拟合通常**冻结预训练骨干网络**，仅在其上添加复杂的特征融合或度量模块。作者重新思考这一范式，发现**只微调骨干网络的一小部分参数**同样可以避免过拟合并提升对新类别的泛化性能。具体而言，SVF将预训练卷积权重用奇异值分解拆解为三个矩阵，只**训练中间的奇异值参数**，冻结奇异向量（两侧矩阵）。这种策略在**保持预训练语义信息**的同时**调整特征以适应新类**。在Pascal-5<sup>i</sup>和COCO-20<sup>i</sup>的1-shot和5-shot分割基准上，SVF取得了**当前最佳性能**，显著优于仅冻结骨干的方案。这证明了微调骨干的一小部分参数可以在小样本分割中取得更好效果。**代码**: 提供了官方开源实现 (GitHub: `zechao-li/SVF-pytorch`)。  
   **发表**: NeurIPS 2022

4. **论文标题**: *Few-Shot Recognition via Stage-Wise Augmented Finetuning*  
   **作者与年份**: Tian Liu，Huixin Zhang，Shubham Parashar，Shu Kong，2024年  
   **摘要**: 提出了一种**分阶段增强微调 (SWAT)** 方法来提升小样本图像识别性能。作者将**检索增强学习 (retrieval-augmented learning, RAL)**引入Few-Shot场景：首先从外部大型图像-文本数据中检索与新类别相关的样本用于辅助训练。研究发现：直接在大量检索数据上微调预训练的视觉-语言模型（VLM，如CLIP）由于数据分布不均衡和域偏差，效果几乎不超越零样本基线；仅使用少量标注样本微调反而已经**大幅超越现有方法**；而**将检索数据与标注的少样本混合微调**可以进一步提升性能。为此，SWAT方法分两阶段进行微调：**第一阶段**在“检索数据+少量标注数据”混合集上端到端微调整个模型，**第二阶段**仅使用少量标注数据重新训练分类头，以缓解不均衡和域差异。这种分阶段策略使模型性能显著提升，在标准few-shot识别基准上**比此前最佳方法准确率高出约10%**。**代码**: 提供了开源代码 (GitHub: `tian1327/SWAT`)。  
   **发表**: CVPR 2024

5. **论文标题**: *Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning*  
   **作者与年份**: Samyadeep Basu，Shell Xu Hu，Daniela Massiceti，Soheil Feizi，2024年  
   **摘要**: 本文对**参数高效的小样本微调**方法进行了大规模系统的实证分析，在Meta-Dataset和ORBIT等大型few-shot基准上进行了超过1800次对比实验。研究的主要发现有两点：(i) **仅微调LayerNorm层参数（LN-Tune）**在few-shot图像分类中是一个**极其强大的基线**，无论主干ViT是经自监督还是监督预训练获得。(ii) 对于自监督预训练的ViT模型，**仅为每个注意力矩阵引入可学习的缩放参数（AttnScale）并配合一个轻量的残差适配器模块（DRA）**即可达到当前最优性能，同时参数规模比微调全部参数减少约9倍。这些结果为few-shot微调提供了强基线，并**呼吁重新思考现有参数高效微调方法的设计**。*（注：作者在文中未公布官方源码）*。  
   **发表**: AAAI 2024

6. **论文标题**: *Few-Shot Object Detection with Foundation Models*  
   **作者与年份**: Guangxing Han，Ser-Nam Lim，2024年  
   **摘要**: 探索了利用**基础模型**解决Few-Shot目标检测（FSOD）的新方法。作者使用两个大型预训练模型：其一是**视觉基础模型DINOv2**，作为**冻结的特征提取骨干**，提供强大的通用视觉特征；其二是**大语言模型 (LLM)**，用于对检测提议区域进行**上下文化的小样本分类**。具体而言，采用Transformer架构（如Deformable DETR）生成**类感知的候选区域**，然后构造文本提示将所有候选及类别标签嵌入到LLM中，通过LLM强大的**上下文推理**能力，对每个候选进行分类。这种方式利用提议之间以及提议与类别之间的关系，大幅提升了少样本情况下的检测准确率。模型在PASCAL VOC和MS COCO等**Few-Shot检测基准**上取得了当前最先进的性能。*（注：源码暂未公开，但模型基于Facebook开放的Detrex检测代码库实现）*。  
   **发表**: CVPR 2024

