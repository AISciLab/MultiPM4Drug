# Multimodal Self-Supervised Learning for Drug Discovery: A Survey and Taxonomy

MultiPM4Drug presents a systematic Review that summarize the foundation of molecular modalities, and revisit the popular network frameworks, self-supervised tasks, training strategies and their application in drug discovery. To promote readers' understanding of multimodal pre-training models in drug discovery, MultiPM4Drug provides the relevant papers that investigated in this Review. 

It is worth noting that this Review highlights the adaptability between various modalities and network frameworks or pre-training tasks. Simultaneously, we systematically compare the difference and relevance between various modalities or pre-training models. The previous works suggest two increasing trends that may be used as the reference points for the future researches. (1) Transformers and graph neural networks are often integrated together as encoders, and then cooperate with multiple pre-training tasks to learn cross-scale molecular representation, thus promoting the performance of drug discovery. (2) Molecule captions as brief biomedical text provide a bridge for collaboration of drug discovery with large language models. Finally, we discuss the challenges of multimodal pre-training models in drug discovery, and explore future opportunities.

![outline](./figures/figure_2.png)

## Table of Contents

- [S2 Database](#s2-database)

- [S3 Multimodal Molecular Representation](#s3-multimodal-molecular-representation)

- [S4 Network Frameworks](#s4-network-frameworks)
  - [S4.1 Deep Neural Networks (DNNs)](#s41-deep-neural-networks-dnns)
  - [S4.2 Convolutional Neural Networks (CNNs)](#s42-convolutional-neural-networks-cnns)
  - [S4.3 Recurrent Neural Network (RNNs)](#s43-recurrent-neural-network-rnns)
  - [S4.4 Graph Neural Networks (GNNs)](#s44-graph-neural-networks-gnns)
  - [S4.5 Transformer](#s45-transformer)

- [S5 Pre-training Task](#s5-pre-training-task)
  - [S5.1 Contrastive Learning](#s51-contrastive-learning)
  - [S5.2 Multi-modal Matching Prediction](#s52-multi-modal-matching-prediction)
  - [S5.3 Masked Prediction](#s53-masked-prediction)
  - [S5.4 Autoregressive Modeling](#s54-autoregressive-modeling)

- [S6 Self-supervised Training Strategy](#s6-self-supervised-training-strategy)

- [S7 Drug Discovery Applications](#s7-drug-discovery-applications)
  - [S7.1 Molecular Generation](#s71-molecular-generation)
  - [S7.2 Molecular Property Prediction](#s72-molecular-property-prediction)
  - [S7.3 DDI Prediction](#s73-ddi-prediction)
  - [S7.4 DTI Prediction](#s74-dti-prediction)
  - [S7.5 Molecule Captioning](#s75-molecule-captioning)

## S2 Database

1. [DrugBank](http://www.drugbank.ca/): DrugBank is comprehensive database containing information on drugs and targets.
2. [ChemDB](https://cdb.ics.uci.edu/): ChemDB is a chemical database that contains nearly 5 million commercially available molecules, along with their predicted or experimentally determined physicochemical properties.
3. [PubChem](https://pubchem.ncbi.nlm.nih.gov/): PubChem provides detailed drug molecule information, including chemical structure and properties, biological activity, toxicity, and more.
4. [ChEMBL](https://www.ebi.ac.uk/chembl/): ChEMBL contains chemical, biological activity and genomic data.
5. [STRING](https://cn.string-db.org/): STRING focuses on protein information, especially the known and predicted protein-protein interactions information.
6. [DisGeNET](https://www.disgenet.org/): DisGeNET is a versatile database of human disease-gene association.
7. [SIDER](http://sideeffects.embl.de/): SIDER focuses on adverse reaction information of marketed medicines.
8. [CTD](https://ctdbase.org/): Comparative Toxicogenomics Database (CTD) contains manually integrated information of chemical molecules, including interactions among molecules, genes diseases and phenotypes.
9. [TTD](https://db.idrblab.net/ttd/): Therapeutic Target Database (TTD) collects information about drugs, targets (proteins and nucleic acids), diseases and pathways.
10. [OMIM](https://www.omim.org/about): Online Mendelian Inheritance in Man (OMIM) mainly studies the relationship between human phenotypes and genes.
11. [RepoDB](http://unmtid-shinyapps.net/shiny/repodb/): Repository of Promoting Data (RepoDB) collects drugs, diseases, and their relationships information.
12. [BioGRID](https://thebiogrid.org/): BioGRID collects protein and genetic interaction information and chemical interaction networks.
13. [HuRI](http://www.interactome-atlas.org/): The Human Reference Protein Interactome Mapping Project (HuRI) collects interaction networks of human proteins.
14. [PharmGKB](https://www.pharmgkb.org/): PharmGKB mainly focuses on information about drug-gene associations and genotype-phenotype relationships.
15. [STITCH](http://stitch.embl.de/): STITCH provides drug-target interaction data.
16. [DrugCentral](https://drugcentral.org/): DrugCentral provides information about pharmaceutical products, active ingredients chemical entities, pharmacologic action, indications, mechanism of action.
17. [PRISM](https://depmap.org/repurposing/): PRISM contains non-oncology drug response data to cancer cell lines.
18. [CCLE](https://sites.broadinstitute.org/ccle): Cancer Cell Line Encyclopedia (CCLE) performs large-scale sequencing of human cancer cell lines, integrating information about DNA mutations, gene expression, and gene copy number.
19. [GDSC](https://www.cancerrxgene.org/): Genomics of Drug Sensitivity in Cancer (GDSC) identifies drug sensitivity and molecular marker information of human cancer cell lines.
20. [DrugCombDB](http://drugcombdb.denglab.org/main): DrugCombDB contains a drug combinations data for cancer cell lines.
21. [DrugComb](https://drugcomb.org/): DrugComb is a monotherapy response and drug combinations database.
22. [GEO](https://www.ncbi.nlm.nih.gov/geo/): Gene Expression Omnibus (GEO) contains gene expression profile data for various biological samples around the world.

## S3 Multimodal Molecular Representation

pass

## S4 Network Frameworks

### S4.1 Deep Neural Networks (DNNs)

1. **[DNN](https://www.nature.com/articles/nature14539)**: LeCun Y, Bengio Y, Hinton G. **Deep learning**. nature, 2015, 521(7553): 436-444.

### S4.2 Convolutional Neural Networks (CNNs)

1. **[VGG](https://ieeexplore.ieee.org/abstract/document/7005506)**: He K, Zhang X, Ren S, et al. **Spatial pyramid pooling in deep convolutional networks for visual recognition**. IEEE transactions on pattern analysis and machine intelligence, 2015, 37(9): 1904-1916.
2. **[AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)**: Krizhevsky A, Sutskever I, Hinton G E. **Imagenet classification with deep convolutional neural networks**. Advances in neural information processing systems, 2012, 25.
3. **[Inception](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)**: Szegedy C, Liu W, Jia Y, et al. **Going deeper with convolutions**. Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.
4. **[ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)**: He K, Zhang X, Ren S, et al. **Deep residual learning for image recognition**. Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

- **CNN-based drug discovery:** [DeepDTA](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245), [MATT_DTI](https://academic.oup.com/bib/article/22/5/bbab117/6231754), [DEEPScreen](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc03414e), [AtomNet](https://arxiv.org/abs/1510.02855), [Toxic Colors](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00338)

### S4.3 Recurrent Neural Network (RNNs)

1. **[LSTM](https://ieeexplore.ieee.org/abstract/document/6795963)**: Hochreiter S. **Long Short-term Memory**. Neural Computation MIT-Press, 1997.
2. **[GRU](https://arxiv.org/abs/1412.3555)**: Chung J, Gulcehre C, Cho K H, et al. **Empirical evaluation of gated recurrent neural networks on sequence modeling**. arXiv preprint arXiv:1412.3555, 2014.
3. **[Bi-RNN](https://ieeexplore.ieee.org/abstract/document/650093)**: Schuster M, Paliwal K K. **Bidirectional recurrent neural networks**. IEEE transactions on Signal Processing, 1997, 45(11): 2673-2681.

- **RNN-based drug discovery:** [REINVENT](https://link.springer.com/article/10.1186/s13321-017-0235-x), [MOSES](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full), [ChemTS](https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1401424), [JANUS](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572)

### S4.4 Graph Neural Networks (GNNs)

#### Spectral-based Graph Neural Networks

1. **[SCNN](https://arxiv.org/abs/1312.6203)**: Bruna J, Zaremba W, Szlam A, et al. **Spectral networks and locally  connected networks on graphs**. arXiv preprint arXiv:1312.6203, 2013.
2. **[ChebNet](https://www.sciencedirect.com/science/article/pii/S1063520310000552)**: Hammond D K, Vandergheynst P, Gribonval R. **Wavelets on graphs via spectral graph theory**. Applied and Computational Harmonic Analysis, 2011, 30(2): 129-150.
3. **[GCN](https://arxiv.org/abs/1609.02907)**: Kipf T N, Welling M. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907, 2016.

#### Spatial-based Graph Neural Networks

1. **[GraphSAGE](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)**: Hamilton W, Ying Z, Leskovec J. **Inductive representation learning on large graphs**. Advances in neural information processing systems, 2017, 30. 
2. **[GAT](https://arxiv.org/abs/1710.10903)**: Veličković P, Cucurull G, Casanova A, et al. **Graph attention networks**. arXiv preprint arXiv:1710.10903, 2017. 
3. **[GIN](https://arxiv.org/abs/1810.00826)**: Xu K, Hu W, Leskovec J, et al. **How powerful are graph neural networks?**. arXiv preprint arXiv:1810.00826, 2018.

- **GNN-based drug discovery:** [D-MPNN](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237), [Attentive FP](https://pubs.acs.org/doi/abs/10.1021/acs.jmedchem.9b00959), [AMPNN](https://link.springer.com/article/10.1186/s13321-019-0407-y), [GeoGNN](http://www.peng-lab.org/lab-chinese/Geometry-enhanced%20molecular%20representation.pdf), [GCPN](https://proceedings.neurips.cc/paper_files/paper/2018/hash/d60678e8f2ba9c540798ebbde31177e8-Abstract.html), [MolDQN](https://www.nature.com/articles/s41598-019-47148-x), [DeepGraphMolGen](https://link.springer.com/article/10.1186/s13321-020-00454-3), [MNCE-RL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/5f268dfb0fbef44de0f668a022707b86-Abstract.html)

### S4.5 Transformer

1. **[Vanilla Transformer](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf)**: Vaswani A. **Attention is all you need**. Advances in Neural Information Processing Systems, 2017. 
2. **[Vision Transformer](https://arxiv.org/abs/2010.11929v2)**: Dosovitskiy A. **An image is worth 16x16 words: Transformers for image recognition at scale**. arXiv preprint arXiv:2010.11929, 2020. 
3. **[Graph Transformer](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)**: Ying C, Cai T, Luo S, et al. **Do transformers really perform badly for graph representation?**. Advances in neural information processing systems, 2021, 34: 28877-28888. 

## S5 Pre-training Task

### S5.1 Contrastive Learning

#### Perturbation Contrast

1. **[SimCLR](http://proceedings.mlr.press/v119/chen20j.html)**: Chen T, Kornblith S, Norouzi M, et al. **A simple framework for contrastive learning of visual representations**. International conference on machine learning. PmLR, 2020: 1597-1607.
2. **[GraphCL](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html)**: You Y, Chen T, Sui Y, et al. **Graph contrastive learning with  augmentations**. Advances in neural information processing systems,  2020, 33: 5812-5823.
3. **[GCC](https://dl.acm.org/doi/abs/10.1145/3394486.3403168)**: Qiu J, Chen Q, Dong Y, et al. **Gcc: Graph contrastive coding for graph neural network pre-training**. Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020: 1150-1160.

- **Perturbation contrast methods for drug discovery:** [CSGNN](https://www.researchgate.net/profile/Chengshuai-Zhao/publication/353837184_CSGNN_Contrastive_Self-Supervised_Graph_Neural_Network_for_Molecular_Interaction_Prediction/links/611a5b7f1e95fe241ad520f7/CSGNN-Contrastive-Self-Supervised-Graph-Neural-Network-for-Molecular-Interaction-Prediction.pdf), [MolCLR](https://www.nature.com/articles/s42256-022-00447-x), [MoCL](https://dl.acm.org/doi/abs/10.1145/3447548.3467186), [KCL](https://ojs.aaai.org/index.php/AAAI/article/view/20313), [CKGNN](https://arxiv.org/abs/2103.13047), [KANO](https://www.nature.com/articles/s42256-023-00654-0)

#### Cross-modal Contrast

1. **[CLIP](http://proceedings.mlr.press/v139/radford21a)**: Radford A, Kim J W, Hallacy C, et al. **Learning transferable visual  models from natural language supervision**. International conference on machine learning. PmLR, 2021: 8748-8763.
2. **[VLMO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d46662aa53e78a62afd980a29e0c37ed-Abstract-Conference.html)**: Bao H, Wang W, Dong L, et al. **Vlmo: Unified vision-language pre-training  with mixture-of-modality-experts**. Advances in Neural Information  Processing Systems, 2022, 35: 32897-32912.
3. **[BLIP](https://proceedings.mlr.press/v162/li22n.html)**: Li J, Li D, Xiong C, et al. **Blip: Bootstrapping language-image  pre-training for unified vision-language understanding and  generation**. International conference on machine learning. PMLR, 2022: 12888-12900.

- **Perturbation contrast methods for drug discovery:** [SMICLR](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c00521), [3D Infomax](https://proceedings.mlr.press/v162/stark22a.html), [GeomGCL](https://ojs.aaai.org/index.php/AAAI/article/view/20377), [MOCO](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c01468), [Text2Mol](https://aclanthology.org/2021.emnlp-main.47/), [MoleculeSTM](https://www.nature.com/articles/s42256-023-00759-6), [MoMu](https://arxiv.org/abs/2209.05481)

### S5.2 Multi-modal Matching Prediction

1. **[ViLT](https://proceedings.mlr.press/v139/kim21k.html)**: Kim W, Son B, Kim I. **Vilt: Vision-and-language transformer without convolution or region supervision**. International conference on machine learning. PMLR, 2021: 5583-5594. 
2. **[UNITER](https://link.springer.com/chapter/10.1007/978-3-030-58577-8_7)**: Chen Y C, Li L, Yu L, et al. **Uniter:  Universal image-text representation learning**. European conference on  computer vision. Cham: Springer International Publishing, 2020: 104-120.
3. **[ALBEF](https://proceedings.neurips.cc/paper_files/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html)**: Li J, Selvaraju R, Gotmare A, et al. **Align before fuse: Vision and language representation learning with momentum distillation**. Advances in neural information processing systems, 2021, 34: 9694-9705. 

- **Multi-modal matching prediction methods for drug discovery:** [ISMol](https://ieeexplore.ieee.org/abstract/document/10375706), [MolCA](https://arxiv.org/abs/2310.12798)

### S5.3 Masked Prediction

1. **[BERT](https://eva.fing.edu.uy/pluginfile.php/524749/mod_folder/content/0/BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding.pdf)** Devlin J. **Bert: Pre-training of deep bidirectional transformers for language understanding**. arXiv preprint arXiv:1810.04805, 2018. 
2. **[RoBERTa](https://arxiv.org/abs/1907.11692)**: Liu Y, Ott M, Goyal N, et al. **Roberta: A robustly optimized bert pretraining approach**. arXiv preprint arXiv:1907.11692, 2019.
3. **[T5](http://www.jmlr.org/papers/v21/20-074.html)**: Raffel C, Shazeer N, Roberts A, et al. **Exploring the limits of transfer learning with a unified text-to-text transformer**. Journal of machine learning research, 2020, 21(140): 1-67.
4. **[SpanBERT](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00300/43539/SpanBERT-Improving-Pre-training-by-Representing)**: Joshi M, Chen D, Liu Y, et al. **Spanbert: Improving pre-training by representing and predicting spans**. Transactions of the association for computational linguistics, 2020, 8: 64-77.

- **Masked prediction methods for drug discovery:** [SMILES-BERT](https://dl.acm.org/doi/abs/10.1145/3307339.3342186), [ChemBERTa](https://arxiv.org/abs/2010.09885), [MOLBERT](https://onlinelibrary.wiley.com/doi/full/10.1155/2021/7181815), [Pre-GNN](https://arxiv.org/abs/1905.12265), [GROVER](https://proceedings.neurips.cc/paper/2020/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html), [deepR2cov](https://academic.oup.com/bib/article/22/6/bbab226/6296505), [BioERP](https://academic.oup.com/bioinformatics/article/37/24/4793/6332000), [MoLFORMER](https://www.nature.com/articles/s42256-022-00580-7)

### S5.4 Autoregressive Modeling

1. **[GPT](https://hayate-lab.com/wp-content/uploads/2023/05/43372bfa750340059ad87ac8e538c53b.pdf)**: Radford A. **Improving language understanding by generative pre-training**. 2018. 
2. **[PaLM](https://www.jmlr.org/papers/v24/22-1144.html)**: Chowdhery A, Narang S, Devlin J, et al. **Palm: Scaling language modeling with pathways**. Journal of Machine Learning Research, 2023, 24(240): 1-113. 
3. **[GPT-GNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403237)**: Hu Z, Dong Y, Wang K, et al. **Gpt-gnn: Generative pre-training of graph neural networks**. Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020: 1857-1867. 

- **Autoregressive modeling methods for drug discovery:** [MolGPT](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.1c00600), [MolXPT](https://arxiv.org/abs/2305.10688), [MGSSL](https://proceedings.neurips.cc/paper/2021/hash/85267d349a5e647ff0a9edcb5ffd1e02-Abstract.html), [PrefixMol](https://arxiv.org/abs/2302.07120)

## S6 Self-supervised Training Strategy

1. **[Adapter](http://proceedings.mlr.press/v97/houlsby19a.html)**: Houlsby N, Giurgiu A, Jastrzebski S, et al. **Parameter-efficient transfer learning for NLP. International conference on machine learning**. PMLR, 2019: 2790-2799. 
2. **[Soft Prompt](https://arxiv.org/abs/2101.00190)**: Li X L, Liang P. **Prefix-tuning: Optimizing continuous prompts for generation**. arXiv preprint arXiv:2101.00190, 2021. 
3. **[Prompt Tuning](https://arxiv.org/abs/2104.08691)**: Lester B, Al-Rfou R, Constant N. **The power of scale for parameter-efficient prompt tuning**. arXiv preprint arXiv:2104.08691, 2021. 
4. **[BitFit](https://arxiv.org/abs/2106.10199)**: Zaken E B, Ravfogel S, Goldberg Y. **Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models**. arXiv preprint arXiv:2106.10199, 2021. 
5. **[LoRA](https://arxiv.org/abs/2106.09685)**: Hu E J, Shen Y, Wallis P, et al. **Lora: Low-rank adaptation of large language models**. arXiv preprint arXiv:2106.09685, 2021. 

- **Different training strategies for drug discovery:** [MRCGNN](https://ojs.aaai.org/index.php/AAAI/article/view/25665), [TIGER](https://ojs.aaai.org/index.php/AAAI/article/view/27777), [MGIB](https://ieeexplore.ieee.org/abstract/document/10584266), [MIRACLE](https://dl.acm.org/doi/abs/10.1145/3442381.3449786), [deepR2cov](https://academic.oup.com/bib/article/22/6/bbab226/6296505), [BioERP](https://academic.oup.com/bioinformatics/article/37/24/4793/6332000), [KANO](https://www.nature.com/articles/s42256-023-00654-0)

## S7 Drug Discovery Applications

### S7.1 Molecular Generation

1. **[TamGen](https://www.nature.com/articles/s41467-024-53632-4)** Wu K, Xia Y, Deng P, et al. **TamGen: drug design with target-aware molecule generation through a chemical language model**. Nature Communications, 2024, 15(1): 9360. 
2. **[Lingo3DMol](https://www.nature.com/articles/s42256-023-00775-6)** Feng W, Wang L, Lin Z, et al. **Generation of 3D molecules in pockets via a language model**. Nature Machine Intelligence, 2024, 6(1): 62-73. 
3. **[3DSMILES-GPT](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc06864e)** Wang J, Luo H, Qin R, et al. **3DSMILES-GPT: 3D molecular pocket-based generation with token-only large language model**. Chemical Science, 2025, 16(2): 637-648. 
4. **[MOLGEN](https://openreview.net/forum?id=9rPyHyjfwP)** Fang Y, Zhang N, Chen Z, et al. **Domain-Agnostic Molecular Generation with Chemical Feedback**. The Twelfth International Conference on Learning Representations. 2024. 
5. **[MolT5](https://arxiv.org/abs/2204.11817)**: Edwards C, Lai T, Ros K, et al. **Translation between molecules and natural language**. arXiv preprint arXiv:2204.11817, 2022.
6. **[Ada-T5](https://ojs.aaai.org/index.php/AAAI/article/view/30198)**: Chen Y, Xi N, Du Y, et al. **From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery**. Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(20): 21958-21966.
7. **[ChatMol](https://academic.oup.com/bioinformatics/article/40/9/btae534/7747661)**: Zeng Z, Yin B, Wang S, et al. **ChatMol: interactive molecular discovery with natural language**. Bioinformatics, 2024, 40(9): btae534.
8. **[Text+Chem T5](https://proceedings.mlr.press/v202/christofidellis23a.html)**: Christofidellis D, Giannone G, Born J, et al. **Unifying molecular and textual representations via multi-task language modelling**. International Conference on Machine Learning. PMLR, 2023: 6140-6157.
9. **[nach0](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc00966e)**: Livne M, Miftahutdinov Z, Tutubalina E, et al. **nach0: Multimodal natural and chemical languages foundation model**. Chemical Science, 2024, 15(22): 8380-8389.
10. **[3D-MolT5](https://arxiv.org/abs/2406.05797)**: Pei Q, Wu L, Gao K, et al. **3D-MolT5: Towards Unified 3D Molecule-Text Modeling with 3D Molecular Tokenization**. arXiv preprint arXiv:2406.05797, 2024.
11. **[GIT-Mol](https://www.sciencedirect.com/science/article/pii/S0010482524001574)**: Liu P, Ren Y, Tao J, et al. **Git-mol: A multi-modal large language model for molecular science with graph, image, and text**. Computers in biology and medicine, 2024, 171: 108073.

### S7.2 Molecular Property Prediction

1. **[MM-Deacon](https://arxiv.org/abs/2109.08830)**: Guo Z, Sharma P, Martinez A, et al. **Multilingual molecular representation learning via contrastive pre-training**. arXiv preprint arXiv:2109.08830, 2021. 
2. **[DVMP](https://dl.acm.org/doi/abs/10.1145/3580305.3599317)**: Zhu J, Xia Y, Wu L, et al. **Dual-view molecular pre-training**. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023: 3615-3627. 
3. **[GRAPHMVP](https://arxiv.org/abs/2110.07728)**: Liu S, Wang H, Liu W, et al. **Pre-training molecular graph representation with 3d geometry**. arXiv preprint arXiv:2110.07728, 2021. 
4. **[U2-3DPT](https://dl.acm.org/doi/abs/10.1145/3534678.3539368)**: Zhu J, Xia Y, Wu L, et al. **Unified 2d and 3d pre-training of molecular representations**. Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022: 2626-2636. 
5. **[GeomGCL](https://ojs.aaai.org/index.php/AAAI/article/view/20377)**: Li S, Zhou J, Xu T, et al. **Geomgcl: Geometric graph contrastive learning for molecular property prediction**. Proceedings of the AAAI conference on artificial intelligence. 2022, 36(4): 4541-4549. 
6. **[3D Infomax](https://proceedings.mlr.press/v162/stark22a.html)**: Stärk H, Beaini D, Corso G, et al. **3d infomax improves gnns for molecular property prediction**. International Conference on Machine Learning. PMLR, 2022: 20479-20502. 
7. **[KANO](https://www.nature.com/articles/s42256-023-00654-0)**: Fang Y, Zhang Q, Zhang N, et al. **Knowledge graph-enhanced molecular contrastive learning with functional prompt**. Nature Machine Intelligence, 2023, 5(5): 542-553. 
8. **[ISMol](https://ieeexplore.ieee.org/abstract/document/10375706)**: Zhang X, Xiang H, Yang X, et al. **Dual-View Learning Based on Images and Sequences for Molecular Property Prediction**. IEEE Journal of Biomedical and Health Informatics, 2023. 
9. **[MGIB](https://ieeexplore.ieee.org/abstract/document/10584266)**: Zang X, Zhang J, Tang B. **Self-supervised Pre-training via Multi-view Graph Information Bottleneck for Molecular Property Prediction**. IEEE Journal of Biomedical and Health Informatics, 2024. 
10. **[MOLEBLEND](https://openreview.net/forum?id=oM7Jbxdk6Z)**: Yu Q, Zhang Y, Ni Y, et al. **Multimodal Molecular Pretraining via Modality Blending**. The Twelfth International Conference on Learning Representations. 2024. 
11. **[MOCO](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c01468)**: Zhu Y, Chen D, Du Y, et al. **Molecular Contrastive Pretraining with Collaborative Featurizations**. Journal of Chemical Information and Modeling, 2024, 64(4): 1112-1122. 
12. **[KV-PLM](https://www.nature.com/articles/s41467-022-28494-3)**: Zeng Z, Yao Y, Liu Z, et al. **A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals**. Nature communications, 2022, 13(1): 862.
13. **[MolLM](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i357/7700902)**: Tang X, Tran A, Tan J, et al. **MolLM: a unified language model for integrating biomedical text with 2D and 3D molecular representations**. Bioinformatics, 2024, 40(Supplement_1): i357-i368.
14. **[MoleculeSTM](https://www.nature.com/articles/s42256-023-00759-6)**: Liu S, Nie W, Wang C, et al. **Multi-modal molecule structure–text model for text-based retrieval and editing**. Nature Machine Intelligence, 2023, 5(12): 1447-1457.
15. **[MoMu](https://arxiv.org/abs/2209.05481)**: Su B, Du D, Yang Z, et al. **A molecular multimodal foundation model associating molecule graphs with natural language**. arXiv preprint arXiv:2209.05481, 2022.

### S7.3 DDI Prediction

1. **[MIRACLE](https://dl.acm.org/doi/abs/10.1145/3442381.3449786)**: Wang Y, Min Y, Chen X, et al. **Multi-view graph contrastive representation learning for drug-drug interaction prediction**. Proceedings of the web conference 2021. 2021: 2921-2933. 
2. **[MRCGNN](https://ojs.aaai.org/index.php/AAAI/article/view/25665)**: Xiong Z, Liu S, Huang F, et al. **Multi-relational contrastive learning graph neural network for drug-drug interaction event prediction**. Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(4): 5339-5347. 
3. **[TIGER](https://ojs.aaai.org/index.php/AAAI/article/view/27777)**: Su X, Hu P, You Z H, et al. **Dual-Channel Learning Framework for Drug-Drug Interaction Prediction via Relation-Aware Heterogeneous Graph Transformer**. Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(1): 249-256. 
4. **[HS-GPF](https://link.springer.com/chapter/10.1007/978-3-031-70371-3_3)**: Ye Y, Zhou J, Li S, et al. **Hierarchical Structure-Aware Graph Prompting for Drug-Drug Interaction Prediction**. Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer Nature Switzerland, 2024: 36-54. 
5. **[H2D](https://dl.acm.org/doi/abs/10.1145/3627673.3679936)**: Zhang R, Wang X, Wang S, et al. **H2D: Hierarchical Heterogeneous Graph Learning Framework for Drug-Drug Interaction Prediction**. Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2024: 4283-4287. 
6. **[ReLMole](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c00798)**: Ji Z, Shi R, Lu J, et al. **ReLMole: Molecular representation learning based on two-level graph similarities**. Journal of Chemical Information and Modeling, 2022, 62(22): 5361-5372. 
7. **[HetDDI](https://academic.oup.com/bib/article/24/6/bbad385/7333670)**: Li Z, Tu X, Chen Y, et al. **HetDDI: a pre-trained heterogeneous graph neural network model for drug–drug interaction prediction**. Briefings in Bioinformatics, 2023, 24(6): bbad385. 

### S7.4 DTI Prediction

1. **[CSCo-DTA](https://academic.oup.com/bib/article/25/1/bbad516/7529149)**: Wang J, Xiao Y, Shang X, et al. **Predicting drug–target binding affinity with cross-scale graph contrastive learning**. Briefings in Bioinformatics, 2024, 25(1): bbad516. 
2. **[DrugLAMP](https://academic.oup.com/bioinformatics/article/40/12/btae693/7906489)**: Luo Z, Wu W, Sun Q, et al. **Accurate and transferable drug–target interaction prediction with DrugLAMP**. Bioinformatics, 2024, 40(12): btae693. 
3. **[C2P2](https://academic.oup.com/bib/article/23/4/bbac269/6628784)**: Nguyen T M, Nguyen T, Tran T. **Mitigating cold-start problems in drug-target affinity prediction with interaction knowledge transferring**. Briefings in Bioinformatics, 2022, 23(4): bbac269. 
4. **[BioT5](https://arxiv.org/abs/2310.07276)**: Pei Q, Zhang W, Zhu J, et al. **Biot5: Enriching cross-modal integration in biology with chemical knowledge and natural language associations**. arXiv preprint arXiv:2310.07276, 2023.

### S7.5 Molecule Captioning

1. **[MolCA](https://arxiv.org/abs/2310.12798)**: Liu Z, Li S, Luo Y, et al. **MolCA: Molecular graph-language modeling with cross-modal projector and uni-modal adapter**. arXiv preprint arXiv:2310.12798, 2023.
2. **[3D-MOLM](https://arxiv.org/abs/2401.13923)**: Li S, Liu Z, Luo Y, et al. **Towards 3d molecule-text interpretation in language models**. arXiv preprint arXiv:2401.13923, 2024.
3. **[MolFM](https://arxiv.org/abs/2307.09484)**: Luo Y, Yang K, Hong M, et al. **Molfm: A multimodal molecular foundation model**. arXiv preprint arXiv:2307.09484, 2023.
4. **[UniMoT](https://arxiv.org/abs/2408.00863)**: Zhang J, Bian Y, Chen Y, et al. **Unimot: Unified molecule-text language model with discrete token representation**. arXiv preprint arXiv:2408.00863, 2024.