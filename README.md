## Ensembling Deep Learning And CIELAB Color Space Model for Fire Detection from UAV images

This repository contains the source code of our paper, Ensembling Deep Learning And CIELAB Color Space Model for Fire Detection from UAV images (publication inprogress in <a href="https://dl.acm.org/conference/aimlsystems">Conference AI-ML-Systems</a>).
Wildfires can cause significant damage to forests and endanger wildlife. Detecting these forest fires at the initial stages helps the authorities in preventing them from spreading further.  In this paper, we first propose a novel technique, termed CIELAB-color technique, which detects fire based on the color of the fire in CIELAB color space. Since deep learning (CNNs) and image processing have complementary strengths, we combine their strengths to propose an ensemble architecture. It uses two CNNs and the CIELAB-color technique and then performs majority voting to decide the final fire/no-fire prediction output.

<img src="reports/figures/Ensemble_Voting_Classifier.png">