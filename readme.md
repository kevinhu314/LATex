# LATex: Leveraging Attribute-based Text Knowledge for Aerial-Ground Person Re-Identification

[![arXiv](https://img.shields.io/badge/arXiv-2503.23722-b31b1b.svg)](https://arxiv.org/abs/2503.23722)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"LATex: Leveraging Attribute-based Text Knowledge for Aerial-Ground Person Re-Identification"**.

> ⚠️ **Note:** This paper is currently **under review**.

## 📜 Abstract

Aerial-Ground person Re-IDentification (AG-ReID) is a challenging task due to drastic viewpoint variations between heterogeneous cameras. Previous methods often overlook the semantic information in person attributes and rely on high-cost full fine-tuning strategies.

To address these issues, we propose **LATex**, a novel framework that adopts **prompt-tuning** strategies to leverage attribute-based text knowledge.

* **Attribute-aware Image Encoder (AIE):** Extracts global semantic features and attribute-aware features using learnable prompts to transfer CLIP's pre-trained knowledge.
* **Prompted Attribute Classifier Group (PACG):** Explicitly predicts person attributes (e.g., gender, clothing) and obtains attribute representations, utilizing interdependencies among attributes.
* **Coupled Prompt Template (CPT):** Transforms attribute representations and view information into structured sentences (e.g., *"A [view] view photo of a [attribute] person"*) to enhance discriminative features.

Our framework achieves state-of-the-art performance on AG-ReID benchmarks while significantly reducing trainable parameters compared to full fine-tuning methods.

![Framework](assets/overall.pdf)
*(Figure 2 from the paper: The illustration of the proposed LATex framework.)*

## ✨ Highlights

* **Attribute Consistency:** We leverage the insight that person attributes remain consistent across drastic aerial-ground viewpoint changes, unlike visual body part distributions.
* **Efficient Training:** By utilizing prompt-tuning on CLIP, LATex reduces training costs and parameters significantly compared to full fine-tuning strategies.
* **SOTA Performance:** Validated on **AG-ReID.v1**, **AG-ReID.v2**, and **CARGO** benchmarks.

