
# Video Super-Resolution Using MMagic Framework

## Project Overview

This project focuses on video super-resolution using the MMagic framework, where three advanced models—**EDVR**, **BasicVSR**, and **IconVSR**—are compared. The goal is to enhance video resolution and evaluate the performance of these models using the REDS dataset. Key performance metrics include PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

## Models Used

### EDVR (Enhanced Deep Video Restoration)

- **Overview:** EDVR utilizes modules like PreDeblur, PCD alignment, TSA fusion, and Reconstructive module to handle blur, motion correction, detail aggregation, and frame consistency.
- **Key Features:**
  - **PreDeblur Module:** Handles input frame blur.
  - **PCD Alignment:** Ensures accurate motion correction and alignment.
  - **TSA Fusion:** Aggregates data from multiple frames.
  - **Reconstructive Module:** Refines output frames for better quality.

### BasicVSR

- **Overview:** BasicVSR employs bidirectional propagation and feature-level alignment for efficient global information processing.
- **Key Features:**
  - **Bidirectional Propagation:** Processes long-term information effectively.
  - **Flow-Based Alignment:** Achieves alignment at the feature level.
  - **Feature Concatenation & Pixel-Shuffle:** Used for aggregation and upsampling.

### IconVSR

- **Overview:** IconVSR builds on BasicVSR with innovations such as the Information-Refill Mechanism and Coupled Propagation.
- **Key Features:**
  - **Information-Refill Mechanism:** Refines features from keyframes to improve alignment and reduce errors.
  - **Coupled Propagation:** Interconnects bidirectional propagation to enhance feature quality and global information retention.

## Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** Measures how closely a restored image matches the original.
- **SSIM (Structural Similarity Index):** Evaluates the overall visual quality by comparing structural details.

## Dataset

- **REDS Dataset:** A high-quality dataset used for training and evaluation.


