# VSR-using-MMAGIC-framework


FACULTY OF SCIENCE, ENGINEERING AND COMPUTING

School of Computer Science & Mathematics

MSc DEGREE IN DATA SCIENCE

Name: Vedant Bhosale

ID Number: K2288954

Project Title: Video Super-Resolution using deep learning methods.

Date: 09th January 2024

Supervisor: Prof. Maria Martini 

Kingston University London

Warranty Statement
This is a student project. Therefore, neither the student nor Kingston university makes any warranty, express or implied, as to the accuracy of the data or a conclusion of the work performed in the project and will not be held responsible for any consequences arising out of any inaccuracies or omissions therein.






ABSTRACT

In the realm of video super resolution (VSR), this study delves into its crucial significance, particularly concerning bandwidth conservation and the enhancement of low-resolution videos. The utilization of techniques such as "Video Super Resolution" becomes paramount in addressing the challenges posed by lower resolutions, ultimately contributing to an improved visual experience. The motivation behind this research stems from the escalating demand for enhanced visual quality, especially with the widespread adoption of high-definition screens. Video Super Resolution stands out as a pivotal solution capable of delivering clearer and more intricate video information, all the while mitigating the constraints associated with bandwidth limitations.

The focal point of the anticipated results is centred on the identification of the model that attains the highest Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) scores, indicative of superior video reconstruction capabilities. The expectation is that IconVSR will emerge as the leading contender, excelling in both PSNR and SSIM metrics, thereby showcasing its efficacy in producing high-quality super-resolved videos. Conversely, EDVR is envisaged to exhibit comparatively lower scores, providing a robust benchmark for comprehensive comparison and evaluation. This research focuses on valuable insights into the domain of VSR, shedding light on the performance differentials among prominent models and their implications for advancing video quality in diverse applications.

Keywords: Video Upscaling, Super resolution, GAN, Video Super Resolution, Peak signal-to-noise ratio (PSNR), structural similarity index (SSIM)












Acknowledgements
I would like to express my deepest gratitude to my project supervisor, Prof. Maria Martini, for her invaluable guidance, unwavering support, and insightful feedback throughout the duration of this project. Her expertise and commitment have been instrumental in shaping the course of this research.

I extend my sincere appreciation to my former project supervisor, Nabajeet Barman, whose early contributions laid the foundation for this work. His dedication to fostering a collaborative and intellectually stimulating environment has been truly enriching.

This endeavour would not have been possible without the support of these individuals, and for that, I am truly grateful.




















Table of Contents
ABSTRACT	2
Acknowledgements	3
Chapter 1: Introduction	6
1.1 Introduction to Data Science	6
1.2 Definition and Importance of Video Super Resolution	6
1.3 Background and Motivation:	7
1.4 Stakeholders Concern	8
1.5 Aims & Objectives:	8
1.6 Summary of Contribution and Thesis Outline	9
1.7 Technologies & Resources:	9
1.8 Ethics Relevance and Progress	10
Chapter 2: Literature Review:	10
2.1 Overview	10
2.2 Interpolation Method	10
2.3 Sparse Coding and Neural Network	12
2.4 Deep Learning	13
2.5 Single Image Super-Resolution: SRCNN (Super-Resolution Convolutional Neural Network)	14
2.6 Overview of Image Super Resolution vs. Video Super Resolution	15
2.7 Generative AI	16
Types of GAN	17
2.8 Efficient Sub-Pixel Convolutional Networks:	18
2.9 Loss Function & its types:	19
2.10 Techniques: Single frame vs. Temporal Approaches	20
2.11 Challenges in Video Super Resolution	21
2.12 Summary of Literature Review	22
Chapter 3: Video Super Resolution Implementation & Evaluation	22
3.1 Overview	22
3.2 Dataset Description	23
3.3 Models and Analysis	24
3.3.1 EDVR	24
3.3.2 BasicVSR	27
3.3.3 Icon VSR	31
3.3.4 Metrics	32
3.3.5 Visual Results	34
Chapter 4:  Conclusion:	36
References	37

List of Figures
Figure 1 Super Resolution Divisions	7
Figure 2  Bilinear Interpolation  and Bicubic Interpolation	11
Figure 3 Functioning of Super resolution using deep learning methods borrowed from [1]	13
Figure 4 Proposed Architecture [11]	14
Figure 5 Comparison of the SRCNN output	14
Figure 6 Reference architecture with three convolutional layers for super-resolution of images. borrowed from [15]	15
Figure 7 Architecture diagram of the Video Enhancement using Single Image Super Resolution Model	16
Figure 8 GAN	17
Figure 9 Vanilla GAN	17
Figure 10 Conditional GAN	18
Figure 11 Temporal Modulation addition in Super Resolution	21
Figure 12 Comparing various images of REDS.	23
Figure 13 The EDVR framework.	24
Figure 14 Basic VSR Architecture and Propagations [16]	27
Figure 15 Unidirectional Propagation frame index vs PSNR	28
Figure 16 Two main components of ICONVSR [16]	31
Figure 17 Comparison of all three models on the scale of SSIM & PSNR	34
Figure 18 Output of ICONVSR	35
Figure 19: Output of BasicVSR	35
Figure 20 Output of EDVR	36












Chapter 1: Introduction

1.1 Introduction to Data Science
Data science plays a pivotal role in addressing challenges related to Visual Speech Recognition (VSR), a field that focuses on interpreting and understanding human speech through visual cues. In the context of VSR, data science encompasses the application of various techniques and methodologies to analyse and extract valuable insights from visual information associated with speech. VSR involves the utilization of video data, such as lip movements and facial expressions, to enhance speech recognition systems. Data scientists working in VSR often employ machine learning algorithms, computer vision techniques, and deep learning models to process and interpret visual speech data. These approaches enable the development of sophisticated systems capable of recognizing spoken language by analysing the visual patterns associated with speech articulation. The interdisciplinary nature of data science brings together expertise from computer science, statistics, and domain-specific knowledge to create robust VSR models that can contribute to advancements in human-computer interaction, accessibility, and communication technologies. As the importance of VSR continues to grow, data science provides the necessary tools and methodologies to harness the potential of visual information in understanding and improving speech recognition systems.
1.2 Definition and Importance of Video Super Resolution
Video super resolution is a pivotal field within computer vision and image processing, dedicated to the enhancement of video quality by increasing the spatial resolution of its frames. Spatial resolution refers to the amount of detail that can be discerned in an image, and video super resolution strives to elevate this aspect, resulting in videos with greater clarity, finer details, and an overall improved visual experience.
In essence, the process involves reconstructing high-resolution frames from their lower-resolution counterparts within a video sequence. This becomes particularly crucial in applications where a high level of visual fidelity is essential, such as surveillance systems, medical imaging, and multimedia content delivery. The importance of video super resolution is underscored by its ability to address the limitations of conventional video formats, which often compromise on spatial detail for the sake of storage and bandwidth efficiency. By mitigating the loss of detail in videos, this technique opens avenues for applications demanding precise visual information, such as facial recognition, object detection, and forensic analysis. Moreover, video super resolution contributes to advancements in virtual reality (VR) and augmented reality (AR) experiences, enriching the immersive quality of these environments. As these technologies continue to evolve, the demand for high-resolution video content becomes increasingly pronounced, positioning video super resolution as a key enabler for pushing the boundaries of visual realism in virtual environments.
 
Figure 1 Super Resolution Divisions


1.3 Background and Motivation: 

The goal of the computer vision and image processing field known as "Video Super Resolution" (VSR) is to improve the quality of video footage by raising its spatial resolution. Spatial resolution in the context of video refers to the degree of clarity and detail in each frame. Resolution issues impact traditional video content frequently, resulting in a loss of small details and overall visual quality. Several variables, such as storage limitations, bandwidth considerations, and recording device limitations, can lead to low-resolution videos. Furthermore, data loss frequently occurs during video compression and transmission, which lowers the quality of the final product. Motion blur, noise, and other abnormalities can also be present in video recordings, which reduces the visual quality of the image. As high-definition monitors, streaming services, and immersive multimedia experiences become more commonplace, so does the need for high-quality video content. Video Super Resolution techniques have become increasingly popular to produce higher-resolution frames and improve the visual attractiveness of video content in response to this demand. 
The constant desire for better visual quality in video content across a wide range of applications is the motivation behind Video Super Resolution (VSR). Videos are expected to have crisper, more detailed frames due to the widespread use of high-definition displays, streaming services, and immersive multimedia content. Lower-resolution content often detracts from the overall visual experience due to bandwidth constraints, recording equipment limitations, and video compression difficulties. The goal of Video Super-Resolution techniques is to intelligently increase the spatial resolution of video frames to overcome these restrictions. The goal of this endeavour is to enhance video quality so that it can be played on contemporary screens, such 4K and 8K televisions. Higher-resolution videos have benefits beyond aesthetics: in computer vision, surveillance, and analytics applications, they improve object recognition and analysis accuracy. Furthermore, Video Super Resolution is essential to post-production procedures since it enables content creators to improve the quality of their footage, especially when working with older or archived material. The goal of developing more realistic and immersive settings for virtual and augmented reality applications is to improve the user's sensation of presence. Furthermore, Video Super Resolution helps with more precise diagnosis and analysis in domains like medical imaging, where accuracy is critical. Fundamentally, better visual quality, more analytical power, and an all-around enhanced user experience in a variety of fields drive the advancement of Video Super Resolution technology.

1.4 Stakeholders Concern

Stakeholders play a pivotal role in the realm of Video Super-Resolution (VSR), representing a spectrum of interests and viewpoints. Key stakeholders encompass academics, businesses, content producers, consumers, governing bodies, and ethical activists, each making distinctive contributions and exerting influence on the trajectory of VSR. Academic researchers are committed to progressing VSR algorithms, aspiring to achieve breakthroughs that drive industry advancements. On the business front, companies are involved in VSR for commercial applications, aiming to leverage its potential for profitability and innovation.
Consumers, as the ultimate recipients of enhanced video content, express a desire for more immersive and visually compelling viewing experiences. Content producers, conversely, are committed to attaining higher visual quality, pushing the boundaries of what VSR can deliver in terms of content enhancement. Simultaneously, governing bodies play a pivotal role in navigating the ethical and legal dimensions of VSR, addressing concerns such as responsible use and compliance with regulations. Ethical activists, as stakeholders, underscore the significance of responsible technology use, privacy considerations, and transparency in the development of VSR.

As technology evolves, finding a balance among the varied concerns of stakeholders becomes crucial. This necessitates considerations of financial viability for businesses, the calibre of enhanced content for user satisfaction, adherence to regulatory frameworks, and ethical considerations to ensure responsible development. In navigating the intricate landscape of VSR, stakeholders must collaborate and discover common ground to address the diverse challenges and opportunities presented by this technology. Striking a balance among the interests and concerns of academics, businesses, content creators, consumers, governing bodies, and ethical activists is indispensable for fostering responsible and sustainable development in the realm of Video Super-Resolution.

1.5 Aims & Objectives:

The primary aim of this project is to comprehensively investigate and evaluate Video Super Resolution (VSR) techniques within the domain of low-resolution videos. The focus is on understanding the significance of VSR in enhancing visual quality while concurrently conserving bandwidth resources. To achieve this aim, the following objectives need to be completed:
•	Conduct an extensive literature review to gain a deep understanding of Video Super Resolution techniques, exploring various models, methodologies, and their applications.
•	Evaluate the performance of three prominent VSR models—BasicVSR, EDVR, and IconVSR—through quantitative measures, specifically the computation of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) scores.
•	Evaluate the performance of three prominent VSR models—BasicVSR, EDVR, and IconVSR—through quantitative measures, specifically the computation of Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) scores.
•	Determine the superior VSR model based on the evaluation results among the three model.
•	Provide practical insights and implications of the findings, offering recommendations for applications requiring enhanced video quality and bandwidth-efficient solutions.
1.6 Summary of Contribution and Thesis Outline
The following are the most important contributions of the thesis:
•	This project provides a complete extensive literature review from the interpolation method to deep learning models for upscaling the video.
•	In this study we understand different GANS in the section of literature review
•	This study evaluates the different models of the VSR such as EDVR, ICON VSR & BASICVSR. It focuses on every layer of the model and explains complete cycle.
•	In this study after evaluating the models they are compared based on the quantitative metric like Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM). 
Below is the details thesis outline of the research:
Chapter 1- Introduction:
This chapter provides brief overview of the Video Super Resolution context, problem statement, Background and motivation, Explanation of the VSR domain and its relevance.
Chapter-2 Literature Review:
In-depth exploration of evolution of VSR techniques and models. Overview of methodologies and applications in the field. Understanding the impact of VSR on bandwidth efficiency.
Chapter-3 Implementation and Evaluation
In this chapter explained about the dataset used for all three models. In this chapter detailed explanation of each model. Quantitative assessment of each model's performance using PSNR and SSIM. Comparative analysis to identify the most effective VSR technique. Interpretation of results and implications.
Chapter-4 Conclusion
Summary of key findings and their significance. Concluding remarks on the effectiveness of VSR models. Suggestions for future research directions.

1.7 Technologies & Resources:
In the execution of this research, we harnessed cutting-edge technologies and resources to facilitate a robust experimental framework. The computational power of Colab Pro, a premium version of Google Colab, played a pivotal role in expediting our evaluation processes. The scalability and efficiency of Colab Pro provided a seamless environment for handling complex computations associated with our deep learning models. Additionally, to ensure a comprehensive evaluation of our models, we leveraged pretrained models from mmagic framework. This framework, renowned for its prowess in MMagic (Multimodal Advanced, Generative, and Intelligent Creation) is an open-source AIGC toolbox for professional AI researchers and machine learning engineers to explore image and video processing, editing and generation], enabled us to conduct rigorous testing and benchmarking, ensuring the reliability and effectiveness of our proposed approach. The synergy between Colab Pro and the mmagic framework significantly contributed to the success of our research endeavours.

1.8 Ethics Relevance and Progress
During our research endeavours, it is noteworthy that no explicit approval or authorization was sought from any authorized individuals, government bodies, or public authorities. Importantly, our research has been meticulously designed to have no impact on individuals' emotions, societal images, values, or any other sensitive aspects. This ensures that our work is conducted with due respect to ethical considerations and societal well-being.
The dataset employed for testing our models is derived from open-access repositories and is widely recognized for its use in diverse research and study objectives. It is imperative to highlight that this dataset, despite containing personal information, is publicly available and has been ethically sourced, aligning with the principles of transparency and responsible data usage.
To address concerns related to compliance with GDPR (General Data Protection Regulation), stringent measures have been implemented. The researchers have committed to securely storing the dataset in a dedicated folder on box.com, ensuring that access is restricted, and data integrity is maintained. Importantly, no personal information from the dataset will be retained by our team. Following a predefined timeframe of six months, all information, including video data and associated subsets, will be automatically purged, mitigating any potential privacy concerns and upholding ethical standards in data handling and storage practices.

Chapter 2: Literature Review:
2.1 Overview
In the progression from the initial proposal, the literature review has evolved to encompass a broader spectrum of VSR methodologies. While the proposal primarily focused on traditional interpolation techniques, the subsequent review revealed the transformative impact of deep learning-based approaches. The inclusion of recent studies, such as [13], further enhances the understanding of adaptive learning models in handling complex motion patterns. Additionally, the critical analysis has shifted towards a nuanced discussion of trade-offs between spatial accuracy and computational efficiency, offering deeper insights into the challenges that persist in VSR methodologies.
In this literature review, we will focus on the evolution of Video Super resolution from the traditional approach like [6]  bicubic or Lanczos interpolation and will move towards the deep learning models such as different types of GAN [4] and at the end will discuss about the VSR models. 
2.2 Interpolation Method
In 2007, paper was published in which comparison of different types of interpolation techniques, including sine-based (linear and cubic) and spline-based methods, and evaluates their accuracy and frequency responses. The comparison also considers the signal-to-noise ratio and processing time required for each interpolation method, with linear interpolation being the fastest but less accurate compared to cubic and spline methods. 
Interpolation is defined as a convolution operation, where the interpolation function is centred at a point to determine the value of the discrete function by calculating the weighted average of neighbouring pixels. The choice of interpolation technique depends on two factors: accuracy and time required for interpolation. Linear methods are suitable for real-time applications where speed is crucial, while high-level spline interpolation is necessary for medical image processing that requires accuracy. [6] The paper aims to compare different interpolation methods based on two parameters: signal-to-noise ratio between the original and reconstructed image and the time needed for interpolation. The paper presents two convolution-based interpolation techniques: linear and cubic.
The interpolation methods are tested on doubling the size of the image in both directions, and the quality of the techniques is assessed based on the signal-to-noise ratio between the original and reconstructed image. Techniques such as Bilinear and Bicubic interpolation were the initial go-to methods for upscaling video content. Quick and easy method for scaling photos is bilinear interpolation [5]. To retrieve the pixel value at the interpolated point, it considers the four closest pixels surrounding a target point and computes a weighted average. It may provide upscaled images that are smoother but less detailed, despite being computationally efficient. Whereas when it comes to Bicubic interpolation, it is an extension of bilinear interpolation and considers larger neighbourhood of 16 pixels. It uses cubic polynomials to compute the weighted average, resulting in a smoother interpolation compared to bilinear. Bicubic interpolation tends to preserve more details than bilinear, making it suitable for certain image upscaling tasks. 
 
Figure 2  Bilinear Interpolation and Bicubic Interpolation
Following the creation and application of bicubic and bilinear interpolation techniques in image processing, Lanczos interpolation attracted interest because of its superior quality output, particularly when it came to maintaining details and reducing artefacts during image resizing. For simple image scaling tasks, bicubic interpolation—which considers a bigger neighbourhood of 16 pixels—and bilinear interpolation—which computes a weighted average of the four nearest neighbouring pixels—were frequently utilised.
A sophisticated method used in image processing, particularly for image resizing, is called Lanczos interpolation. Fundamental to the interpolation kernel is the sinc function, which can be expressed as sin(πx) / (πx). Zero crossings at integer values are a defining characteristic of this function, which is essential for maintaining image features while resizing. Lanczos interpolation limits the influence of the sinc function by using the Lanczos window to regulate its range. The Lanczos interpolation kernel is the resultant windowed sinc function. To create the interpolated image, image resizing entails convolving this kernel with the original image and blending pixel values. Compared to more straightforward techniques like bilinear or bicubic interpolation, Lanczos interpolation produces smoother results by carefully balancing the preservation of detail and the avoidance of artefacts. However, this improved quality necessitates a trade-off between efficiency and accuracy because to the increased computing complexity. A key factor is kernel size; larger kernels need more processing power but produce results that are more precise. Lanczos interpolation is well known for its ability to generate high-quality scaled images despite its computing expense; this makes it especially useful in applications like medical imaging or photography where maintaining sharp edges and minute details is crucial. [12]
2.3 Sparse Coding and Neural Network
The evolution of Video Super-Resolution (VSR) has witnessed a substantial shift from traditional signal processing techniques to the advent of learning-based approaches, showcasing a progressive journey marked by significant milestones. In the realm of learning-based strategies, Sparse Coding emerged as an early contender, reflecting attempts to learn dictionaries and sparse representations for image patches. This methodology aimed to capture the intrinsic structures within video frames through the identification of sparse and redundant features, providing a more nuanced understanding of complex visual information.

Group Sparse Representation
In this paper [9], the authors propose a novel approach to image denoising, introducing a Group Sparse Representation-based method. The core of their methodology involves tackling sparse coding problems within similar patch groups, iteratively refining this process on progressively higher-quality images. Notably, the method seamlessly integrates with an enhanced patch search scheme, emphasizing the exploration of increasingly superior image quality during the search process. The effectiveness of the proposed method is thoroughly evaluated, benchmarked against two state-of-the-art algorithms, namely Ksvd [8] and BM-3d [7]. The comprehensive evaluation includes both qualitative and quantitative assessments, revealing the superior performance of the proposed method. Furthermore, the study introduces an improved similar patch search scheme, contrasting it with a naive search scheme. The results demonstrate that the enhanced search scheme contributes to achieving superior denoising quality, evident both visually and statistically. This innovative denoising approach, with its emphasis on group sparse representation and a refined patch search strategy, emerges as a promising contender in the realm of image upscaling, showcasing its efficacy through rigorous comparative evaluations.
 Neural Network: 
As sparse coding struggles with its adaptability to intricate video dynamics, neural networks empower VSR models to learn and adapt from the data itself, unlocking the potential for more accurate and context-aware super-resolution. This seamless transition from traditional techniques to the advanced realm of neural network-based approaches reflects the dynamic landscape of VSR research and sets the stage for more effective and adaptive solutions in handling complex video content.
The advent of learning-based techniques set the stage for the subsequent rise of Neural Networks, which heralded a paradigm shift in VSR research. Neural Networks ushered in a new era by discarding traditional handcrafted features in favor of learned representations. Leveraging the inherent ability of neural networks to autonomously discern complex patterns and relationships within video data, this approach marked a significant departure from the previous reliance on pre-defined rules. Instead, it empowered models to adapt and learn from data, unlocking the potential for more accurate and context-aware super-resolution. In the domain of neural network-based image processing, a pivotal initial step is preprocessing, [10] where raw input data undergoes various operations to enhance its quality and facilitate subsequent analysis. Following this, data reduction and feature extraction techniques play a crucial role in condensing the information while retaining essential characteristics, a pivotal step in handling large datasets efficiently. The subsequent phase, image segmentation, involves partitioning the processed image into meaningful segments, laying the foundation for more targeted analysis. As we delve into object recognition, the neural network endeavours to identify and classify specific entities within the segmented regions, contributing to the broader goal of image understanding. This understanding is enriched through optimization processes that leverage pixel data, refining the neural network's performance and ensuring it adapts effectively to diverse image complexities. The synergy of these interconnected steps, from preprocessing to optimization based on pixel data, establishes a comprehensive framework for neural network-driven image processing, addressing challenges and advancing the capabilities of image analysis systems.

2.4 Deep Learning
Deep learning techniques have revolutionized the field of VSR by enabling the development of highly complex and effective models. Deep neural networks, particularly Convolutional Neural Networks (CNNs), have shown remarkable success in capturing spatial dependencies and learning high-level representations from video frames. Deep learning models are capable of automatically extracting relevant features directly from the data, removing the need for handcrafted features. Techniques such as transfer learning and pretraining on large-scale datasets, such as ImageNet, have been employed to leverage the power of deep learning in VSR. By utilizing deep learning algorithms, researchers can achieve state-of-the-art performance in terms of visual quality and detail preservation. The overall deep learning pipeline for VSR tasks. Keep in mind that while the up-sampling and feature extraction & fusion modules typically use deep CNNs, the inter-frame alignment module may use conventional techniques or deep CNNs.[1]
 
Figure 3 Functioning of Super resolution using deep learning methods borrowed from [1]

2.5 Single Image Super-Resolution: SRCNN (Super-Resolution Convolutional Neural Network)
The emergence of Single Image Super-Resolution (SISR) marked a pivotal moment with the development of the Super-Resolution Convolutional Neural Network (SRCNN). SRCNN demonstrated the efficacy of convolutional neural networks (CNNs) in learning hierarchical features for super-resolution. The Author in the paper [11] explains that there are multiple crucial steps in the Perceptual Image Super Resolution architecture that is being suggested. First, bicubic interpolation is used to upscale the image to the required size. Following the interpolation process, the resulting image, designated as 'Y,' is refined to yield an enhanced image 'Y^' that is comparable to the ground truth high-resolution image 'X.' Three layers make up the architecture: layers #1, #2, and #3 of the convolution process. The third layer creates an output image with a greater resolution by operating between the bicubic interpolated LR image.
 
Figure 4 Proposed Architecture [11]


The primary goal is to, given a low-resolution image that has been downscaled from the corresponding high-resolution image (IHR), estimate the high-resolution image (ISR). The architecture concentrates on enlarging or compressing the image in the second layer, which causes a rise in PSNR and a loss of pixels because of incorrect space and edge definition. The PSNR is assessed using the mean squared error (MSE), and it is important to gradually reduce PSNR values in order to precisely delineate edges without adding blur. The studies make use of a trained dataset that contains images that have been processed using different methods. 
 
Figure 5 Comparison of the SRCNN output

The suggested architecture reduces spatial size and computational cost by processing the pictures using convolution layers. Two types of pooling techniques, Max Pooling and Average Pooling, are employed. Max Pooling returns the maximum value from the portion covered by the kernel, while Average Pooling returns the average of the values of the portion covered by the kernel. The goal of the architecture is to provide an output image that resembles a high-resolution image yet has low PSNR values. The procedure, which combines pooling, padding, and convolution processes and shows how well the suggested method works to provide perceptual image super-resolution.
 
Figure 6 Reference architecture with three convolutional layers for super-resolution of images. borrowed from [15]

2.6 Overview of Image Super Resolution vs. Video Super Resolution
2.6.1 Overview of Image Super Resolution vs. Video Super Resolution
Image Super Resolution (ISR): Image Super Resolution (ISR) is a well-established domain in computer vision that concentrates on enhancing the spatial resolution of a single image. The primary goal is to generate a high-resolution version of a given low-resolution image. Single image super-resolution (SISR) reconstruction is the development of high-resolution (HR) images with richer details and clearer texture from low-resolution (LR) images or degraded images and has wide applications in the field of image processing [2]. This process is crucial in various applications such as medical imaging, surveillance, and content creation where a clearer and more detailed image is desirable.
ISR techniques often employ methods like bicubic interpolation, single-image convolutional neural networks (CNNs), and generative adversarial networks (GANs). While these techniques have shown considerable success in improving the visual quality of static images, they inherently lack consideration for the temporal aspect found in video sequences.
Video Super Resolution (VSR):
Video Super Resolution (VSR) extends the principles of ISR to the domain of videos, acknowledging the temporal dependencies between consecutive frames. Unlike ISR, which operates on individual images in isolation, VSR considers the coherence and continuity of information across frames within a video sequence. In VSR, the goal is not only to enhance the spatial resolution of each frame but also to ensure smooth transitions and consistency throughout the entire video. This is particularly important in scenarios such as video streaming, surveillance, and video content production, where maintaining visual quality over time is essential.

 
Figure 7 Architecture diagram of the Video Enhancement using Single Image Super Resolution Model
Distinguishing Factors:
Temporal Context: One of the primary distinctions lies in the temporal context. ISR algorithms treat each image independently, without considering the relationship between consecutive frames. VSR, however, leverages the temporal context by incorporating information from neighbouring frames to enhance the resolution collectively.
Dynamic Changes: Videos inherently involve dynamic changes over time, including object motion, scene variations, and evolving contexts. VSR methods need to address these dynamic aspects, ensuring that the enhanced frames not only reflect higher spatial resolution but also align seamlessly with the evolving content in the video.
Applications: While ISR finds applications in scenarios where a single high-resolution image is essential, VSR is particularly relevant in applications where videos play a crucial role. This includes video surveillance, video conferencing, and any context where the temporal evolution of visual information is significant. Understanding these differences is pivotal for developing effective super resolution algorithms tailored to the specific characteristics and demands of video content. The subsequent sections will delve into the techniques and challenges associated with Video Super Resolution, providing a comprehensive exploration of this dynamic and evolving field.
	
2.7 Generative AI 
Generative AI, specifically Generative Adversarial Networks (GANs), has emerged as a powerful tool in the field of VSR. GANs can generate high-resolution video frames by introducing adversarial training between a generator and a discriminator network. Through this adversarial training process, the generator is encouraged to create realistic and eye-catching frames that closely reflect the high-resolution ground truth. The two models that make up a GAN are the Generator and the Discriminator. The first is the Generative Model (G), which seeks to produce entirely original input samples from the original input. Although the generator has never seen the real input, it shares a comparable input structure with the real dataset. The Generator creates fresh samples to facilitate the distribution of actual information samples in addition to accepting random noise as input. Discriminator model (D), the second model, is primarily used to distinguish between different types of data and calculate the likelihood that a given piece of data originated from the Generator or the original dataset. The general Generative Adversarial Networks are depicted in Figure 1, where the true picture is denoted by G(Z), the fake image by X, the generator by G, the discriminator by D, and the noise vector by Z.[4]
 
Figure 8 GAN
Types of GAN 
Vanilla GAN:
The Vanilla GAN is based on a straightforward yet effective adversarial approach. It consists of two neural networks, a generator, and a discriminator, that compete with one another during training. The discriminator's job is to discern between actual and created samples, whereas the generator's goal is to create synthetic data, such images, from random noise. Iterative training is a back-and-forth interaction in which the discriminator adjusts to better distinguish between phoney and real samples, while the generator modifies its output to trick the discriminator. The success of the model's training is indicated by the equilibrium attained when the generator produces samples that are identical to genuine ones. Even though it is straightforward, the Vanilla GAN has paved the way for more intricate GAN variations, greatly advancing the field of generative models and applications, which spans from image synthesis to style transfer.
 
Figure 9 Vanilla GAN
Conditional GAN: An extension of the Vanilla GAN that presents the idea of conditioning on extra information during the generation phase is the Conditional generation Adversarial Network (CGAN). The generator in a typical GAN generates data samples from random noise, and the discriminator's job is to tell the difference between created and real samples. To direct the generation process, however, a CGAN provides extra information, usually as supplementary input, to both the discriminator and the generator. The extra information that is supplied is usually in the form of class labels. To create realistic images of various animals, for instance, the CGAN would require a random noise vector and a class label indicating the kind of animal to be created (such as "dog" or "cat"). The generator can create outputs that are more precise and targeted because to this conditioning. In a CGAN, the generator's goal is still to produce realistic samples that match the conditioning information provided and fool the discriminator into thinking they are real. Conversely, the discriminator verifies that the generated samples meet the specified criteria in addition to assessing the generated samples' realism. A CGAN's training procedure aims to strike a compromise between the generator's learning to generate realistic and varied samples for each class and the discriminator's learning to  
discriminate between actual and generated samples under the given conditions. Applications for CGANs can be found in several fields, including style transfer, image-to-image translation, and producing high-resolution images that are conditioned on features. They are an invaluable tool in many machine learning tasks because of their conditional character, which improves the control and customisation of the generative process. 
2.8 Efficient Sub-Pixel Convolutional Networks:
Efficient Sub-Pixel Convolutional Networks (ESPCN) is a deep learning architecture designed for single-image super-resolution tasks. Super-resolution involves increasing the spatial resolution of an image, generating a high-resolution (HR) image from a low-resolution (LR) input. ESPCN specifically aims to efficiently upscale the resolution of images while maintaining good performance. Here are the key steps of the ESPCN model:

Super-Resolution Phase:	
Input: The process starts with a low-resolution image (LR), which is the degraded version of the high-resolution image that we want to reconstruct.
Convolutional Feature Extraction: ESPCN employs a convolutional neural network (CNN) for feature extraction from the LR image. This helps capture relevant information and features.
Sub-Pixel Convolution (Pixel Shuffle): The distinctive feature of ESPCN is the use of sub-pixel convolution. This layer is responsible for the upscaling of the LR feature maps to the desired high resolution. Sub-pixel convolution is also known as pixel shuffle, and it involves rearranging the elements of the feature maps to effectively increase the spatial resolution.
Convolution Neural Network (CNN) Phase:
Feature Extraction (FE): The CNN further extracts feature from the enhanced feature maps obtained after the sub-pixel convolution. This step helps to capture more complex and abstract features.
Convolutional Layers (CL): The model utilizes convolutional layers for additional processing of features. These layers contribute to learning hierarchical representations of the input data.
Classification (C): The final layer of the CNN is responsible for classification. It maps the learned features to the output classes or generates the final HR image.
Convolutional Super-Resolution (CSR):
Convolutional Super-Resolution: This step involves applying convolutional operations to refine and improve the quality of the upscaled image. It contributes to reducing artifacts and enhancing the visual quality of the final output.
Training:
Training Data: ESPCN is trained on a dataset containing pairs of LR and HR images. During training, the model learns to map LR images to their corresponding HR counterparts.
Loss Function: The training process involves minimizing a loss function, such as mean squared error (MSE), which measures the difference between the predicted HR images and the ground truth HR images.
In summary, ESPCN efficiently combines sub-pixel convolutional layers with traditional convolutional neural network components to achieve real-time single-image super-resolution. The sub-pixel convolutional layer is particularly effective in upscaling low-resolution feature maps to high-resolution images, and the overall architecture is trained to generate visually appealing and detailed reconstructions.
2.9 Loss Function & its types:
A loss function is a crucial component used during the training of machine learning models. The primary goal of a loss function is to quantify the difference between the predicted high-resolution frames generated by the model and the actual high-resolution frames, which serve as ground truth. The model aims to minimize this loss during the training process, essentially learning to produce high-quality super-resolved videos. The choice of an appropriate loss function is vital because it guides the optimization process and determines how well the model aligns its predictions with the desired outcomes. Different aspects of video quality, such as pixel-level accuracy, perceptual similarity, and temporal coherence, need to be considered in the design of a loss function for VSR. Here are some common types of loss functions used in VSR:

Mean Squared Error (MSE): MSE is a straightforward and commonly used loss function that measures the average squared difference between the predicted and ground truth pixel values. While MSE is easy to compute, it tends to prioritize pixel-level accuracy and may not fully capture perceptual differences.
Structural Similarity Index (SSIM): SSIM is a metric that considers luminance, contrast, and structure, providing a more perceptually relevant measure compared to MSE. Incorporating SSIM into the loss function can encourage the model to generate results that are not only pixel-accurate but also visually like the ground truth.
Perceptual Loss: Inspired by human perception, perceptual loss leverages pre-trained neural networks, such as VGG or ResNet, to extract high-level features from both predicted and ground truth frames. The loss is then computed based on the difference in these feature representations. Perceptual loss is valuable for encouraging the model to focus on capturing high-level content and structural details.
Temporal Consistency Loss: In VSR, maintaining temporal consistency across consecutive frames is crucial for producing smooth and realistic video sequences. Temporal consistency loss penalizes the model when it fails to preserve motion and coherence between frames, promoting better performance in dynamic scenes.
Adversarial Loss (GAN-based Loss): Generative Adversarial Networks (GANs) can be employed in VSR, introducing an adversarial loss that involves a discriminator distinguishing between real and generated frames. This adversarial training helps the model generate more visually convincing and natural-looking high-resolution frames.
The combination of these loss functions or a carefully designed hybrid loss can provide a comprehensive objective for training VSR models. Researchers often experiment with different loss formulations to strike a balance between pixel-level fidelity and perceptual quality, tailoring the loss function to the specific requirements of the VSR task at hand.
2.10 Techniques: Single frame vs. Temporal Approaches

Single-frame Approaches:
 In video super resolution, single-frame methods concentrate on improving the resolution of individual frames separately, disregarding the temporal correlations between successive frames. These approaches use sophisticated image processing techniques to increase spatial resolution by treating each frame as a separate entity. Super-resolution convolutional neural networks (SRCNNs), generative adversarial networks (GANs), and bicubic interpolation are common upscaling algorithms used in single-frame approaches. The goal of these algorithms is to enhance the visual quality of individual frames by predicting high-resolution information from low-resolution inputs.
Temporal Approaches:
Conversely, temporal techniques leverage the temporal information conveyed by successive frames. By considering motion and dynamic changes, these techniques make use of the temporal dependencies to improve video sequences cohesively. Recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or three-dimensional convolutional neural networks (3D CNNs) are frequently used in temporal techniques. By capturing temporal patterns and dependencies, these systems enable super resolution that looks more consistent and natural across frames. Temporal approaches are better equipped to handle motion-related challenges such as blur and deformation.
The proposed Temporal Adaptive Neural Network introduces an innovative approach to enhance video super-resolution (SR) by intelligently leveraging temporal information. This neural network is designed to dynamically select the optimal temporal scale for video SR, addressing the challenge of effectively utilizing temporal information in the process. The architecture comprises multiple SR inference branches operating on different temporal scales (Bi) and a crucial Temporal Modulation Branch (T) responsible for determining the ideal temporal scale based on motion information.
 
Figure 11 Temporal Modulation addition in Super Resolution
Each SR inference branch utilizes filters with varying temporal lengths, allowing them to work on distinct temporal scales. The Temporal Modulation Branch operates on the largest temporal scale, employing convolutional layers to predict pixel-level weight maps. These weight maps indicate the contribution of each temporal scale, enabling adaptive aggregation of the HR estimates. The primary objective is to jointly learn the components of SR inference branches and the Temporal Modulation Branch in a unified manner, facilitating the adaptive determination of the optimal temporal scale for each frame based on motion information. The integration of these estimates from different temporal scales produces the final HR frame.

The paper [14] extensively covers the detailed architecture and training objectives of the Temporal Adaptive Neural Network, providing a comprehensive understanding of its design and functionality. It also conducts thorough evaluations of various network architectures to assess their impact on SR performance. Through these evaluations, the paper demonstrates the network's effectiveness in handling complex motion and emphasizes the advantages of its adaptive aggregation approach over traditional methods. The Temporal Adaptive Neural Network presents a promising advancement in the field of video super-resolution, showcasing its adaptability to diverse motion scenarios and its potential to deliver superior results.
2.11 Challenges in Video Super Resolution	
Despite the significant advancements in video super-resolution (VSR), the field is not without its set of challenges. One persistent hurdle lies in effectively addressing complex motion patterns within video sequences. The dynamic nature of objects, scene variations, and evolving contexts introduces intricacies that traditional methods and even some early learning-based approaches struggle to navigate. Achieving temporal consistency across consecutive frames poses another formidable challenge, particularly in scenarios involving rapid motion or dynamic scene changes. The balance between spatial accuracy and computational efficiency remains a delicate trade-off, with the demand for real-time processing amplifying the complexity. Additionally, there is a continuous need for more comprehensive benchmark datasets that capture diverse video content, ensuring that VSR models generalize well across different scenarios. As the field advances, tackling these challenges becomes paramount to unlocking the full potential of video super-resolution in practical applications such as video streaming, surveillance, and content creation.
2.12 Summary of Literature Review
In the literature review, the evolution of Video Super Resolution (VSR) is explored, transitioning from traditional interpolation methods to deep learning-based approaches. The review delves into interpolation techniques such as bicubic and Lanczos, highlighting their advantages and limitations. Sparse coding and group sparse representation are introduced as early learning-based methods, paving the way for neural network-driven approaches. The shift from sparse coding to neural networks is discussed, emphasizing the adaptability of the latter in handling complex video dynamics. Deep learning, particularly Convolutional Neural Networks (CNNs), emerges as a revolutionary tool in VSR, enabling the capture of spatial dependencies and high-level representations. The review encompasses Single Image Super-Resolution (SISR), featuring the influential Super-Resolution Convolutional Neural Network (SRCNN). It also distinguishes Image Super Resolution (ISR) from Video Super Resolution (VSR), emphasizing the importance of temporal context in the latter. Generative Adversarial Networks (GANs) and Efficient Sub-Pixel Convolutional Networks (ESPCN) are introduced as powerful tools in VSR. The discussion extends to loss functions, encompassing MSE, SSIM, perceptual loss, temporal consistency loss, and adversarial loss, crucial in training VSR models. The distinction between single-frame and temporal approaches is explored, with a focus on Temporal Adaptive Neural Network as an innovative solution. Despite progress, challenges in addressing complex motion patterns, achieving temporal consistency, and balancing accuracy with efficiency persist in the VSR landscape. The comprehensive review sets the stage for the subsequent exploration of techniques and challenges in Video Super Resolution.

Chapter 3: Video Super Resolution Implementation & Evaluation
3.1 Overview 
The project utilizes the REDS dataset, consisting of 720 x 1280 resolution video frames with natural and artificial content to train and assess example-based video deblurring and super-resolution techniques. The REDS dataset is designed to enhance realism in deterioration, particularly motion blur, and includes matching degraded frames along with excellent ground truth reference frames. The research evaluates three different models, namely BasicVSR, EDVR, and ICONVSR, using the REDS dataset. This chapter provides an in-depth analysis of the EDVR model, detailing its architecture, modules, and components such as the PreDeblur Module, PCD Align Module, TSA Fusion Module, and Reconstructive Module. Each module's functionality and impact on video super-resolution are thoroughly explained. This chapter also delves into the BasicVSR model, highlighting its bidirectional recurrent network, propagation strategies, alignment methodologies, aggregation, and upsampling procedures. The analysis compares local, unidirectional, and bidirectional propagation methods, emphasizing the importance of global receptive fields. The BasicVSR model's effectiveness in addressing spatial misalignments through feature alignment is detailed. At last, but not least the ICONVSR model, an evolution of BasicVSR, introduces innovative components like the Information-Refill Mechanism and Coupled Propagation. The Information-Refill Mechanism addresses alignment challenges in occluded regions, while Coupled Propagation interconnects propagation modules, enhancing information aggregation.
	The study concludes with an exploration of evaluation metrics, including Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM). PSNR measures distortion, while SSIM considers structural and perceptual aspects. Model performance scores are provided for EDVR, BasicVSR, and ICONVSR based on these metrics. The combined use of PSNR and SSIM is emphasized for a more comprehensive evaluation that considers both technical precision and perceptual relevance.

3.2 Dataset Description
The dataset used is called Realistic and Dynamic Scenes, or REDS. High-quality, dynamic scenes with 720 x 1280 resolution video frames are used to train and assess example-based video deblurring and super-resolution techniques. It features thirty thousand frames with different scenes, natural, and artificial content.
The goal of REDS is to enhance the current video deblurring and SR datasets by adding more diversity to the content and enhancing the realism of deterioration, particularly motion blur. It features 300 RGB video clips that it has captured, taking into consideration the dynamics of different motion, diversity of source contents (scenes and locales), and quality of each frame. And also it has recorded videos at 120 frames per second with a resolution of 1080 × 1920 using the GoPro HERO6 Black camera. The REDS dataset is used for benchmarking super-resolution and video deblurring. REDS dataset offers both matching degraded frames and excellent ground truth reference frames. Every degraded frame simulates common video degradations like compression, down sampling, and motion blur.
By combining succeeding frames, the 120 fps videos used to create the REDS dataset are enhanced with fuzzy images. The following frames are offered for training and validation data; they are used to create fuzzy images. Several zip files include the dataset because of the enormous file sizes. A total of fifteen 500-length sequences, or the same amount of time as the regular 24 frames per second version, are contained in every zip file. Roughly 10 GB of each file. With each clip having 100 consecutive frames, REDS is composed of 240 training clips, 30 validation clips, and 30 testing clips. 
 
3.3 Models and Analysis
We trained and evaluated 3 different models using REDS dataset to evaluate the performance of all models on the same dataset. The models we used to perform the are BasicVSR, BasicVSR++, EDVR and ICONVSR.
3.3.1 EDVR 
EDVR, short for "Enhanced Deep Video Restoration," is a cutting-edge video super-resolution (VSR) model designed to enhance the quality of frames in low-resolution video sequences. This model, introduced in the publication "Video Super-Resolution with Enhanced Deformable Convolutional Networks" (CVPR 2019), is particularly notable for its effectiveness in addressing diverse degradations commonly found in real-world videos. EDVR extends traditional deformable convolutions with Enhanced Deformable Convolutional Networks (EDCN), allowing the network to intelligently adapt its focus based on the input content. The architecture employs a two-stage process involving feature alignment and high-resolution frame synthesis, contributing to superior performance. With its ability to smartly filter and enhance video details, along with adaptive feature calibration and training on diverse real-world datasets, EDVR excels in overcoming challenges such as blurriness and noise. Its computational efficiency ensures swift results, and a stochastic degradation scheme enables flexible adjustments to varying levels of video degradation. Overall, EDVR stands out as a sophisticated solution for enhancing video quality in real-world scenarios.
 
Figure 13 The EDVR framework.
The EDVR framework. It is a unified framework suitable for various video restoration tasks, e.g., super-resolution and deblurring. Inputs with high spatial resolution are first down sampled to reduce computational cost. Given blurry inputs, a PreDeblur Module is inserted before the PCD Align Module to improve alignment accuracy. We use three input frames as an illustrative example. [17]
3.3.1.1 PreDeblur Module
EDVR (Enhanced Deep Video Restoration) model's PreDeblur module is essential for improving the clarity of low-resolution video frames by reducing motion blur. Its main goal is to solve the problems caused by fast object motions during video capture, which frequently cause frames to lose sharpness and information. The module uses sophisticated deblurring methods to estimate the motion or blur kernels that are connected to every frame. It recovers lost data and produces sharper, more detailed representations of the input frames by meticulously crafting a deblurring technique that reverses the effects of motion blur. This module functions as a fundamental part of the larger EDVR architecture, coming before feature extraction and fusion and other later restoration steps. It can adjust to different degrees of motion blur and different types of video footage, using temporal information from adjacent frames to improve its deblurring performance. The PreDeblur module greatly improves the EDVR model's overall resilience, allowing it to handle motion-blurred real-world video sequences and generate visually appealing results. Depending on the EDVR model variant being utilized, certain implementation details may change. Additional information can be found by consulting the model's source code or documentation.

3.3.1.2 PCD Align Module
One key component of the EDVR (Enhanced Deep Video Restoration) model, which is intended to handle issues with distortions and misalignments in video frames, is the PCD (Pyramid, Cascading, and Deformable) Align module. The module utilizes a hierarchical pyramid structure and multi-level feature maps to accomplish accurate alignment at various spatial scales. With the help of this hierarchical method, the PCD Align module can adjust for misalignments brought on by the motion and deformations in the video sequence. Additionally, by repeatedly improving the alignment process, cascading mechanisms improve consistency in the final output and feature map alignment accuracy. The module can more successfully capture spatial relationships by adaptively adjusting receptive fields due to the presence of deformable convolutions, particularly in locations with large deformations or structural changes.
Within the larger framework of the EDVR architecture, the PCD Align module is essential to the preprocessing phases, coming before later processes like the Temporal Feature Enhancement Network and the Feature Fusion Network. Through precise feature map alignment, this module makes it easier for later steps to use coherent data for the best possible video super-resolution. The hierarchical and iterative structure of the PCD Align module makes it an excellent tool for addressing the many difficulties presented by real-world video sequences, which greatly enhances the model's overall performance in video restoration tasks. Depending on the model variant, the PCD Align module's specific implementation details and optimizations may change. For a more thorough understanding, it is advised to consult the model's source code or documentation.

3.3.1.3 TSA Fusion Module
A key element that concentrates on efficiently fusing information across both temporal and spatial dimensions in the context of the EDVR (Enhanced Deep Video Restoration) model is the TSA (Temporal and Spatial Attention) Fusion module. Selected attention to pertinent features in the temporal (across frames) and spatial (across regions within a frame) domains is what drives its principal goal of improving the quality of video frames. The TSA Fusion module's main features will be dissected as follows:

1. Temporal Attention:
With the use of temporal attention processes, the TSA Fusion module can identify and highlight pertinent information from several frames in a video sequence. This allows the model to consider the dynamism and changes in content over time, which is especially significant for video super-resolution jobs.


2. Spatial Attention:
The module incorporates mechanisms for spatial attention in addition to temporal attention. This lets the model concentrate on areas of interest throughout each frame, allowing details to be selectively enhanced in areas that make the biggest contributions to the overall visual quality.
3.Feature Fusion:
Through a feature fusion procedure, the TSA Fusion module integrates the data obtained from both temporal and spatial attention mechanisms. Through this fusion step, the model is guaranteed to be able to utilize the most pertinent and instructive features for the restoration process's next stages.
4. Adaptability to Varying Contexts:
The TSA Fusion module's capacity to adapt to different situations within video sequences is one of its strongest points. Through dynamic adjustments of the attention weights in response to the content of various frames and areas, the module can manage a wide range of problems, including changes in scene dynamics, motion, and occlusions.
5. Integration into EDVR Pipeline:
In the larger EDVR architecture, the TSA Fusion module is usually integrated after elements such as the PCD (Pyramid, Cascading, and Deformable) Align module. It plays a vital part in determining how features are represented before they move on to further phases, such as the Feature Fusion Network, which enhances the overall effectiveness of video restoration.
6. Impact on Video Super-Resolution:
The TSA Fusion module contributes significantly to enhancing the accuracy and perceived quality of the final high-resolution video frames generated by the EDVR model by paying particular attention to pertinent temporal and spatial characteristics. It contributes to the preservation of spatial features and temporal coherence, producing visually stunning super-resolved videos.
To summarize, the EDVR TSA Fusion module effectively fuses information from numerous frames and areas by integrating temporal and spatial attention mechanisms. This improves the model's performance in video super-resolution. Because of its flexibility and seamless integration into the larger architecture, it is essential to getting high-quality outcomes in video restoration jobs that are performed in the real world.

3.3.1.4 Reconstructive Module
An essential part of the EDVR (increased Deep Video Restoration) model is the reconstruction module, which combines the increased features from earlier phases of the model to create the final high-resolution video frames. The enhanced feature maps are supplied into the Reconstruction module to reconstruct the high-resolution output after passing through procedures like feature extraction, alignment, and fusion. The module effectively restores finer details and enhances the overall visual quality of the video by upscaling the spatial resolution of the feature maps using carefully crafted architectures and procedures. By utilizing sophisticated methods like non-linear activations and deconvolutional layers, the reconstruction module makes sure that the super-resolved frames have more contrast and definition. The Reconstruction module's crucial role in converting the enriched feature representations into a logical and aesthetically pleasing series of high-resolution video frames and achieving the model's main goal in video restoration tasks is highlighted by the module's successful integration into the EDVR pipeline.
Summary of EDVR Model:
For efficient video super-resolution, the Enhanced Deep Video Restoration (EDVR) model consists of several essential modules. The procedure is started by the PreDeblur module, which takes care of input frame blur. Accurate motion correction and alignment are ensured by the PCD (Pseudo-Convolutional Distillation) alignment module, which is essential for high-quality restoration. Then, to improve overall detail aggregation, the TSA (Temporal and Spatial Attention) fusion module cleverly mixes data from many frames. The procedure is completed by the Reconstructive module, which also ensures consistency and improves restoration quality by fine-tuning the output frames. These modules work together to create a complete pipeline in the EDVR paradigm for advanced video super-resolution.

3.3.2 BasicVSR
BasicVSR emerges as a formidable player, presenting a generic and efficient baseline that redefines the landscape of video super-resolution. Characterized by minimal redesigns of existing components, BasicVSR introduces a novel approach to high-efficiency video enhancement. At its core, BasicVSR adopts a bidirectional recurrent network, a fundamental architecture that enables the simultaneous propagation of information both forward and backward in time. This bidirectional scheme proves instrumental in addressing inherent challenges within the video super-resolution domain, allowing the model to capture nuanced temporal dynamics and outperform existing state-of-the-art methods. As we delve into the intricacies of BasicVSR, we explore its distinctive components, including innovative propagation strategies and precise alignment methodologies, that collectively contribute to its exceptional prowess in elevating the visual fidelity of video content.

 
Figure 14 Basic VSR Architecture and Propagations [16]
3.3.2.1 Propagation
 One of the most important aspects of VSR is propagation. It outlines the methods for utilizing the information in a video sequence. There are three primary categories of existing propagation schemes: local, unidirectional, and bidirectional. We address the shortcomings of the local & unidirectional in the sections that follow to justify our decision to use bidirectional propagation in BasicVSR.
Local Propagation: The limitations of sliding-window methods in video super-resolution are underscored as they rely on local information within a confined window of low-resolution frames for restoration. This design restricts the accessible information to a local neighbourhood, leading to the neglect of distant frames. To substantiate this claim, an experiment is conducted, commencing with a global receptive field, and gradually reducing it by segmenting test sequences into K segments. BasicVSR is employed to independently restore each segment, and the resulting PSNR (Peak Signal-to-Noise Ratio) difference is compared to the case of global propagation (K=1). The analysis reveals a reduction in the PSNR difference as the number of segments decreases, signifying improved performance with an increased temporal receptive field. This emphasizes the beneficial nature of information from distant frames in the restoration process. Furthermore, the experiment highlights that the largest PSNR difference occurs at the two ends of each segment, indicating the necessity of adopting long sequences to accumulate long-term information effectively. In summary, the findings stress the importance of a global receptive field and the limitations of relying solely on local information within a sliding window for optimal video super-resolution. [16]
Unidirectional Propagation: The issue mentioned earlier can be solved by using a unidirectional propagation approach, where information is passed sequentially from the first frame to the last frame in a video sequence. However, in this setup, there's a problem with the balance of information received by different frames. Specifically, the first frame gets no information from the rest of the sequence, except for itself, while the last frame receives information from the entire sequence. This imbalance leads to less-than-ideal results, especially for the earlier frames.
 
Figure 15 Unidirectional Propagation frame index vs PSNR
Early timesteps receive less information in unidirectional propagation, which results in subpar performance. A lower PSNR than the bidirectional counterpart is shown by values smaller than zero (dotted line). Due to the bidirectional model's zero feature initialization, it should be noted that the unidirectional model performs better than the bidirectional model only in the final frame. [16]
To illustrate this, we compared BasicVSR using bidirectional propagation (information moving both forward and backward) with a version using unidirectional propagation, but with a similar network complexity. Looking at the results in Fig. 4, we found that the unidirectional model performs significantly worse in terms of PSNR (Peak Signal-to-Noise Ratio) at early time steps. The performance difference gradually reduces as more information is considered with an increase in the number of frames. Interestingly, we consistently observed a drop of 0.5 dB in performance when using only partial information. These observations highlight the suboptimal nature of unidirectional propagation. To enhance output quality, one effective strategy is to propagate information backward from the last frame of the sequence.
Bidirectional Propagation: Bidirectional propagation is an effective technique to address the two previously described problems. In this method, features are propagated separately forward and backward in time. The fundamental design of BasicVSR is a bidirectional recurrent network, which permits simultaneous information flow both forward and backward in time. This bidirectional method allows the model to consider information from both previous and subsequent frames, which helps to address issues related to video super-resolution. The BasicVSR upsampling module (U), denoted by the colours red and blue for forward and backward propagations, combines several convolutions and pixel-shuffle operations. This bidirectional technique makes use of extensive temporal context to generate high-resolution frames.
Moreover, bidirectional propagation is quite helpful for BasicVSR's propagation branches, which contain parts for residual blocks, spatial warping, and flow estimation. The bidirectional nature of the model guarantees that it considers temporal relationships in both directions, captures optical flow, and spatially warps and refines features based on this flow. In video super-resolution challenges, BasicVSR outperforms current state-of-the-art approaches thanks to the bidirectional recurrent network, proving the value of taking a wider temporal context into account throughout the restoration process.
3.3.2.2 Alignment
Video Super-Resolution (VSR) relies heavily on alignment since it tackles the problem of aligning similar but misaligned pictures or features for later aggregation. Three major categories can be used to classify conventional approaches in the context of VSR: feature alignment, image alignment, and without alignment.
Without Alignment: Explicit alignment strategies are not necessary for some VSR procedures to function. This method ignores any potential spatial misalignments between consecutive frames as the model examines input frames independently. This may result in undesirable outcomes, particularly in situations where misalignments are noticeable, like in different, real-world video sequences.
Image Alignment: Aligning complete images to increase consistency between frames is the focus of another class of techniques. Image alignment approaches use transformations like translation, rotation, or scaling to adjust spatial disparities across frames. While these techniques improve the overall alignment, localized fine-grained features might not be captured.
Feature Alignment: In contrast, feature alignment techniques aim to precisely align high-level features that are taken out of the input frames. These techniques concentrate on aligning feature representations—which may contain optical flow fields or other pertinent features—rather than aligning complete images. When the misalignments are not consistent throughout the image, feature alignment is useful because it enables more focused rectification.
BasicVSR uses a feature alignment approach in this instance. Bidirectional recurrent networks with thoughtfully crafted elements, such as residual blocks and optical flow, are included in the model. Features can propagate separately forward and backward in time when bidirectional propagation is used. By reducing the impact of spatial misalignments, this bidirectional feature alignment helps the model produce higher-quality, super-resolved video frames. When obtaining tiny details is crucial to the overall quality of the super-resolved output, or when misalignments are non-uniform, feature alignment is especially helpful. In summary, BasicVSR employs feature alignment to address spatial misalignments in video sequences, contributing to its effectiveness in achieving state-of-the-art performance in video super-resolution tasks.
3.3.2.3 Aggregation & Upsampling
Information from nearby frames is integrated during the aggregation process, which is made possible by elements like residual blocks and optical flow. Accurate feature alignment in the model is made possible by optical flow estimation, which aids in comprehending motion between frames. Important features are captured and preserved during the aggregation process with the help of residual blocks. BasicVSR's aggregation technique was carefully designed to ensure that the model can take full use of temporal relationships to provide better video super-resolution performance.
Increasing an image's or feature representation's spatial resolution is known as upsampling. The upsampling module in BasicVSR oversees producing high-resolution frames from the combined features. Typically, the upsampling module (designated as 'U' in the BasicVSR architecture) consists of several convolutions and pixel-shuffle operations.
Pixel-Shuffle: To effectively raise the spatial resolution of a low-resolution feature map, the elements of each spatial block are rearranged using the pixel-shuffle upsampling approach. Reconstructing precise details in the super-resolved frames requires this operation.
Convolutions: In the upsampling module, convolutional layers help to improve the details derived from the aggregated data and to fine-tune the spatial characteristics. Upscaled features are smoothed, and complex patterns are captured with the aid of convolutional techniques. The upsampling module's mix of convolutions and pixel-shuffle guarantees that the model can produce high-quality, super-resolved frames with better spatial resolution. The aggregated data must be upsampled to produce visually pleasing and crisp video frames.
In conclusion, the BasicVSR Aggregation and Upsampling procedures cooperate to take advantage of temporal dependencies, align features, and boost spatial resolution, all of which result in the creation of excellent super-resolved video sequences. These elements help the model achieve cutting-edge results in the field of super-resolution video.
Summary of BasicVSR
BasicVSR is motivated by a thoughtful analysis. It employs bidirectional propagation for effective long-term and global information processing. Alignment is achieved through a straightforward flow-based method at the feature level. Popular techniques like feature concatenation and pixel-shuffle are utilized for aggregation and upsampling. Despite its simplicity, BasicVSR excels in both restoration quality and efficiency. Moreover, its versatility allows seamless integration of additional components for handling more complex scenarios, as demonstrated later.
3.3.3 Icon VSR 
In the evolution of the BasicVSR model into IconVSR, two innovative components, namely the Information-Refill Mechanism and Coupled Propagation, have been introduced to enhance the model's performance.
3.3.3.1 Information-Refill Mechanism
The Information-Refill Mechanism addresses the challenge of inaccurate alignment in occluded regions and along image boundaries, a common source of error accumulation in long-term propagation scenarios. This mechanism selectively refines features from keyframes, employing an additional feature extractor to extract deep features and fuse them with aligned features through convolution. The refined features are then processed through residual blocks, effectively mitigating errors, and enhancing overall restoration quality. [16]
3.3.3.2 Coupled Propagation
Coupled Propagation component redefines the bidirectional settings by interconnecting the propagation modules. Traditionally, features are propagated independently in two opposite directions, but in Coupled Propagation, features propagated backward are utilized as inputs in the forward propagation module. This interconnected scheme allows the forward propagation branch to receive information not only from past frames but also from future frames, resulting in higher-quality features and improved overall output quality. Notably, the coupled intervals in IconVSR maintain global information propagation, setting it apart from existing methods that isolate intervals for independent processing.

 
Together, these components Coupled Propagation & Information-Refill Mechanism contribute to the effectiveness of IconVSR in mitigating errors during propagation and facilitating robust information aggregation for video super-resolution.
3.3.4 Metrics 
Within the domain of video super-resolution (VSR), a metric function as a numerical gauge for assessing how well algorithms perform in terms of improving the resolution of low-quality video frames. This score offers an unbiased evaluation of how effectively a VSR model replicates the specifics and calibre of the initial high-definition video. Peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), mean squared error (MSE), and other metrics that assess temporal consistency and perceptual characteristics of video quality are examples of common metrics. These measures are essential for comparing and evaluating various VSR algorithms objectively. They also help practitioners and researchers improve their models for better video reconstruction outcomes.

3.3.4.1 Peak Signal-to-Noise Ratio (PSNR)
Peak Signal-to-Noise Ratio (PSNR), which measures the degree of distortion generated during the process, is a commonly used metric in image and video processing to assess the quality of a reconstructed or compressed signal. Evaluating the improved video's quality in relation to its original high-resolution counterpart is especially important in video super-resolution (VSR).[18]
The formula to calculate PSNR is given by:
PSNR =10 Log10 (MAX2/ MSE)
MAX is the maximum possible pixel value of the image or video (commonly 255 for 8-bit images),
MSE is the mean squared error between the original and reconstructed signals.
PSNR is a numerical value that represents the ratio of the peak signal intensity to the noise generated by the reconstruction process, and it is expressed in decibels (dB). Lower distortion levels are correlated with higher PSNR values, which indicate improved detail preservation fidelity. PSNR is significant since it is straightforward and simple to understand. Superior image or video quality is implied by a higher PSNR value, which indicates a closer match between the original and reconstructed signals. It's important to remember that PSNR does not necessarily correspond with perceptual quality because it assumes that all aberrations are equally noticeable to the human eye, which is not always the case. Because of its simple computation and easy interpretation, PSNR is still a commonly used statistic despite its drawbacks.
Model	PSNR Score
EDVR	30.4261 dB
Basic VSR	31.4255 dB
IconVSR	31.7017 dB
Table 3.1 PSNR SCORES
3.3.4.2 Structural Similarity Index (SSIM)
In image and video processing, the Structural Similarity Index (SSIM) is a metric that is used to measure how similar two signals are, considering not only pixel-by-pixel changes but also structural details and perceptual characteristics. When comparing the visual quality of enhanced recordings to their high-resolution counterparts, SSIM is very helpful in video super-resolution (VSR).
The SSIM formula involves three key components luminance, contrast & structure. The SSIM scale goes from -1 to 1, with 1 denoting perfect similarity. In contrast to PSNR, which only considers pixel-by-pixel variations, SSIM considers perceptual factors, which puts it more in line with how people see the world. The value of SSIM is found in its capacity to offer a more complex assessment of picture and video quality. Because it considers elements like brightness, contrast, and structure, it is sensitive to aberrations that conventional measurements might not be able to adequately represent. Since SSIM is an index of similarity rather than an absolute measure, its unit is dimensionless.
Model	SSIM Score
EDVR	0.8690
Basic VSR	0.8915
IconVSR	0.8957
Table 3.2 SSIM SCORES
To sum up, SSIM is essential to VSR because it provides a more thorough evaluation of visual fidelity that is more in line with human perception than pixel-by-pixel comparisons. It offers a more comprehensive assessment of the calibre of super-resolved videos, balancing out measurements like PSNR.

3.3.4.3 PSNR & SSIM as Single metric
It is better to comprehend image or video quality by combining the two metrics. When pixel-by-pixel accuracy is critical, PSNR works well; nevertheless, SSIM excels at capturing perceptual details. Visual quality frequently relies on human perception and interpretation of pictures in addition to numerical accuracy. Because it takes structural information into account, SSIM is useful for situations where preserving perceptual accuracy is crucial.
A greater PSNR may not always translate into better perceptual quality in some image processing scenarios. In certain cases, global characteristics and structural details are important. By helping to detect these cases, SSIM makes it possible to conduct a fairer assessment that takes perceptual similarity and pixel-level accuracy into account.
 
Figure 17 Comparison of all three models on the scale of SSIM & PSNR
The combination facilitates better decision-making and directs advancements in models and algorithms. It is especially helpful in situations when maintaining visual quality and reaching numerical precision require trade-offs.
In summary, the combination of PSNR and SSIM results in a more thorough evaluation of picture and video quality that takes into consideration both technical precision and perceptual relevance. This joint strategy is essential for improving and developing technologies pertaining to image and video processing.
3.3.5 Visual Results
The outputs from the IconVSR, BasicVSR, and EDVR models are presented below, showcasing subtle yet noteworthy differences. In the Icon VSR result, the impact might appear marginal at first glance. However, upon closer inspection, minute details, such as the number plate of a car, become more discernible in the enhanced image. Moving to the Basic VSR output, the significance of the model becomes apparent. The clarity improvement is evident when zooming in on the after image, revealing details like the price displayed on a board that were less distinguishable in the original. The EDVR output, in the third instance, underscores the model's ability to bring out pronounced distinctions. Notably, the faces of individuals in the output exhibit enhanced visibility, emphasizing the model's proficiency in capturing finer details.

    After								Before


After							Before
 





Figure 19: Output of BasicVSR





After							Before
 
After							Before
                                         
Figure 20 Output of EDVR

Chapter 4:  Conclusion: 

In conclusion, this research delved into the realm of Video Super Resolution (VSR) with a focus on the significance of improving visual quality in low-resolution videos while efficiently managing bandwidth resources. The investigation centred on three prominent VSR models—BasicVSR, EDVR, and IconVSR—and aimed to identify the most effective technique for video reconstruction quality. The comprehensive literature review provided a foundational understanding of existing and traditional VSR methodologies and their applications, emphasizing the critical role of VSR in conserving bandwidth.
The evaluation process, utilizing Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) scores, offered quantitative insights into the performance of each model. The comparative analysis revealed distinctive strengths and weaknesses, ultimately positioning IconVSR as the superior performer, with the highest PSNR and SSIM scores. This outcome underscores the model's effectiveness in achieving optimal video reconstruction quality.
MODEL NAME	PSNR SCORE	SSIM
EDVR	30.4261 dB	0.8690
BASICVSR	31.4255 dB	0.8915
ICONVSR	31.7017 dB	0.8957
Table 4.1 PSNR & SSIM of all models
The findings of this research have implications for various applications where video quality is paramount, especially in scenarios with bandwidth constraints. The identified superior performance of IconVSR suggests its potential for practical implementation in real-world settings, contributing to enhanced visual experiences without compromising bandwidth efficiency.

Future work in Video Super Resolution (VSR) is poised to explore and enhance existing models, with a specific focus on delving deeper into advanced versions such as BasicVSR++. The dynamic nature of the field opens avenues for refining and extending the capabilities of these models. One key direction involves the continuous improvement and adaptation of deep learning architectures to better capture intricate motion patterns and complex temporal dependencies within video sequences. Researchers may explore novel techniques to enhance the performance of BasicVSR++ by integrating more sophisticated components, optimizing network architectures, or leveraging additional sources of information. Additionally, future work can delve into addressing specific challenges, such as achieving a finer balance between spatial accuracy and computational efficiency. In this research we carried out all the model’s performance on REDS dataset in future work we can validate using the other dataset such vimeo90k or vid4 dataset.

References

1.	Liu, H., Ruan, Z., Zhao, P. et al. Video super-resolution based on deep learning: a comprehensive survey. Artif Intell Rev 55, 5981–6035 (2022). https://doi.org/10.1007/s10462-022-10147-y
2.	Jiang Z, Huang Y, Hu L. Single Image Super-Resolution: Depthwise Separable Convolution Super-Resolution Generative Adversarial Network. Applied Sciences. 2020; 10(1):375. https://doi.org/10.3390/app10010375
3.	Huynh-Thu, Q., Ghanbari, M. The accuracy of PSNR in predicting video quality for different video scenes and frame rates. Telecommun Syst 49, 35–48 (2012). https://doi.org/10.1007/s11235-010-9351-x
4.	K. S and M. Durgadevi, "Generative Adversarial Network (GAN): a general review on different variants of GAN and applications," 2021 6th International Conference on Communication and Electronics Systems (ICCES), Coimbatre, India, 2021, pp. 1-8, doi: 10.1109/ICCES51350.2021.9489160.
5.	K. T. Gribbon and D. G. Bailey, "A novel approach to real-time bilinear interpolation," Proceedings. DELTA 2004. Second IEEE International Workshop on Electronic Design, Test and Applications, Perth, WA, Australia, 2004, pp. 126-131, doi: 10.1109/DELTA.2004.10055.
6.	P., Miklos. (2007). Comparison of Convolutional Based Interpolation Techniques in Digital Image Processing.  87-90. Available from: 10.1109/SISY.2007.4342630
7.	M. Elad and M. Aharon, "Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries," in IEEE Transactions on Image Processing, vol. 15, no. 12, pp. 3736-3745, Dec. 2006, doi: 10.1109/TIP.2006.881969.
8.	K. Dabov, A. Foi, V. Katkovnik and K. Egiazarian, "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," in IEEE Transactions on Image Processing, vol. 16, no. 8, pp. 2080-2095, Aug. 2007, doi: 10.1109/TIP.2007.901238.
9.	Zhang, J., Zhao, D. and Gao, W., 2014. Group-based sparse representation for image restoration. IEEE transactions on image processing, 23(8), pp.3336-3351.
10.	Egmont-Petersen, M., de Ridder, D. and Handels, H., 2002. Image processing with neural networks—a review. Pattern recognition, 35(10), pp.2279-2301.
11.	Vb, S.K., 2020. Perceptual image super resolution using deep learning and super resolution convolution neural networks (SRCNN). Intelligent Systems and Computer Technology, 37(3).
12.	Fadnavis, S., 2014. Image interpolation techniques in digital image processing: an overview. International Journal of Engineering Research and Applications, 4(10), pp.70-73.
13.		arXiv:2012.02181 [cs.CV] (or arXiv:2012.02181v2[cs.CV] for this version  https://doi.org/10.48550/arXiv.2012.02181 
14.	D. Liu et al., "Learning Temporal Dynamics for Video Super-Resolution: A Deep Learning Approach," in IEEE Transactions on Image Processing, vol. 27, no. 7, pp. 3432-3445, July 2018, doi: 10.1109/TIP.2018.2820807.
15.	A. Kappeler, S. Yoo, Q. Dai and A. K. Katsaggelos, "Video Super-Resolution With Convolutional Neural Networks," in IEEE Transactions on Computational Imaging, vol. 2, no. 2, pp. 109-122, June 2016, doi: 10.1109/TCI.2016.2532323.
16.		arXiv:2012.02181 [cs.CV] (or arXiv:2012.02181v2 [cs.CV] for this version) https://doi.org/10.48550/arXiv.2012.02181
17.	arXiv:1905.02716 [cs.CV] (or arXiv:1905.02716v1 [cs.CV] for this version) https://doi.org/10.48550/arXiv.1905.02716
18.	A. Horé and D. Ziou, "Image Quality Metrics: PSNR vs. SSIM," 2010 20th International Conference on Pattern Recognition, Istanbul, Turkey, 2010, pp. 2366-2369, doi: 10.1109/ICPR.2010.579.



