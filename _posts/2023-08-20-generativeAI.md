---
title: "Generative AI in Medical mages"
date: 2023-08-20
---

Generative AI is a field with the potential to create new nonexistent/synthetic content that is eye-catching and eye-deceiving at the same time. This evolving part of AI has touched many domains, including text, audio, images, videos, etc. Even a combination of these domain types can be synthetically stitched together, which is very difficult to catch by the human eye. Amazingly, this mind-blowing creative content can be generated using the knowledge acquired from the existing data patterns. In other words, generative AI models are capable of using just the raw data to create new content with overall similar but not identical, rather unique information. This research area has experienced many breakthroughs in the past years and is currently the most trending topic in tech talks. Starting from 3D animations, gaming, media, and advertising, it has reached many more critical and crucial applications, such as healthcare.

Generative AI has the potential to transform and complement clinicians' abilities and hence, can improve patient care, drug discovery, and diagnostics. In addition, it can assist in overcoming the challenges faced by medical image analysis researchers due to lack of data, especially for some rare to find abnormalities. Including generative AI in clinical treatments provides various benefits, such as increased training samples for undersampled medical image classes, exemption from the procedures involved in collecting sensitive data and related privacy concerns, and personalized plans for retrospective treatments based on the patient's medical history. Apart from new content generation, the image-to-image translations supported by generative models introduce new ways to explore some significant tasks, such as noise/artifacts removal and domain adaptation. This article discusses a similar artifacts removal technique using CycleGAN which explores the abilities of generative models in transforming the uninformative colonoscopy image frames into clinically significant frames. This discussion is based on "Can Adversarial Networks Make Uninformative Colonoscopy Video Frames Clinically Informative? (Student Abstract)", published in AAAI 2023. The article covers the concept, implementation details, and code execution guidelines.

Colonoscopy videos are acquired using a coloscope mounted with a camera. During the procedure, a large amount of video frames are captured. However, not all frames satisfy the image quality requirements necessary for correct pathological diagnosis. The factors affecting the image quality include improper patient preparation and abrupt camera movements that introduce unwanted artifacts such as ghost colors, interlacing, and motion blur. The presence of these artifacts can hinder both manual and automated diagnosis of serious abnormalities like colorectal cancer. To address the issue of insignificant/uninformative frames and  extract obscured significant/informative details, an adversarial network based approach is discussed, inspired by the unpaired image-to-image translation supported by CycleGAN. In medical imaging, obtaining paired sets of data with uninformative and informative counterparts is difficult. Hence, it is important to adopt an unpaired approach that can learn from the data distribution of one domain (a pool of significant frames) and translate the images of another domain such that they are indistinguishable from the former. In short, we have a domain A containing uninformative frames and a domain B with informative frames. Our aim is to translate domain A frames into domain B frames such that the translated data is indistinguishable from the original insignificant frames.

```
Domain A: Uninformative colonoscopy frames
Domain B: Informative colonoscopy frames
Aim: Translate Domain A to Domain B
```
Note that we do not have paired data, i.e., no mapping between the images of two domains. This major challenge which we often come across in medical images, is gracefully handled by the concept introduced in CycleGAN. Like a general GAN, it has a generator and a discriminator but notice their count difference. *It has two generators and two discriminators.* Let's discuss their specific tasks.
The generator GAB learns the mapping function that targets to translate domain A frames to domain B frames, whereas the generator GBA learns to translate domain B frames to domain A frames. Both generators have their corresponding discriminator that aims to distinguish the synthetic images from the real ones.
```
G_AB_: A -> B
G_BA_: B -> A
DB: Distinguishes B from G(AB)
DA: Distinguishes B from G(BA)
```
