# APTSniffer
## Introduction:

This is a project related to APT traffic detection.

This work has been submitted to **ICASSP2025**. 

We submitted part of the project, and the full project will be published after admission.

## Abstract:

Advanced Persistent Threats (APT) differ from traditional attacks like DDoS or webshells. They employ more complex and covert infiltration strategies to execute long-term assaults on target systems, posing a severe threat to organizational and national security. Identifying complex APT activities without relying on known Indicator of Compromise (IOC) clues has proven challenging. Due to problems like the shortage of APT traffic data and encrypted traffic obfuscation, existing methods cannot accurately identify APT traffic with just a few attack traffic samples.

To overcome the above limitation, we propose a novel encrypted APT traffic detection model, APTSniffer, which combines large language models (LLM) and retrieval-augmented technology. Our model first extracts features from the predicted traffic samples. Then, it uses retrieval-augmented technology to identify few-shot traffic information with similar behavior patterns and attack habits in historical traffic samples. Finally, the retrieved sample information and the auxiliary fine-tuning weight matrix, which has prior knowledge from the training set, are fed into a large generative language model for adaptive inference decisions. APTSniffer utilizes the few-shot inference and generalization abilities of large language models by converting raw traffic data into natural language inference examples understandable by the LLM. Experimental results show that, compared to other baseline models, APTSniffer exhibits SOTA performance. It achieves F1 scores above 97\% on three APT datasets, making it practically applicable for APT traffic detection tasks.

## Script Description:

  XX.py
      A 
      
  XX.py
      E
## Dataset:

The download link is as follows: https://drive.google.com/file/d/1ShVpTLPTsvGHdZ9xLT8JKKICEYAHHULg/view
