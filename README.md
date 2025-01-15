# APTSniffer
## Introduction:

This is a project related to APT traffic detection.

## Abstract:

Advanced Persistent Threats (APT) differ from traditional attacks like DDoS or webshells. They employ more complex and covert infiltration strategies to execute long-term assaults on target systems, posing a severe threat to organizational and national security. Identifying complex APT activities without relying on known Indicator of Compromise (IOC) clues has proven challenging. Due to problems like the shortage of APT traffic data and encrypted traffic obfuscation, existing methods cannot accurately identify APT traffic with just a few attack traffic samples.

To overcome the above limitation, we propose a novel encrypted APT traffic detection model, APTSniffer, which combines large language models (LLM) and retrieval-augmented technology. Our model first extracts features from the predicted traffic samples. Then, it uses retrieval-augmented technology to identify few-shot traffic information with similar behavior patterns and attack habits in historical traffic samples. Finally, the retrieved sample information and the auxiliary fine-tuning weight matrix, which has prior knowledge from the training set, are fed into a large generative language model for adaptive inference decisions. APTSniffer utilizes the few-shot inference and generalization abilities of large language models by converting raw traffic data into natural language inference examples understandable by the LLM. Experimental results show that, compared to other baseline models, APTSniffer exhibits SOTA performance. It achieves F1 scores above 97\% on three APT datasets, making it practically applicable for APT traffic detection tasks.

## Script Description:

1. **a1_sam3_ok_label_distribution_pro2.py**: This corresponds to the exact sequence matching module.
2. **a2_rag2_sim.py**: This corresponds to the fuzzy sequence similarity matching module.
3. **a3_Corrs_2_ok.py**: This corresponds to the Traffic correlation graph matching module.
4. **main.py**: This is the main scheduling function, primarily responsible for generating label distribution and fine-tuning the weight matrix.
5. **main2_2_predict.py**: After generating label distribution and other information, this script calls the large language model for decision-making.


## Sample Explanation

Step 1：Assuming we have a PCAP file, we first extract features such as the payload packet length sequence and JA4 fingerprints for different flows in the PCAP file based on the flow (defined by the five-tuple of IP, port, and protocol).

Step 2: Then, we follow the steps below to transform the traffic feature data into understandable knowledge information.

Through exact and fuzzy matching of payload length sequences, we can effectively capture similar patterns in APT organizations' attack processes. For example, as shown in the figure below, even if the APT group changes the packet length in its attack, the new sequence of packet lengths remains highly similar to the original sequence.

<img width="707" alt="image" src="https://github.com/user-attachments/assets/66fcae87-50eb-4ac7-92fe-76626872905e">

As shown in the figure below, through graph connections, we can correlate the multi-hop traffic sample attribute information related to node A (B, C, D, E) rather than only associating with samples with the same attributes as node A (B, C).

<img width="653" alt="image" src="https://github.com/user-attachments/assets/5a3316ec-89fb-4e01-9440-070a4d248c67">


Step 3: After the above transformations, the traffic feature data will be converted into sample information as shown below. This information will then be input into a large language model to obtain the final prediction results.

Exact Match:
Benign: 20, APT: 50,
Recommended Weights: 0.2, 0.8

Fuzzy Match:
Benign: 50, APT: 52,
Recommended Weights: 0.4, 0.6

Associated Attribute Match:
Benign: 13, APT: 24,
Recommended Weights: 0.2, 0.8

The “Recommended Weights” in the above example represent the auxiliary weight matrix data trained through our simulated fine-tuning mechanism.

**An example of input for querying the large model is shown below.**

You  are  an  experienced  APT traffic analysis expert. Please classify the following traffic support data  into  one  of  the  categories  below  based  on  the  reference information  provided.  Categories:  Label0 Benign Traffic,  Label1 APT  Traffic.

Exact Match:
Benign: 20, APT: 50,
Recommended Weights: 0.2, 0.8

Fuzzy Match:
Benign: 50, APT: 52,
Recommended Weights: 0.4, 0.6

Associated Attribute Match:
Benign: 13, APT: 24,
Recommended Weights: 0.2, 0.8

## Dataset:

The Anyrun2024 dataset is our self-collected large-scale APT traffic dataset containing actual attack traffic samples from multiple APT groups. The APT traffic data in this dataset was manually selected from the Any.run sandbox by filtering for APT-labeled samples and then manually downloading the corresponding PCAP files. We downloaded 2044 PCAP files from different APT group attack samples, covering the years 2019-2024. To ensure the accuracy of the labels, we used Suricata to detect this dataset and filtered the traffic generated by the attack IPs from the alerts. We then used Tshark to extract the traffic generated by the attack IPs to ensure it was APT attack traffic. 

If you require the dataset file, please contact axuhongbo@126.com for access.

**Detailed Explanation of the Dataset:**

In the dataset file, there is a column representing the PCAP name, which contains detailed information about the dataset. The PCAP name is structured using the following format: APT group name_date, sample hash, five-tuple information, detected Suricata rule ID, ATT&CK tactic, and ATT&CK technique.

An example is as follows:
APT_APT10-20230130-684888079aaf7ed25e725b55a3695062-5_192.168.100.23_58521_37.48.65.148_80_sid-2826183_ttp-T1041_ta-TA0011.pcap

