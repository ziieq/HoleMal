# Introduction
HoleMal is a lightweight host-level NIDS for resource constrained environments. 

HoleMal provides a comprehensive suite of host-level traffic monitoring, processing, and detection solutions, aiming to achieve optimal network protection with minimal resource cost.

# Important Files
**df_maker.py:** This file is used to convert the pcap datasets into csv datasets for subsequent model training and testing.

**example_pcap2csv_single_thread.py:** (example_pcap2csv_multi_process.py is recommended.) This file implements HoleMal's process of extracting features from one pcap to csv. This file uses all host-level features. In the actual deployment, the a file needs to be run first to obtain the best feature subset, so as to improve the detection capability and detection speed in the specified scenario.

**example_pcap2csv_multi_process.py:** Same function as example_pcap2csv_single_thread.py, but with multi-process acceleration.

**main_experiment.py:** This file is used to test the metrics of HoleMal on various datasets with different chunk sizes.

**robust_experiment.py:** This file is used to test the HoleMal detection capability under different packet loss rates.

**run_time_test.py:**  This file is used to test the running efficiency of HoleMal. In our paper, this script is run on a resource constrained device.

**./detector_constructor/cost-sensitive_feature_selector/CoseSelector/pso.py:** This file is the script to run the Cost-Sensitive Feature Selector, which is responsible for selecting the feature subset with high classification ability and low time loss.

**./detector_constructor/cost-sensitive_model_selector/cost-sensitive_model_selector.py:** This file is the script to run the Cost-sensitive Model Selector, which is responsible for selecting the classification model with high detection capability and low time cost.

**dataset:** This folder includes processed HoleMal samples (results of df_maker.py) that are available for experimental use. This folder is shared on goolge drive (See **Other Large Files** below).

**HoleMal/detector_constructor/cost-sensitive_feature_selector/pcap:** This folder contains the traffic pcap used to obtain the FCTT, containing the IoT scenario traffic of the scenario. This folder is shared on goolge drive (See **Other Large Files** below).

# Flow of Execution
df_maker.py (results are already in dataset folder) -> ./detector_constructor/cost-sensitive_feature_selector/CoseSelector/pso.py (results are already in feature_subset_dict in subsequent scripts) -> main_experiment.py -> robust_experiment.py

example_pcap2csv.py and run_time_test.py can be run directly, but the source code for extracting features needs to be modified according to the scene.

# Other Large Files
We shared large files on https://drive.google.com/drive/folders/1bJThgSdmBgW74QVLZqAx3wx5XlUizJ6P?usp=drive_link

# Article
HoleMal: A lightweight IoT malware detection framework based on efficient host-level traffic processing

