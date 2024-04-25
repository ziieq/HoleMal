HoleMal is a lightweight host-level NIDS for resource-constrained environment.

extract_feature.py: input a pcap file, output a sample csv file. The csv file can be used to train detectors.

run_time_test.py: evaluate the runtime performance of processing a pcap file. set listener_or_read = 1 before starting.

train_classifier.py: train and evaluate ml classifiers. Modify the csv path before starting.
