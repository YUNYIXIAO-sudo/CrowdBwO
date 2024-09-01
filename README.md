# CrowdBwO
## Introduction
This repository contains the implementation of CrowdBwO for Efficient Crowdsourcing Task Assignments.

## Requirements
Install the required libraries using:
```bash
pip install -r requirements.txt
```

## Experiment on Synthetic Workers

Run the task assignment experiment on synthetic workers under problem setting 1 (S1) and 2 (S2) using:
```bash
./ExpSyntheticWorkers/runS1exp.sh
./ExpSyntheticWorkers/runS2exp.sh
```

Plot the experiment results using:
```bash
./ExpSyntheticWorkers/getS1Figures.sh
./ExpSyntheticWorkers/getS2Figures.sh
```

## Experiment on Real Workers

We used MTurk API to publish tasks on Amazon Mechanical Turk ([API Reference](https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/Welcome.html)).
The result data of CrowdBwO is in AMTexperimentA and the data of other baselines is in AMTexperimentB.

Plot the experiment results using:

```bash
./ExpRealWorkers/getFigures.sh
```

