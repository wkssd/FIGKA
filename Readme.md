# FI-GKA: Flexible and Intelligent Group Key Agreement for UANET

*Authors: Maintaining anonymity during the review phase! ! !*

**Keywords:** UANET, Group Key Agreement, CNN.

## Requirements

- Python 3.10

* NS-3 (version == 3.25)

## Description

The code in the repository implements the flexible and intelligent group key agreement scheme mentioned in the paper to test the performance of the proposed scheme. 

- **ModelTraining.py:** This code implements the training and accuracy test of the intelligent threshold selection model mentioned in this paper.
- **NS3.py:** This code implements the UANET environment simulation using NS-3, and tests the group key agreement success rate and group key agreement delay of the proposed scheme.
- **ConvertImg.py:** This code converts a network connectivity matrix M into a grayscale image.

## Installation

**Note:** The NS-3 simulation environment with python support needs to be installed, more details see: [ns-3 | a discrete-event network simulator for internet systems (nsnam.org)](https://www.nsnam.org)

### Python Dependencies:

- pytorch
- numpy
- PIL
- pandas

## Usage

- After altering the train parameters, to train a new model:
  ```bash
  $ python branching_dqn.py
  ```
