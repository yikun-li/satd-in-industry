# Replication Package for Self-Admitted Technical Debt in the Embedded Systems Industry: An Exploratory Case Study

##### Authors: Yikun Li, Mohamed Soliman, Paris Avgeriou, and Lou Somers

## Description of this study:

Technical debt denotes shortcuts taken during software development, mostly for the sake of expedience.
When such shortcuts are admitted explicitly by developers (e.g., writing a TODO/Fixme comment), they are termed as *Self-Admitted Technical Debt* or *SATD*.
There has been a fair amount of work studying SATD management in Open Source projects, but SATD in industry is relatively unexplored. At the same time, there is no work focusing on developers' perspectives towards SATD and its management.
To address this, we conducted an exploratory case study in cooperation with an industrial partner to study how they think of SATD and how they manage it.
Specifically, we collected data by identifying and characterizing SATD in different sources (issues, source code comments and commits) and carried out a series of interviews with 12 software practitioners.
The results show: 1) the core characteristics of SATD in industrial projects; 2) developers' attitudes towards identified SATD and statistics; 3) triggers for practitioners to introduce and repay SATD; 4) relations between SATD in different sources; 5) practices used to manage SATD; 6) challenges and tooling ideas for SATD management.


## Structure of the replication package:

The replication package includes the interview documents, used scripts, and trained SATD detector.

```
├── LICENSE
├── README.md
├── _interview_document
│   ├── interview_instrument.pdf
│   └── qualitative_map.pdf
├── _other_scripts
│   ├── comment_extractor.py
│   └── requirements.txt
└── _satd_detector
    ├── requirements.txt
    └── satd_detector.py
```

## Getting Started With SATD Detector

### Requirements

- fasttext==0.9.2
- nltk==3.7
- torch==1.11.0

### Identifying SATD

1. Download model weight and word embedding files at [LINK](https://doi.org/10.5281/zenodo.6783762).
2. Unzip the fasttext_word_embeddings.bin.zip file.
3. Replace the file path with the real path and run the following command.

```bash
python satd_detector.py 
	--embed-vectors "{PATH}/fasttext_issue_300.bin"
	--snapshot "{PATH}/satd_detector.pt"
```

### Example Output

```
Source type: code_comment
Text: TODO: support multiple signers
Predicted result: requirement-debt

Source type: code_comment
Text: TODO: please add some javadoc
Predicted result: documentation-debt

Source type: code_comment
Text: TODO: lack of tests
Predicted result: test-debt

Source type: issue
Text: I would like to remove this as its no longer needed.
Predicted result: code|design-debt

Source type: issue
Text: to make their code more readable. I would like to see something like this in the API.
Predicted result: code|design-debt

Source type: issue
Text: To experiment with transfer learning, we first combine all the issue sections
Predicted result: non-SATD

Source type: issue
Text: We need to update this documentation
Predicted result: documentation-debt

Source type: issue
Text: There are unimplemented requirements
Predicted result: requirement-debt

Source type: issue
Text: This is a good patch
Predicted result: non-SATD

Source type: commit_message
Text: Get rid of some superfluous informational messages
Predicted result: code|design-debt

Source type: commit_message
Text: fix bugs in SystemML - removed XXX
Predicted result: non-SATD

Source type: commit_message
Text: fix typo in error message
Predicted result: documentation-debt

Source type: pull_request
Text: nit: use local variable if possible
Predicted result: code|design-debt

Source type: pull_request
Text: Use the Python Postinstall implementation by default
Predicted result: non-SATD
```

## Paper

Latest version available on [arXiv](https://arxiv.org/abs/2205.13872)

If you publish a paper where this word helps your research, we encourage you to cite the following paper in your publication:

```
@article{li2022self,
    author={Li, Yikun and Soliman, Mohamed and Avgeriou, Paris and Somers, Lou},
    journal={IEEE Transactions on Software Engineering},
    title={Self-Admitted Technical Debt in the Embedded Systems Industry: An Exploratory Case Study},
    year={2022},
    volume={},
    number={},
    pages={1-22},
    doi={10.1109/TSE.2022.3224378}
}
```

## Contact

- Please use the following email addresses if you have questions:
    - :email: <yikun.li@rug.nl>
