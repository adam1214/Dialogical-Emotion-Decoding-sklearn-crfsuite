# Dialogical-Emotion-Decoding-sklearn-crfsuite
* Implement by a python package [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)
    * Construct a NER system：[Example code](https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb)
* Using dataset：IEMOCAP

## Some other toolkits or resources can implement CRF
*    [CRF++](https://taku910.github.io/crfpp/)(Yet Another CRF toolkit)
      *    [Chinese reference](https://taku910.github.io/crfpp/)
      *    Developed by C++
      *    Need to use template file(fixed format and syntax) to define features
      *    Does not provide Python training interface, only Python test interface
      *    Other people's work：[Chinese word segmentation system](https://github.com/phychaos/pycrfpp)

*    [pcrf](https://github.com/huangzhengsjtu/pcrf)
      *    Implemented in python2, numpy, scipy. 100% python.
      *    Train file format and Feature template are compatible with popular CRF implementations, like CRF++
