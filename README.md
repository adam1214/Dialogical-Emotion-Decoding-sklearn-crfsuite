# Dialogical-Emotion-Decoding-sklearn-crfsuite
## Intro.
* Implement by a python package [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)
    * Construct a NER system：[Example code](https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb)
* Using dataset：IEMOCAP
## Usage for tuning model (`with_tuning/` dir)
### Train Model(Because of 5 fold CV, need to train 5 model)
*    python CRF_train.py -n 1
*    python CRF_train.py -n 2
*    python CRF_train.py -n 3
*    python CRF_train.py -n 4
*    python CRF_train.py -n 5
*    This 5 models would be saved in `with_tuning/model/` dir

### Test Model UAR & ACC
*    python CRF_test.py

## Results
### Without tuning model
|                | Original Training Data UAR | Original Training Data ACC |
| -------------- | -------------------------- | -------------------------- |
|sequential_utt     |0.7183|0.7022|
|spk_info            |0.7114|0.7064|
|fixed_len_5_spk_info  |0.6848|0.6803|
|fixed_len_6_spk_info  |0.6866|0.6843|
|fixed_len_7_spk_info  |0.6781|0.6722|
|fixed_len_8_spk_info  |0.6956|0.6905|
|fixed_len_9_spk_info  |0.6924|0.6897|
|fixed_len_10_spk_info  |0.6822|0.6800|
|fixed_len_11_spk_info  |0.6531|0.6500|
|fixed_len_12_spk_info  |0.6587|0.6592|
|fixed_len_13_spk_info  |0.6779|0.6767|
|fixed_len_14_spk_info  |0.6920|0.6809|
|fixed_len_15_spk_info  |0.6708|0.6711|
|fixed_len_16_spk_info  |0.6707|0.6668|
|fixed_len_17_spk_info  |**0.6958**|**0.6907**|
|fixed_len_18_spk_info  |0.6825|0.6807|
|fixed_len_19_spk_info  |0.6474|0.6610|
|fixed_len_20_spk_info  |0.6407|0.6422|

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