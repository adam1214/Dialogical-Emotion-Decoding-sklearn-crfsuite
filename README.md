# Dialogical-Emotion-Decoding-sklearn-crfsuite
## Intro.
* Implement by a python package [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)
    * Construct a NER system：[Example code](https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb)
* Using dataset：IEMOCAP
## Usage for tuning model (`with_tuning/` dir)
### Train Model(Because of 5 fold CV, need to train 5 model)

### Test Model UAR & ACC


## Results
### Without tuning model (l2sgd algo.)
|                       | Original Training Data UAR | Original Training Data ACC | Class to Class Training Data UAR | Class to Class Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------- | -------------------------------- | -------------------------------- | --- | --- |
| sequential_utt        | **0.7183**|**0.7022** |0.4948|0.5095|0.6768|0.6635|
| spk_info              | **0.7114**|**0.7064** |0.4889|0.5147|0.6950|0.6852|
| fixed_len_5_spk_info  | 0.6848|0.6803 |0.5320|0.5565|0.6888|0.6755|
| fixed_len_6_spk_info  | 0.6866|0.6843 |0.5414|0.5645|0.6963|0.6840|
| fixed_len_7_spk_info  | 0.6781|0.6722 |0.5174|0.5449|0.6974|0.6841|
| fixed_len_8_spk_info  | 0.6956|0.6905 |0.5443|0.5704|0.6974|0.6850|
| fixed_len_9_spk_info  | 0.6924|0.6897 |0.5314|0.5563|0.6968|0.6838|
| fixed_len_10_spk_info | 0.6822|0.6800 |0.5400|0.5599|0.6958|0.6836|
| fixed_len_11_spk_info | 0.6531|0.6500 |0.5139|0.5419|0.7042|0.6926|
| fixed_len_12_spk_info | 0.6587|0.6592 |0.4919|0.5221|0.7000|0.6885|
| fixed_len_13_spk_info | 0.6779|0.6767 |0.5206|0.5410|0.7042|0.6939|
| fixed_len_14_spk_info | 0.6920|0.6809 |0.5093|0.5413|0.6945|0.6852|
| fixed_len_15_spk_info | 0.6708|0.6711 |0.5337|0.5541|0.6993|0.6869|
| fixed_len_16_spk_info | 0.6707|0.6668 |0.5235|0.5498|0.7027|0.6903|
| fixed_len_17_spk_info | **0.6958**|**0.6907** |0.5146|0.5420|0.7008|0.6899|
| fixed_len_18_spk_info | 0.6825|0.6807 |0.5441|0.5717|0.6950|0.6849|
| fixed_len_19_spk_info | 0.6474|0.6610 |0.5146|0.5395|**0.7058**|**0.6948**|
| fixed_len_20_spk_info | 0.6407|0.6422 |0.5179|0.5446|0.7051|0.6934|

### Tuning model (lbfgs algo.)
|                       | Original Training Data UAR | Original Training Data ACC | Class to Class Training Data UAR | Class to Class Training Data ACC |Utt to Utt Training Data UAR|Utt to Utt Training Data ACC|
| --------------------- | -------------------------- | -------------------------- | -------------------------------- | -------------------------------- | --- | --- |
| sequential_utt        |0.6937|0.6822|0.5042|0.5254|0.6795|0.6635|
| spk_info              |**0.7149**|**0.7080**|0.4965|0.5234|0.6954|0.6843|
| fixed_len_5_spk_info  |0.6864|0.6791|0.5240|0.5491|0.6919|0.6791|
| fixed_len_6_spk_info  |0.6897|0.6858|0.5263|0.5522|0.6959|0.6836|
| fixed_len_7_spk_info  |0.7022|0.6948|0.5265|0.5525|0.6890|0.6784|
| fixed_len_8_spk_info  |0.6963|0.6890|0.5248|0.5500|0.6906|0.6782|
| fixed_len_9_spk_info  |0.6899|0.6852|0.5176|0.5435|0.6962|0.6852|
| fixed_len_10_spk_info |0.6910|0.6876|0.5258|0.5504|0.6967|0.6856|
| fixed_len_11_spk_info |0.6867|0.6829|0.5146|0.5433|0.7022|0.6912|
| fixed_len_12_spk_info |0.6777|0.6755|0.5142|0.5415|0.6971|0.6857|
| fixed_len_13_spk_info |0.6764|0.6729|0.5131|0.5417|0.7010|0.6907|
| fixed_len_14_spk_info |0.6973|0.6923|0.5006|0.5283|0.7020|0.6916|
| fixed_len_15_spk_info |0.6974|0.6934|0.5086|0.5353|0.7021|0.6903|
| fixed_len_16_spk_info |0.6935|0.6896|0.5033|0.5305|0.7005|0.6897|
| fixed_len_17_spk_info |0.6862|0.6845|0.5078|0.5364|0.7008|0.6903|
| fixed_len_18_spk_info |0.6748|0.6744|0.5079|0.5350|0.7071|0.6964|
| fixed_len_19_spk_info |0.6825|0.6829|0.5096|0.5382|**0.7110**|**0.7002**|
| fixed_len_20_spk_info |0.6805|0.6794|0.5188|0.5482|0.7068|0.6966|

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