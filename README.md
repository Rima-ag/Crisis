# Tweet Classification for Disaster Response

Compares Supervised and Unsupervised Learning models to filter tweet that are related to a natural disaster from those who are not.

Dataset: [CrisisMMDv2.0](https://arxiv.org/pdf/1805.00713.pdf)

## Supervised Models
Neural Network architectures inspired from this [paper](http://idl.iscram.org/files/xukunli/2020/2280_XukunLi+DoinaCaragea2020.pdf).

Three models were built:
1. RNN:  GRU for text processing
2. CNN: for image processing
3. MM: Multimodal approach combining both models for text and image processing

```
python deep_net_class.py --model MM
```

## Unsupervised Model
Adaptive Resonance Theory (ART1) Network for text processing
[Algorithm](https://www.emis.de/journals/GM/vol12nr3/popovici/popovici.pdf)

Text preprocessing was inspired from this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231219315723). 

```
python clustering.py
```

## Results

Models were tested on Hurricane IRMA tweets in CrisisMMDv2.0

|              | RNN | CNN | Multimodal | ART1 |
|--------------|-----|-----|------------|------|
| Accuracy (%) | 77  | 68  | 83         | 79   |
