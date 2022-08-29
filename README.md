# Medical paper classification using transformer-based Pretrained Language Models
This repository contains the code and resources for text classification task using state-of-the-art Transformer models. This work aims to classifying medical paper into 6 categories [bone, pitu/adrenal, diabetes, thyroid, others, x].     
The major purpose is to firstly predict endocrinology paper and the one in other departments, and to secondly classify it into subgroups of endocrinological topics if the paper is  classified under endocrinology department. When given title and abstract of the paper, our model decides the specific category this paperbased on the information from the title and abstract. We explore how the performance of model prediction improves when given the abstracts as well as titles compared to when given only titles. 

## Data Format
Both train.py and inference.py scripts receive the datasets in .csv format with the following format:
|ID|category|TI|abst|
|:---:|:---:|:-----:|:-----:|
|25,729,272|bone|Long Term Effect of High Glucose and Phosphate Levels on the OPG/RANK/RANKL/TRAIL System in the Progression of Vascular Calcification in rat Aortic Smooth Muscle Cells.|...|
|25,750,573|pitu/adrenal|A giant carotid aneurysm with intrasellar extension: a rare cause of panhypopituitarism.|...|
|25,630,229|thyroid|Salivary gland dysfunction after radioactive iodine (I-131) therapy in patients following total thyroidectomy: emphasis on radioactive iodine therapy dose.|...|
|25,561,730|diabetes|Ligand Binding Pocket Formed by Evolutionarily Conserved Residues in the Glucagon-like Peptide-1 (GLP-1) Receptor Core Domain.|...|

## Models
All of the models on the [Huggingface](https://huggingface.co/transformers) that support `AutoModelForSequenceClassification` are supported by this repository and can be used by setting the model parameter of the train.py with the appropriate name of the model. Some of them are listed below and the others can be found on Huggingface website.
```
Models = {
    "BERT base uncased": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "Clinical BERT": "clinical_bert",
    "BioBERT": "biobert",
    "BleuBERT base uncased Pubmed": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
    "BleuBERT base uncased Pubmed MIMIC": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "Clinical Assertion / Negation Classification BERT": "bvanaken/clinical-assertion-negation-bert"
}
```

## Environment
```shell
conda env create --file environment.yaml
```

## Train
In order to fine-tune each of the transformer models on your dataset, you can execute the following bash file:
```shell
bash train.sh
```
Please note that before executing the bash file, you need to define a set of files path in it.

#### Option
```
--model                     bert-base-uncased, roberta-base, etc.
--type                      whether to use both `title` and `abstract`(title_abst) for classification or just use `title`(title).
--source                    dataset dir.
--res                       path to result dir.
--log                       path to log dir.
--checkpoint                path to best-performed model dir.
--num_labels                number of classes.
--max_sequence_len          max length of sequence tokens.
--epoch                     number of epochs.
--train_batch_size          train batch size.
--valid_batch_size          evaluation batch size.
--lr                        learning rate.
--n_warmup_steps            warmup steps.
--local_rank                local rank.
```
## Inference
In order to inference the fine-tuned models, you can execute the following bash file:
```shell
bash inference.sh
```

### Option
```
--model                     bert-base-uncased, roberta-base, etc.
--type                      whether to use both `title` and `abstract`(title_abst) for classification or just use `title`(title).
--test                      path to test dataset.
--res                       path to result dir.
--log                       path to log dir.
--checkpoint                path to best-performed model dir.
--num_labels                number of classes.
--max_sequence_len          max length of sequence tokens.
--test_batch_size           test batch size.
```

## Results
<table class="tg">
  <tr>
    <td class="tg-fymr" rowspan="2">Classifier</th>
    <td class="tg-fymr" rowspan="2">Accuracy</th>
    <td class="tg-fymr" rowspan="2">Precision</th>
    <td class="tg-fymr" rowspan="2">Recall</th>
    <td class="tg-fymr" rowspan="2">F1</th>
  </tr>
<tbody>
  <tr>
    <td class="tg-xnov">BERT (base uncased)</td>
    <td class="tg-oyjm"><b> 0.85 </td>
    <td class="tg-oyjm"><b> 0.84 </td>
    <td class="tg-oyjm"><b> 0.85 </td>
    <td class="tg-oyjm"><b> 0.85 </td>
  </tr>
  <tr>
    <td class="tg-xnov">RoBERTa</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
  <tr>
    <td class="tg-xnov">Bio_ClinicalBERT</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
  <tr>
    <td class="tg-xnov">BleuBERT Pubmed (base unccased)</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
  <tr>
    <td class="tg-xnov">BleuBERT Pubmed MIMIC (base unccased)</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
  <tr>
    <td class="tg-xnov">SciBERT (base unccased)</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
    <tr>
    <td class="tg-xnov">Clinical Assertion/Negation Classification BERT</td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
    <td class="tg-xnov"> </td>
  </tr>
</tbody>
</table>

