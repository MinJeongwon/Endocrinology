import os
import logging
import pandas as pd
import re
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class EndoDataset(Dataset):
    def __init__(self, args, tokenizer, file_path:str, desc:str=None):
        self.args = args
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.desc = desc
        self._create_examples() 

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)
    
    def _create_examples(self):
        if self.file_path:
            data_file = self.file_path

        data = pd.read_csv(data_file, index_col=0)
        data['category'] = data['category'].str.replace('x', 'not endocrinology-related')
        data['category'] = data['category'].str.replace('pitu/adrenal', 'pituitary adrenal')
        labelEncoder = LabelEncoder()
        data['label'] = labelEncoder.fit_transform(data['category'])
        self.classes = labelEncoder.classes_
        self.encoded_classes = labelEncoder.transform(self.classes)
        self.df = data
        
        self.examples = []
        ct = 0
        for idx, row in data.iterrows():
            max_len = 0 

            # get data
            title = row['TI']
            abstract = self.process_abstract(row['abst'])
            category = row['category']

            # input seq: <s> premise </> hypothesis </s>
            if self.args.type == "title_abst":
                tokens_a = self.tokenizer.tokenize(title)
                tokens_b = self.tokenizer.tokenize(abstract)
                input_ids_a = self.tokenizer(title)['input_ids']
                input_ids_b = self.tokenizer(abstract)['input_ids']
                if len(tokens_a)+len(tokens_b)+3>self.args.max_sequence_len:
                    truncate = len(tokens_a)+len(tokens_b)+3-self.args.max_sequence_len
                    tokens_b = tokens_b[0:len(tokens_b)-truncate]

                tokens_a = ["[CLS]"] + tokens_a
                tokens_b = ["[SEP]"] + tokens_b + ["[SEP]"]
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens_a+tokens_b)
                input_tokens = tokens_a + tokens_b
                len_input_ids = len(input_ids)
                segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b)

            elif self.args.type == "title":
                text_a = "[CLS]" + title + "[SEP]"
                input_ids = self.tokenizer(text_a)['input_ids']
                len_input_ids = len(input_ids)
                input_tokens = self.tokenizer.tokenize(text_a)
                segment_ids = [0]*len_input_ids 

            # mask
            input_mask = [1] * len(input_ids)

            # label
            label_dict = {'bone':0, 'diabetes':1, 'others':2, 'pituitary adrenal':3, 'thyroid':4, 'not endocrinology-related':5}
            labels = label_dict[category]

            while len(input_ids) < self.args.max_sequence_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            # check
            assert len(input_ids)==len(segment_ids)==len(input_mask)
            assert len(input_ids)<=self.args.max_sequence_len

            self.examples.append({
                                'title': title,
                                'abstract': abstract,
                                'input_ids': input_ids,
                                'input_tokens': input_tokens,
                                'input_mask': input_mask,
                                'segment_ids': segment_ids,
                                'target':labels,
                                'category':category
                            })
            ct+=1
            if ct < 3:
                logger.info("*** Example ***")
                logger.info("tokens: %s" % " ".join(
                        [printable_text(x) for x in input_tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (category, labels))

            input_len = len(input_ids)
            max_len = max(input_len, max_len)

        self.max_len = max_len
        print('\n{} Data Statistics: {} examples'.format(self.desc.upper(), len(self.examples)))


    def process_abstract(self, abstract):
        abst = ""

        if "@NlmCategory" in abstract:
            text_idx = [i.end()+4 for i in re.finditer("#text", abstract)]
            end_idx = [i.start()-21 for i in re.finditer("@Label", abstract)]
            end_idx = end_idx[1:]+[len(abstract)-8]
            
            for a, b in zip(text_idx, end_idx):
                abst += abstract[a:b] + " " 

            rm_idx = abst.find(")])])")
            if rm_idx != -1:
                abst = abst[:rm_idx-1].strip()

        elif "OrderedDict" in abstract:
            idx = [i.start() for i in re.finditer("AbstractText", abstract)] 
            idx = idx[0]+len("AbstractText")+4
            abst = abstract[idx:-4].strip()

            rm_idx = abst.find("CopyrightInformation")
            if rm_idx != -1:
                abst = abst[:rm_idx-6].strip()

        else:
            abst = abstract

        return abst


    def _pad(self, sentences, pad_id):
        '''
            sentences: a list of list with ids
        '''
        #max_len = max((map(len, sentences))) #########
        max_len = self.max_len
        attention_mask = []
        sentences_pad = []
        for sent in sentences:
            pad_len = max_len - len(sent)
            sentences_pad.append( sent + [pad_id]*pad_len )
            attention_mask.append( [1]*len(sent) + [0]*pad_len)
        return sentences_pad, attention_mask


    def collate_fn(self, batch):
        '''
            to tensor
        '''
        input_ids = [example['input_ids'] for example in batch]
        input_ids, attention_mask = self._pad(input_ids, 0) 
        input_ids, attention_mask = torch.tensor(input_ids).long().to(self.args.device), torch.tensor(attention_mask).long().to(self.args.device)
        segment_ids = [example['segment_ids'] for example in batch]
        segment_ids, _ = self._pad(segment_ids, 0) 
        segment_ids = torch.tensor(segment_ids).long().to(self.args.device)
        target = [example['target'] for example in batch]
        target = torch.tensor(target).long().to(self.args.device)
        title = [example['title'] for example in batch]
        abstract = [example['abstract'] for example in batch]

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'segment_ids': segment_ids, 'target': target, 'title': title, 'abstract': abstract}


    def get_info(self):
        return self.classes, self.encoded_classes, self.df.shape


    def get_dataframe(self):
        return self.df


def split_data(args):
    if len(os.listdir(args.source_data_dir))!=4:
            dataset_excel = pd.read_excel("source/data.xlsx", engine="openpyxl", index_col=0)
            dataset_excel = dataset_excel.dropna(axis=0)
            dataset_excel = dataset_excel.sample(frac=1).reset_index(drop=True)
            
            total_n = len(dataset_excel)
            train_n, dev_n = int(total_n*0.8), int(total_n*0.9)
            
            dataset_excel.iloc[:train_n, :].to_csv(
                                                    "source/train.csv",
                                                    index = None,
                                                    header=True
                                                    ) 
            dataset_excel.iloc[train_n:dev_n, :].to_csv(
                                                        "source/dev.csv",
                                                        index = None,
                                                        header=True
                                                        ) 
            dataset_excel.iloc[dev_n:, :].to_csv(
                                                "source/test.csv",
                                                index = None,
                                                header=True
                                                ) 