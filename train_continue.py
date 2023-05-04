import argparse
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    logging
    )
logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore') 

from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score, precision_recall_fscore_support #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
from collections import defaultdict
from tqdm import tqdm
import time
import datetime
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg',force=True)
#import matplotlib.pyplot as plt
import os 
import random

import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch.optim as optim

from data import split_data, EndoDataset
from models.model_bert import EndoClsBert
from models.model_roberta import EndoClsRoberta

from utils import get_predictions, visualization, visualize_layerwise_embeddings

# seed
def seed_everything(seed: int = 111):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:   
		raise argparse.ArgumentTypeError('Boolean value expected.')


#Training function
def train(model, data_loader, optimizer, device, scheduler, n_examples, epoch, args):
    model = model.train()
    losses = []
    correct_predictions = 0
    precisions, recalls, f1s = [], [], []
    layerwise_hidden_states, layerwise_attn_mask, layerwise_label = None, None, None
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Training")):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels = targets
        )

        loss, preds = outputs[0], torch.max(outputs[1], dim=1)[1]
        precision, recall, f1, _ = precision_recall_fscore_support(targets.cpu(), preds.cpu(), average='macro')

        if args.n_gpu > 1:
            loss = loss.mean()
        
        correct_predictions += torch.sum(preds == targets).item()
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        
        if ((epoch+1)==args.epoch) and (30<batch_idx<61):
        #if (30<batch_idx<61):
            if type(layerwise_hidden_states) == type(None):
                    layerwise_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in outputs[2])
                    layerwise_attn_mask = attention_mask.cpu()
                    layerwise_label = targets.cpu()
            else:
                layerwise_hidden = outputs[2]
                layerwise_hidden_states = tuple(torch.cat([ex,layer_hidden_state_batch.cpu()]) for ex,layer_hidden_state_batch in zip(layerwise_hidden_states, layerwise_hidden))
                layerwise_attn_mask = torch.cat([layerwise_attn_mask, attention_mask.cpu()])
                layerwise_label = torch.cat([layerwise_label, targets.cpu()])

    if ((epoch+1)==args.epoch):
        visualize_layerwise_embeddings(layerwise_hidden_states, layerwise_attn_mask, layerwise_label, epoch, args)
    #visualize_layerwise_embeddings(layerwise_hidden_states, layerwise_attn_mask, layerwise_label, epoch, args)

    return correct_predictions / n_examples, np.mean(losses), np.mean(precisions), np.mean(recalls), np.mean(f1s)


#Evaluation function 
def eval(model, data_loader, device, n_examples, args):
    model = model.eval()
    losses = []
    correct_predictions = 0
    precisions, recalls, f1s = [], [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = targets
            )

            loss, preds = outputs[0], torch.max(outputs[1], dim=1)[1]
            precision, recall, f1, _ = precision_recall_fscore_support(targets.cpu(), preds.cpu(), average='macro')

            if args.n_gpu > 1:
                loss = loss.mean()

            correct_predictions += torch.sum(preds == targets).item()
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            losses.append(loss.item())

    return correct_predictions / n_examples, np.mean(losses), np.mean(precisions), np.mean(recalls), np.mean(f1s)


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--source_data_dir', type=str, default='source')
    parser.add_argument('--max_sequence_len', type=int, default=512)
    parser.add_argument('--type', type=str, default="title_abst")
    parser.add_argument('--num_labels', type=int, default=6, help="Number of classes of the fine-tuned model")
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=28)
    parser.add_argument('--valid_batch_size', type=int, default=28)
    parser.add_argument('--res', type=str, default='outputs')
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_warmup_steps', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default="org", help="Scheduler options")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--contrast_loss', default=0, type=int, help='Whether use contrastive model.')
    parser.add_argument('--dropout_prob', default=0.2, type=float, help='Classifier dropout probability')
    parser.add_argument('--lamb', default=0.1, type=float, help='Weight for contrastive loss.')
    parser.add_argument('--threshold', default=0.05, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
    parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
    parser.add_argument('--loss_weight', type=int, default=0, help="Whether to use weight for cross entropy")
    parser.add_argument('--focal_loss', default=0, type=int, help='Whether to use focal loss')
    parser.add_argument('--class0_loss_w', type=float, default=0.9, help="Loss weight for `bone` class")
    parser.add_argument('--class1_loss_w', type=float, default=0.6, help="Loss weight for `diabetes` class")
    parser.add_argument('--class2_loss_w', type=float, default=0.8, help="Loss weight for `not endocrinology-related` class")
    parser.add_argument('--class3_loss_w', type=float, default=0.9, help="Loss weight for `others` class")
    parser.add_argument('--class4_loss_w', type=float, default=1., help="Loss weight for `pituitary adrenal` class")
    parser.add_argument('--class5_loss_w', type=float, default=0.9, help="Loss weight for `thyroid` class")
    parser.add_argument('--version', default="0", type=str, help='version')
    parser.add_argument('--save_when', default="accuracy", type=str, help='')
    parser.add_argument('--layerwise_pooling', default=False, type=str2bool, help='whether to use layerwise pooling')
    parser.add_argument('--pooler_type', default="cls", type=str, help='What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).')
    parser.add_argument('--last_ckpt', default=0, type=int, help='Whether to save the last checkpoint')
    args = parser.parse_args()
    print(args)


    # seed
    seed_everything(42)

    # make output directories
    model_name = args.model.replace("/", "_") if "/" in args.model else args.model
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(os.path.join(args.res, model_name+"_"+args.type), exist_ok=True)
    os.makedirs(os.path.join(args.res, model_name+"_"+args.type, "2023-04-28"+"_"+args.version), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint, model_name+"_"+args.type), exist_ok=True)


    # log settings
    if "/" in args.model:
        model_name = args.model.replace("/", "_")
        log_path = os.path.join(args.log, "2023-04-28" + "_" + model_name)+"_"+args.type+"_"+args.version+".log"
    else:
        log_path = os.path.join(args.log, "2023-04-28" + "_" + args.model+"_"+args.type+"_"+args.version+".log")

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename=log_path,
                        filemode="a") # 로그파일 작성 https://jh-bk.tistory.com/40    \
    logger = logging.getLogger(__name__)
    logger.info("Model: " + args.model)
    logger.info("Args: " + str(args))


    # split data into train/dev/test
    if len(os.listdir(args.source_data_dir))==1:
        split_data(args)


    # train dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_dataset = EndoDataset(args, tokenizer=tokenizer, file_path="./source/train.csv", desc="train")
    train_data_loader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = args.train_batch_size,
        collate_fn = train_dataset.collate_fn
    )


    # class encodings
    classes, encoded_classes, train_set_shape = train_dataset.get_info()
    class_dict = { value:name for value, name in zip(encoded_classes, classes)}
    class_dict = dict(sorted(class_dict.items()))
    logger.info("Label Encoding: " + str(classes) + "-->" + str(np.sort(encoded_classes)))
    class_values = list(class_dict.values())
    new_class_names = {"bone": "Bone & Calcium Metabolism",
                       "diabetes": "Diabetes & Metabolic Disease",
                       "pituitary adrenal": "Pituitary & Adrenal Disease",
                       "thyroid": "Thyroid Disease", 
                       "others": "Other Endocrine Disease",
                       "not endocrinology-related": "Unrelated Article"}
    class_values = [new_class_names[old] for old in class_values]
    for k, _ in class_dict.items():
        class_dict[k] = new_class_names[class_dict[k]]


    # loss weight for cross entropy
    if not args.loss_weight:
        loss_weight = None
    elif args.class0_loss_w<1:
        train_df = train_dataset.get_dataframe()
        class_dict_reversed = {v:k for k,v in class_dict.items()}
        class_num_list = [len(train_df[train_df['category']==cate]) for cate in class_values]
        loss_weight = [round(1 - (x / sum(class_num_list)), 1) for x in class_num_list] #version 4
        loss_weight = [round(weight/sum(loss_weight), 2) for weight in loss_weight] #version 4_1
    else:
        loss_weight = [args.class0_loss_w, args.class1_loss_w, args.class2_loss_w, args.class3_loss_w, args.class4_loss_w, args.class5_loss_w]


    # model
    config = AutoConfig.from_pretrained(args.model)
    if "roberta" in args.model.lower(): #TODO roberta
        model = EndoClsRoberta.from_pretrained(args.model, config=config, num_labels=args.num_labels) 
    else:
        model = EndoClsBert.from_pretrained(args.model, config=config, args=args, tokenizer=tokenizer, num_labels=args.num_labels, \
                                            label_dict=class_dict, contrast_loss=args.contrast_loss, lamb=args.lamb, tau=args.tau, threshold=args.threshold, loss_weight=loss_weight) 
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()


    # multi gpu settings
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)     
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, model_name+"_"+args.type, "2023-04-28" ) +"_"+args.version+ "_best_performed.bin", map_location = args.device))
    print("Model Loaded Successfully!")


    # dev dataset
    dev_dataset = EndoDataset(args, tokenizer=tokenizer, file_path="./source/dev.csv", desc="dev")
    valid_data_loader = DataLoader(
        dev_dataset,
        sampler = SequentialSampler(dev_dataset),
        batch_size = args.valid_batch_size,
        collate_fn = dev_dataset.collate_fn
    )
    _, _, dev_set_shape = dev_dataset.get_info()
    df_dev = dev_dataset.get_dataframe()

    logger.info("shape of the train set: {}".format(train_set_shape))
    logger.info("shape of the dev set: {}\n".format(dev_set_shape))
    logger.info('=' * 70)


    # optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_data_loader) * args.epoch
    if args.scheduler == "org":
        scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps = args.n_warmup_steps,
                    num_training_steps = total_steps
                    )
    elif args.scheduler == "warmup":
        scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps = (len(train_dataset)//args.train_batch_size) * 5, #2_1: (len(train_dataset)//args.train_batch_size) * 5
                    num_training_steps = total_steps
                    )
    elif args.scheduler == "polynomial":
        power = 3.0
        scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=(len(train_dataset)//args.train_batch_size)*5, 
                    num_training_steps=total_steps,
                    power=power
                    )
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=(len(train_dataset) // args.train_batch_size) * 4, 
                    num_training_steps=total_steps
                    )
    elif args.scheduler == "lambdalr":
        lambda1 = lambda epoch: 0.99 ** epoch
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    else:
        scheduler = None


    # store best score
    history = defaultdict(list)
    history = {'train_acc': [],
               'train_loss': [],
               'val_acc': [],
               'val_loss': []
            }
    best_epoch = best_score = 0

    # training
    print("Training the Model")
    t0 = time.time()
    for epoch in range(args.epoch):
        logger.info(f'[Epoch {epoch + 1}/{args.epoch}]')
        print(f'\nEpoch {epoch + 1}/{args.epoch}')

        train_acc, train_loss, train_pre, train_recall, train_f1 = train(
                                            model,
                                            train_data_loader,
                                            optimizer,
                                            args.device,
                                            scheduler,
                                            train_dataset.__len__(),
                                            epoch,
                                            args
                                            )
        logger.info('****** Train result ******   loss : {:.3f}, accuracy : {:.3f}, precision : {:.3f}, recall : {:.3f}, f1 : {:.3f}'.format(train_loss, train_acc, train_pre, train_recall, train_f1))

        val_acc, val_loss, val_pre, val_recall, val_f1 = eval(
                                    model,
                                    valid_data_loader,
                                    args.device,
                                    dev_dataset.__len__(),
                                    args
                                    )
        logger.info('****** Validation result ******   loss : {:.3f}, accuracy : {:.3f}, precision : {:.3f}, recall : {:.3f}, f1 : {:.3f}'.format(val_loss, val_acc, val_pre, val_recall, val_f1))
        #logger.info('=' * 70)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if args.save_when=="accuracy":
            val_score = val_acc
        elif args.save_when=="f1":
            val_score = val_f1
        elif args.save_when=="precision":
            val_score = val_pre
        elif args.save_when=="recall":
            val_score = val_recall
            
        condition = val_score > best_score
        if args.last_ckpt:
            condition = epoch+1==args.epoch

        if condition:
            torch.save(model.state_dict(), os.path.join(args.checkpoint, model_name+"_"+args.type, "2023-04-28") +"_"+args.version+ "_best_performed.bin")
            best_score = val_score
            best_epoch = epoch + 1
            print("[training continue] Saving best model({}={:.3f}) at Epoch {}".format(args.save_when, best_score, best_epoch))
            logger.info("[training continue] Saving best model({}={:.3f}) at Epoch {}".format(args.save_when, best_score, best_epoch))
        logger.info('=' * 70)

    time_consumed = time.time() - t0
    print("Training Finished! | took {}".format(str(datetime.timedelta(seconds=time_consumed))))
    logger.info("Training took {}".format(str(datetime.timedelta(seconds=time_consumed))))


    # result visualization
    assert len(history['train_acc'])==len(history['val_acc'])
    assert len(history['train_loss'])==len(history['val_loss'])
    visualization(history['train_acc'], history['val_acc'], os.path.join(args.res, model_name+"_"+args.type, "2023-04-28"+"_"+args.version), "accuracy", ct=1)
    visualization(history['train_loss'], history['val_loss'], os.path.join(args.res, model_name+"_"+args.type, "2023-04-28"+"_"+args.version), "loss", ct=2)


    # best score
    logger.info("Best Epoch: {:}".format(best_epoch))
    logger.info("Best {}: {:.3f}".format(args.save_when, best_score))


    # load best model
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, model_name+"_"+args.type, "2023-04-28") +"_"+args.version + "_best_performed.bin"))
    model = model.to(args.device)


    # predictions from best model
    _, y_pred, y_pred_probs, y_test = get_predictions(
                    model,
                    valid_data_loader,
                    args
                    )


    # append predicted label to dev df
    predicted_label = []
    for y_pred_single in y_pred:
        predicted_label.append(class_dict[y_pred_single.item()])
    df_dev['predicted_label'] = predicted_label   


    # append prediction probability for each class
    for i in range(len(encoded_classes)):
        df_dev["weight_class_"+str(i)] = y_pred_probs[:, i]
    df_dev.to_excel(os.path.join(args.res, model_name+"_"+args.type, "2023-04-28") +"_"+args.version+"/classification_result_dev.xlsx")


    # accuracy and classification report
    logger.info('=' * 70)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    if args.save_when=="accuracy": score=accuracy
    elif args.save_when=="f1": score=f1
    elif args.save_when=="precision": score=precision
    elif args.save_when=="recall": score=recall

    print("Best {} on the Validation set: {:.3f}".format(args.save_when, score))
    logger.info("Classification Report")
    logger.info("\n{}\n".format(classification_report(y_test, y_pred, target_names = class_values)))
    logger.info('=' * 70)
    logger.info("{} on the Validation set: {:.3f}".format(args.save_when, score))
    logger.info('=' * 70)


if __name__ == "__main__":
    main()
