import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import logging
import os
from tqdm import tqdm

import torch
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from transformers import AutoTokenizer, AutoConfig

from data import EndoDataset
from models.model_bert import EndoClsBert
from models.model_roberta import EndoClsRoberta


# predictions
def get_predictions(model, data_loader, args, test_len):
    model = model.eval()
    losses = []
    correct_predictions = 0

    sequences = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            title = data["title"]
            abstract = data["abstract"]
            texts = [a + "[SEP]" + b for a, b in zip(title, abstract)]
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            targets = data["target"].to(args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )

            loss, preds = outputs[0], torch.max(outputs[1], dim=1)[1]
            sequences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs[1], dim=-1))
            real_values.extend(targets)

            if args.n_gpu > 1:
                loss = loss.mean()
            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return sequences, predictions, prediction_probs, correct_predictions / test_len, np.mean(losses), real_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--type', type=str, default="title_abst", help="")
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--num_labels', type=int, default=6, help="Number of classes of the fine-tuned model")
    parser.add_argument('--test', type=str, default='source/test.csv')
    parser.add_argument('--max_sequence_len', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--res', type=str, default='outputs')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--contrast_loss', default=0, type=int, help='Whether use contrastive model.')
    parser.add_argument('--lamb', default=0.3, type=float, help='lambda')
    parser.add_argument('--threshold', default=0.02, type=float, help='Threshold for keeping tokens. Denote as gamma in the paper.')
    parser.add_argument('--tau', default=1, type=float, help='Temperature for contrastive model.')
    parser.add_argument('--version', default="1", type=str, help='version')
    args = parser.parse_args()
    print(args)

    model_name = args.model.replace("/", "_") if "/" in args.model else args.model
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(os.path.join(args.res, model_name+"_"+args.type), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint, model_name+"_"+args.type), exist_ok=True)

    # log settings
    if "/" in args.model:
        model_name = args.model.replace("/", "_")
        log_path = os.path.join(args.log, str(datetime.date.today()) + "_" + model_name)+"_"+args.type+"_"+args.version+".log"
    else:
        log_path = os.path.join(args.log, str(datetime.date.today()) + "_" + args.model+"_"+args.type+"_"+args.version+".log")

    # log settings
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename=log_path,
                        filemode="a"
                        ) # 로그파일 작성 https://www.delftstack.com/ko/howto/python/python-log-to-file/
    logger = logging.getLogger(__name__)


    if len(args.test) == 0:
        print("You should specify the path of the test set")
        exit()


    # test set
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    test_dataset = EndoDataset(args, tokenizer=tokenizer, file_path="./source/test.csv", desc="test")
    test_data_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size = args.test_batch_size,
        collate_fn = test_dataset.collate_fn
    )
    classes, encoded_classes, test_set_shape = test_dataset.get_info()
    class_dict = {name:value for name, value in zip(encoded_classes, classes)}
    class_dict = dict(sorted(class_dict.items()))
    class_values = list(class_dict.values())
    print("shape of the test set: {} \n".format(test_set_shape))


    # model
    print("Loading the Model ...")
    config = AutoConfig.from_pretrained(args.model)
    model = EndoClsBert.from_pretrained(args.model, config=config, args=args, tokenizer=tokenizer, num_labels=args.num_labels, \
                                            label_dict=class_dict, contrast_loss=args.contrast_loss, lamb=args.lamb, tau=args.tau, threshold=args.threshold)
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()


    # multi gpu settings
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)   
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, model_name+"_"+args.type, str(datetime.date.today())) +"_"+args.version+ "_best_performed.bin", map_location = args.device))
    print("Model Loaded Successfully!")


    # start model prediction
    print("Classifying ...")
    _, y_pred, y_pred_probs, test_acc, test_loss, y_test = get_predictions(model, test_data_loader, args, test_dataset.__len__())
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    logger.info('=' * 70+'\n')
    logger.info('****** Test result ******   loss : {:.3f}, accuracy : {:.3f}, f1 : {:.3f}'.format(test_loss, accuracy, f1))


    # save model prediction
    pred_df = test_dataset.get_dataframe()
    predicted_label = []
    for y_pred_single in y_pred:
        predicted_label.append(class_dict[y_pred_single.item()])
    pred_df['predicted_label'] = predicted_label   

    for i in range(args.num_labels):
        pred_df["weight_class_"+str(i)] = y_pred_probs[:, i]
    pred_df.to_excel(os.path.join(args.res, model_name+"_"+args.type, str(datetime.date.today())) +"_"+args.version+ "/classification_result_test.xlsx") 
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on the Test set: {:.3f}".format(accuracy))


    # classification report
    logger.info("Classification Report")
    logger.info("\n{}".format(classification_report(y_test, y_pred, target_names = class_values)))
    logger.info('=' * 70)


    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, 
                annot=True, 
                fmt='.2%', 
                cmap='Blues',
                cbar=True,
                xticklabels=class_values,
                yticklabels=class_values)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig(os.path.join(args.res, model_name+"_"+args.type, str(datetime.date.today()))+"_"+args.version+"/confusion_matrix.png")

    print("Classification Finished!")


if __name__ == "__main__":
    main()

