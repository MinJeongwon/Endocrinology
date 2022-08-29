import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
import os
from tqdm import tqdm
from torch import cuda
from data_loader import Dataset, DataLoader

device = 'cuda' if cuda.is_available() else 'cpu'

def get_predictions(model, data_loader):
    model = model.eval()
    sequences = []
    predictions = []
    prediction_probs = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            title = data["title"]
            abstract = data["abstract"]
            texts = [a + "[SEP]" + b for a, b in zip(title, abstract)]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs[0], dim=1)
            sequences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs[0])
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    return sequences, predictions, prediction_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--num_labels', type=int, default=6, help="Number of classes of the fine-tuned model")
    parser.add_argument('--test', type=str, default='source/test.csv')
    parser.add_argument('--max_sequence_len', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--res', type=str, default='')
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()
    args.model = args.model.lower()

    if len(args.test) == 0:
        print("You should specify the path of the test set")
        exit()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    test_set = Dataset(args.test, tokenizer, args.max_sequence_len, True)
    _, _, test_set_shape = test_set.get_info()
    #df_test = test_set.get_dataframe()
    print("shape of the test set: {} \n".format(test_set_shape))
    test_data_loader = DataLoader(test_set, args.test_batch_size, shuffle = False)

    print("Loading the Model ...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = args.num_labels) 
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # multi gpu
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)   
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, args.model) + "/best_performed.bin", map_location = device))
    print("The Model Loaded Successfully!")

    print("Classifying ...")
    class_dict = {0: 'bone', 3: 'diabetes', 4: 'others', 1: 'pitu/adrenal', 2: 'thyroid', 5: 'x'}
    _, y_pred, y_pred_probs = get_predictions(model, test_data_loader)
    pred_df = test_set.get_dataframe()
    predicted_label = []
    for y_pred_single in y_pred:
        predicted_label.append(class_dict[y_pred_single.item()])
    pred_df['predicted_label'] = predicted_label   
    for i in range(args.num_labels):
        pred_df["weight_class_"+str(i)] = y_pred_probs[:, i]
    pred_df.to_csv(os.path.join(args.res, args.model) + "/classification_result_test.csv", index = False)
    print("Classification Finished!")

if __name__ == "__main__":
        main()

