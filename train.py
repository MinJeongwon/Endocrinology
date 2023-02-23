import argparse
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup
    )
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from tqdm import tqdm
import time
import datetime
import matplotlib.pyplot as plt
import os 
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data import split_data, EndoDataset

from models.model_bert import EndoCls
from models.model_roberta import EndoClsRoberta

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


#Training function
def train(model, data_loader, optimizer, device, scheduler, n_examples, epoch, args):
    model = model.train()
    losses = []
    correct_predictions = 0
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

        if args.n_gpu > 1:
            loss = loss.mean()
        
        correct_predictions += torch.sum(preds == targets).item()
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if ((epoch+1)==1 or (epoch+1)%10==0) and (30<batch_idx<61):
            if type(layerwise_hidden_states) == type(None):
                    layerwise_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in outputs[2])
                    layerwise_attn_mask = attention_mask.cpu()
                    layerwise_label = targets.cpu()
            else:
                layerwise_hidden = outputs[2]
                layerwise_hidden_states = tuple(torch.cat([ex,layer_hidden_state_batch.cpu()]) for ex,layer_hidden_state_batch in zip(layerwise_hidden_states, layerwise_hidden))
                layerwise_attn_mask = torch.cat([layerwise_attn_mask, attention_mask.cpu()])
                layerwise_label = torch.cat([layerwise_label, targets.cpu()])

    if ((epoch+1)==1) or ((epoch+1)%10==0):
        visualize_layerwise_embeddings(layerwise_hidden_states, layerwise_attn_mask, layerwise_label, epoch, args)

    return correct_predictions / n_examples, np.mean(losses)


#Evaluation function 
def eval(model, data_loader, device, n_examples, args):
    model = model.eval()
    losses = []
    correct_predictions = 0
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

            if args.n_gpu > 1:
                loss = loss.mean()

            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())

    return correct_predictions / n_examples, np.mean(losses)


#Prediction function
def get_predictions(model, data_loader, args):
    print("Testing the Best-Perfomred Model")
    model = model.eval()
    sequences = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            title = data["title"]
            abstract = data["abstract"]
            texts = [a+"[SEP]"+b for a, b in zip(title, abstract)]
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            targets = data["target"].to(args.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = targets
            )

            _, preds = outputs[0], torch.max(outputs[1], dim=1)[1]
            sequences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs[1])
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return sequences, predictions, prediction_probs, real_values


# visualization function
def visualization(train_score, dev_score, path, scorename, ct):
    plt.figure(ct)
    plt.plot(train_score)
    plt.plot(dev_score)
    plt.title('Model '+ scorename)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(path+"/"+scorename+".png")
    print(scorename +" graph saved!")
    plt.close()


def visualize_layerwise_embeddings(layerwise_hidden_states, masks, labels, epoch, args, layers_to_visualize=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    dim_reducer = TSNE(n_components=2)
    num_layers = len(layers_to_visualize)

    fig = plt.figure(figsize=(24,(num_layers/4)*6)) #each subplot of size 6x6, each row will hold 4 plots
    fig.suptitle('Embeddings in vector space at layer {}'.format(epoch+1), fontsize=16)

    ax = [fig.add_subplot(int(num_layers/4),4,i+1) for i in range(num_layers)]

    labels = labels.cpu().numpy().reshape(-1)
    label_to_class = {0:'Bone', 1:'Diabetes', 2:'Others', 3:'Pitu/Adrenal', 4:'Thyroid', 5:'X'}
    new_labels = np.array([label_to_class[label] for label in labels])
    for i, layer_i in enumerate(layers_to_visualize):
        layer_embeds = layerwise_hidden_states[layer_i]
        layer_averaged_hidden_states = torch.div(layer_embeds.sum(dim=1), masks.cpu().sum(dim=1, keepdim=True))
        layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states.detach().numpy())
        df = pd.DataFrame.from_dict({'x':layer_dim_reduced_embeds[:,0],'y':layer_dim_reduced_embeds[:,1],'classes':new_labels})

        color_dict = dict({'Bone':'#d65f5f',
                            'Diabetes':'#4878cf',
                            'Others': '#ab6aca',
                            'Pitu/Adrenal': '#ffffff',
                            'Thyroid': '#6acc65',
                            'X': '#c4ad66'})
        sns.scatterplot(data=df, x='x', y='y', hue='classes', ax=ax[i], palette=color_dict).set_title("Layer {}".format(layer_i)) #palette : Paired muted pastel  

    plt.savefig(os.path.join(args.res, args.model+"_"+args.type)+'/layerwise_embeddings_epoch_{}.png'.format(epoch+1), pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--source_data_dir', type=str, default='source')
    parser.add_argument('--max_sequence_len', type=int, default=512)
    parser.add_argument('--type', type=str, default="tytle_abst")
    parser.add_argument('--num_labels', type=int, default=6, help="Number of classes of the fine-tuned model")
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--res', type=str, default='outputs')
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--n_warmup_steps', type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = parser.parse_args()


    # make output directories
    os.makedirs(args.log, exist_ok=True)
    os.makedirs(os.path.join(args.res, args.model+"_"+args.type), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint, args.model+"_"+args.type), exist_ok=True)


    # log settings
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename=os.path.join(args.log, args.model+"_"+args.type+".log"),
                        filemode="w") # 로그파일 작성 https://jh-bk.tistory.com/40    \
    logger = logging.getLogger(__name__)
    logger.info("Model: " + args.model)


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
    class_dict = { name:value for name, value in zip(encoded_classes, classes) }
    class_dict = dict(sorted(class_dict.items()))
    logger.info("Label Encoding: " + str(classes) + "-->" + str(np.sort(encoded_classes)))
    class_values = list(class_dict.values())


    # model
    config = AutoConfig.from_pretrained(args.model)
    if "roberta" in args.model.lower(): #TODO roberta
        model = EndoClsRoberta.from_pretrained(args.model, config=config, num_labels=args.num_labels) 
    else:
        model = EndoCls.from_pretrained(args.model, config=config, args=args, num_labels=args.num_labels) 
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()


    # multi gpu settings
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)   


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
    optimizer = AdamW(params=model.parameters(), lr=args.lr)
    total_steps = len(train_data_loader) * args.epoch
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps = args.n_warmup_steps,
                    num_training_steps = total_steps
                    )


    # store best score
    history = defaultdict(list)
    history = {'train_acc': [],
               'train_loss': [],
               'val_acc': [],
               'val_loss': []
            }
    best_accuracy = best_epoch = 0


    # training
    print("Training the Model")
    t0 = time.time()
    for epoch in range(args.epoch):
        logger.info(f'[Epoch {epoch + 1}/{args.epoch}]')
        print(f'\nEpoch {epoch + 1}/{args.epoch}')

        train_acc, train_loss = train(
                                model,
                                train_data_loader,
                                optimizer,
                                args.device,
                                scheduler,
                                train_dataset.__len__(),
                                epoch,
                                args
                                )
        logger.info('****** Train result ******   loss : {:.3f}, accuracy : {:.3f}'.format(train_loss, train_acc))

        val_acc, val_loss = eval(
                            model,
                            valid_data_loader,
                            args.device,
                            dev_dataset.__len__(),
                            args
                            )
        logger.info('****** Validation result ******   loss : {:.3f}, accuracy : {:.3f}'.format(val_loss, val_acc))
        logger.info('=' * 70)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(args.checkpoint, args.model+"_"+args.type) + "/best_performed.bin")
            best_accuracy = val_acc
            best_epoch = epoch + 1

    time_consumed = time.time() - t0
    print("Training Finished! | took {}".format(str(datetime.timedelta(seconds=time_consumed))))
    logger.info("Training took {}".format(str(datetime.timedelta(seconds=time_consumed))))


    # result visualization
    assert len(history['train_acc'])==len(history['val_acc'])
    assert len(history['train_loss'])==len(history['val_loss'])
    visualization(history['train_acc'], history['val_acc'], os.path.join(args.res, args.model+"_"+args.type), "accuracy", ct=1)
    visualization(history['train_loss'], history['val_loss'], os.path.join(args.res, args.model+"_"+args.type), "loss", ct=2)


    # best score
    logger.info("Best Epoch: {}".format(best_epoch))
    logger.info("Best Accuracy: {:.3f}".format(best_accuracy))


    # load best model
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, args.model+"_"+args.type) + "/best_performed.bin"))
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
    df_dev.to_excel(os.path.join(args.res, args.model+"_"+args.type) + "/classification_result_dev.xlsx")


    # accuracy and classification report
    logger.info('=' * 70)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best Accuracy on the Validation set: {:.3f}".format(accuracy))
    logger.info("Classification Report")
    logger.info("\n{}\n".format(classification_report(y_test, y_pred, target_names = class_values)))
    logger.info('=' * 70)
    logger.info("Accuracy on the Validation set: {:.3f}".format(accuracy))
    logger.info('=' * 70)


if __name__ == "__main__":
    main()
