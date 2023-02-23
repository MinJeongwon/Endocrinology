import torch
from tqdm import dqtm
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


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

        color_dict = dict({'Bone':'#d35b64',
                            'Diabetes':'#4a7bcb',
                            'Others': '#c5a96c',
                            'Pitu/Adrenal': '#fe7d22',
                            'Thyroid': '#b37cc4',
                            'X': '#6dcb6d'})
        sns.scatterplot(data=df, x='x', y='y', hue='classes', ax=ax[i], palette=color_dict).set_title("Layer {}".format(layer_i)) #palette : Paired muted pastel  

    plt.savefig(os.path.join(args.res, args.model+"_"+args.type)+'/layerwise_embeddings_epoch_{}.png'.format(epoch+1), pad_inches=0)
    plt.close()