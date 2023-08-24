
import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup # for optimizer and scheduler
import joblib
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random


ner_train = "./hi_train.txt"
ner_test = "./hi_dev.txt"
ner_path = "../input/a2-dataset/wikiann-mr.bio"
model_path = "./final_model.h5"


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print( torch.cuda.device_count())
    print('Available:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# ## Data Preprocessing


## preprocessing util
#uique method
def unique(list1):
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def addAttentionMasks(sentences,tokenizer):
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    input_ids = []
    attention_masks = []

    for sent in sentences:

        sent_str = ' '.join(sent)
        encoded_dict = tokenizer.encode_plus(
                        sent_str,                 
                        add_special_tokens = True,
                        truncation = True,
                        max_length = 235,           
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',     
                   )
    
        
        input_ids.append(encoded_dict['input_ids'][0])
    
        # And its attention mask
        attention_masks.append(encoded_dict['attention_mask'][0])
        
    return input_ids, attention_masks

def removeSpecialChar(data):
    # edit line with 3 words
    temp = []
    for i in data:
        if len(i) == 7:
            i[4], i[5] = i[6], i[6]
            temp.append(i[:6])
        else:
            temp.append(i)
    # remove special char
    special_char = [ "''", '७८', '४०', '५५', '[[', 'दे', '०६', '०५', '``', '३६', '७३', '६९', '१०', '९०', '४४', '८०', '४५', '७५', '७७', '००', 'or', '९६', '४६', '६४', '५३', '15', '13', 'at', '९९', 'in', '३४', 'of', 'an', 'on', '७०', '६७', '३३', '१७', '५२', '५०', '५६', '५९', '३९', '७४', '८५', '४९', '३७', '16', '९३', '७२', '६६', '६८', '३५', '४२', '४८', '७१', '३२', '५४', '३८', '४७', '४३', '५८', '३१', '९७', 'VC', 'XI', '६०', '५७', '९२', '**', '७६', '२३', '८३', '८२', '५१', '६१', '८१', '०४', '६२', 'In', 'is', 'no', '45', '09', '३०', '९५', '24', '79', '१९', '११', '२४', '९१', '१५', '०९', '--', '६५', '८६', '८४', "'-", '४१', '10', 'vc', '६३', '28', '21', '==', '२८', 'St', 'DD', 'DS', 'VD', 'TS', 'AT', 'BG', '१२', '७९', 'Si', 'O4', 'OH', 'Mg', 'Fe', '८९', '1२', '1८', 'to', '75', '90', '73', '59', '33', '35', '50', '46', '56', '92', '47', '43', '26', '23', '22', '74', '87', '82', 'as', '69', '२७']
    for i in temp:
        if len(i) == 3 and len(i[0]) < 2:
            special_char.append(i[0])
    special_char = unique(special_char)
    data = temp   

    # remove special char
    temp = []
    for i in data:
        if len(i) == 3 and i[0] in special_char:
            continue
        elif len(i) == 3 and i[1] == data[0][1]:
            temp.append([i[0],i[2]])
        elif len(i) == 6:
            temp.append([i[0],i[3]])
            temp.append([i[1],i[4]])
            temp.append([i[2],i[5]])
        elif len(i) == 0:
            temp.append(i)
    data = temp
    return data

def labelToNumber(input_ids,labels,label_map,tokenizer):
    ## change labels of numbers
    new_labels = []

    # The special label ID we'll give to "extra" tokens.
    null_label_id = -100

    for (sen, orig_labels) in zip(input_ids, labels):
        padded_labels = []

        orig_labels_i = 0 

        for token_id in sen:
            token_id = token_id.numpy().item()

            if (token_id == tokenizer.pad_token_id) or \
                (token_id == tokenizer.cls_token_id) or \
                (token_id == tokenizer.sep_token_id):
                    padded_labels.append(null_label_id)
            else:
                label_str = orig_labels[orig_labels_i]
                padded_labels.append(label_map[label_str])
                orig_labels_i += 1
        assert(len(sen) == len(padded_labels))    
        new_labels.append(padded_labels)
    return new_labels


### DATASET preprocessing

def preprocess(path,tokenizer):
    with open(path,"r",encoding='utf-8') as file:
        data = file.readlines()
    # split data
    data = [i.split() for i in data]
    
    List = []
    for i in data:
        if i == []:
            List.append(i)
        elif i[0] != '#':
            List.append(i)
    data = List
    
    #data = removeSpecialChar(data)
    
    # make sentence
    sentences = []
    labels = []
    tokens = []
    token_labels = []
    unique_labels = set()

    for line in data:
        if line == []:
            sentences.append(tokens)
            labels.append(token_labels)           
            tokens = []
            token_labels = []        
        else: 
            tokens.append(line[0])
            token_labels.append(line[3])
            unique_labels.add(line[3])
            
    # create label map 
    label_map = {}
    for (i, label) in enumerate(unique_labels):
        # Map it to its integer
        label_map[label] = i
    
    input_ids, attention_masks = addAttentionMasks(sentences,tokenizer)
    
    ## Correct labels as per tokennizer output
    temp_label = []
    for i in range(len(sentences)):
        l = []
        for j in range(len(sentences[i])):
            length = len(tokenizer.encode(sentences[i][j]))-2

            for k in range(length):
                l.append(labels[i][j])
        temp_label.append(l)
    labels = temp_label
    
    new_labels = labelToNumber(input_ids,labels,label_map,tokenizer)
    
    # convert to py torch
    # Concatenates a sequence of tensors along a new dimension
    # [14978, 235].
    pt_input_ids = torch.stack(input_ids, dim=0)
    pt_attention_masks = torch.stack(attention_masks, dim=0)
    pt_labels = torch.tensor(new_labels, dtype=torch.long)
    return pt_input_ids, pt_attention_masks, pt_labels
    
    


#  split data
def splitData(pt_input_ids, pt_attention_masks, pt_labels):

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(pt_input_ids, pt_attention_masks, pt_labels)

    # Create a split.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - (train_size)
    

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("split data:")
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    return train_dataset, val_dataset


def createDataLoader(train_dataset, val_dataset):
    # Create data loader
    
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size )
    validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size   )
    #test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size   )
    
    return train_dataloader, validation_dataloader


## Class for model
class IndicBertClassifier(nn.Module):
    def __init__(self):
        super(IndicBertClassifier, self).__init__()
        self.bert = AutoModelForTokenClassification.from_pretrained('ai4bharat/indic-bert',cache_dir="bert/", num_labels = 14)
        self.softmax = nn.Softmax(dim=2)
    
    
    
    def forward(self, t, mask, l) -> torch.Tensor:
        output = self.bert.forward(input_ids=t, attention_mask=mask, labels=l)
        output.logits = self.softmax(output.logits)
        return output
    
# train function
def train(train_dataloader, epochs = 4):
    # 1. Create model
    model = IndicBertClassifier()
    model.to(device)

    # 2. Create optimizer
    optimizer = AdamW(model.parameters(),lr = 5e-3,eps = 1e-8)

    # 3. Create scheduler
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []
    print('Training...')
    for epoch_i in range(0, epochs):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            print('Epoch {}/{} -- {}/{} '.format(epoch_i + 1, epochs,step, len(train_dataloader)), end="\r")
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            
            model.zero_grad()        
            outputs = model(b_input_ids, mask=b_input_mask, l=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)            
        loss_values.append(avg_train_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
    return model



def test(model, test_dataloader,pt_input_ids):
    print('Predicting labels for {:,} test sentences...'.format(len(pt_input_ids)))
    model.to(device)
    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
  
      # Telling the model not to compute or store gradients, saving memory and 

        with torch.no_grad():
        # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, mask=b_input_mask, l=b_labels)
            

        logits = outputs[1]
        #print(len(outputs[0]))

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
  
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    return predictions, true_labels

def eval(predictions, true_labels):
    # First, combine the results across the batches. # (2248, 235, 10)
    all_predictions = np.concatenate(predictions, axis=0)
    all_true_labels = np.concatenate(true_labels, axis=0)

    # For each token, pick the label with the highest score. # (2248, 235)
    predicted_label_ids = np.argmax(all_predictions, axis=2)

    # Eliminate axis 0, which corresponds to the sentences. # (528280,); (528280,)
    predicted_label_ids = np.concatenate(predicted_label_ids, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)

    # Remove padding and special token
    real_token_predictions = []
    real_token_labels = []

    # For each of the input tokens in the dataset...
    for i in range(len(all_true_labels)):
        # If it's not a token with a null label...
        if not all_true_labels[i] == -100:
            # Add the prediction and the ground truth to their lists.
            real_token_predictions.append(predicted_label_ids[i]) # 528,280
            real_token_labels.append(all_true_labels[i]) #69,031
        
    f1 = f1_score(real_token_labels, real_token_predictions, average='micro') 
    print ("F1 score: {:.2%}".format(f1))




def load_model(filename = '../input/a2-dataset/final_model.h5'):
    model = joblib.load(filename)
    return model.cuda()


def train_model(epochs = 2):
    # 1. load lokenizer
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert',cache_dir="bert/")

    # 2. Data preprocessing
    # preprocess data
    pt_input_ids, pt_attention_masks, pt_labels = preprocess(ner_train,tokenizer)

    # split data
    train_dataset, val_dataset = splitData(pt_input_ids, pt_attention_masks, pt_labels)
    # Create data loader
    train_dataloader, validation_dataloader = createDataLoader(train_dataset, val_dataset)

    # 6. Train model
    model = train(train_dataloader, epochs)

    # 7. save model
    print("Saving model")
    filename = 'final_model.h5'
    joblib.dump(model.cpu(), filename)



def test_model():
    
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert',cache_dir="bert/")

    pt_input_ids_test, pt_attention_masks_test, pt_labels_test = preprocess(ner_test,tokenizer)
    
    test_dataset = TensorDataset(pt_input_ids_test, pt_attention_masks_test, pt_labels_test)
    test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = 32   )
    model = load_model(model_path)
    predictions, true_labels = test(model, test_dataloader, pt_input_ids_test)
    
    eval(predictions, true_labels)


train_model(epochs = 1)
test_model()

# %% [code]
