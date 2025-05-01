import re
import math
import os
import sys
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tag import pos_tag
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models import LdaModel
from gensim import models
from gensim.corpora import Dictionary
from transformers import BertPreTrainedModel, BertModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# ================================================================
#                             Proposed
# ================================================================
# ================================================================
# ulimit -n 65536
class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open("", 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# ================================================================
#                        personalized feature
# ================================================================
def per_feature(text_vec, text_data, text_original_data):
    """
    :param text_data
    :return: personalized features
    """
    batch_vec, batch_data = text_vec, text_data
    for i in range(len(batch_data)):
        if '[CLS]' in batch_data[i]:
            j = batch_data[i].index('[CLS]')
            batch_data[i].remove(batch_data[i][j])
    del i

    # ========= attribute_features ==========
    # --- 1. https_feature ---
    feature_https = []
    for i in range(len(batch_data)):
        if 'https' in batch_data[i]:
            feature_http = 1
        else:
            feature_http = -1
        feature_https.append(feature_http)
    del feature_http, i

    # --- 2. len_feature ---
    feature_len = []
    for i in range(len(batch_data)):
        if 'https' in batch_data[i]:
            j = batch_data[i].index('https')
            feature_le = j
        else:
            feature_le = len(batch_data[i])
        feature_len.append(feature_le / 50)
    del feature_le, i

    # ========= language_features ==========
    # --- 3-7. TRR_feature ---
    seq_len, pos_t, str_batch_datas = [], [], []
    for i in range(len(batch_data)):
        seq_le = len(batch_data[i])
        seq_len.append(seq_le)

        str_batch_data = " ".join(batch_data[i])
        pos_ta = pos_tag(word_tokenize(str_batch_data))
        pos_t.append(pos_ta)
        str_batch_datas.append(str_batch_data)

    Noun, Verb, Prep, ADJV, Other = [], [], [], [], []
    for i in range(len(pos_t)):
        noun, verb, prep, adjv, other = 0, 0, 0, 0, 0
        for j in range(len(pos_t[i])):
            if pos_t[i][j][1] in ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$']:
                noun += 1
            elif pos_t[i][j][1] in ['VB', 'VBP', 'VBG', 'VBZ', 'VBD', 'VBN']:
                verb += 1
            elif pos_t[i][j][1] in ['IN', 'WDT', 'CC', 'DT', 'UH']:
                prep += 1
            elif pos_t[i][j][1] in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'WRB']:
                adjv += 1
            else:
                other += 1
        # if seq_len[i] == 0:
        #     seq_len[i] = 1
        # else:
        if noun == 0:
            Noun.append(-0.5)
        else:
            if seq_len[i] == 0:
                Noun.append(-0.5)
            else:
                Noun.append(noun / seq_len[i])

        if verb == 0:
            Verb.append(-0.5)
        else:
            if seq_len[i] == 0:
                Verb.append(-0.5)
            else:
                Verb.append(verb / seq_len[i])

        if Prep == 0:
                Prep.append(-0.5)
        else:
            if seq_len[i] == 0:
                Prep.append(-0.5)
            else:
                Prep.append(prep / seq_len[i])

        if adjv == 0:
            ADJV.append(-0.5)
        else:
            if seq_len[i] == 0:
                ADJV.append(-0.5)
            else:
                ADJV.append(adjv / seq_len[i])

        if other == 0:
            Other.append(-0.5)
        else:
            if seq_len[i] == 0:
                Other.append(-0.5)
            else:
                Other.append(other / seq_len[i])
    del i, j, seq_le, pos_t, pos_ta, noun, verb, prep, adjv, other, str_batch_data

    # --- 8-9. emotion_features ---
    emotion, subjectivity = [], []
    for i in range(len(batch_data)):
        blob = TextBlob(str_batch_datas[i])
        polarity = blob.sentiment.polarity
        subjectivit = blob.sentiment.subjectivity
        emotion.append(polarity)
        subjectivity.append(subjectivit)
    del i, blob, polarity, subjectivit

    # --- 10. Thematic_features ---
    with suppress_stdout_stderr():
        Thamatic1, Thamatic2 = [], []
        dictionary = corpora.Dictionary(text_original_data)
        corpus = [dictionary.doc2bow(text) for text in text_original_data]
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=1)
        doc_topic = [a for a in lda[corpus]]
        for i in range(len(doc_topic)):
            if doc_topic[i][0][1] < doc_topic[i][1][1]:
                doc_topi1 = -0.5
                doc_topi2 = 0.5
            else:
                doc_topi1 = 0.5
                doc_topi2 = -0.5
            Thamatic1.append(doc_topi1)  
            Thamatic2.append(doc_topi2)

    # --- 11. ---
    def get_max_count(str):
        d = {}
        for i in str:
            count = str.count(i)
            d[i] = count
        list = []
        for i, v in d.items():
            list.append((i, v))
        for i in range(len(list) - 1):
            for j in range(i + 1, len(list)):
                if list[i][1] < list[j][1]:
                    list[i], list[j] = list[j], list[i]
        return list[0][1]

    interjection1 = []
    for i in range(len(batch_data)):
        inter1 = []
        for j in range(len(batch_data[i])): 
            inter, Char = [], []
            for k in range(len(batch_data[i][j])):
                char = batch_data[i][j][k]
                Char.append(char)
            max_count = get_max_count(Char)
            inter.append(max_count)
            inter1.append(inter)
        interjection1.append(inter1)

    Num_str = []
    for i in range(len(batch_data)):
        num_str = []
        for j in range(len(batch_data[i])):
            str_count = 0
            str_coun  = []
            for s in batch_data[i][j]:
                if s.isalpha():
                    str_count += 1
            str_coun.append(str_count)
            num_str.append(str_coun)
        Num_str.append(num_str)

    interjection = []
    for i in range(len(interjection1)):
        Inter = []
        for j in range(len(interjection1[i])):
            if Num_str[i][j][0] == 0:
                inter_rati = 0
            else:
                inter_rati = interjection1[i][j][0] / Num_str[i][j][0]
            Inter.append(inter_rati)
        interjection.append(Inter)

    inter_feature = []
    for i in range(len(interjection)):
        num_inte = 0
        for j in range(len(interjection[i])):
            if interjection[i][j] >= 0.5 and Num_str[i][j][0] >= 5:
                num_inte += 1
        if num_inte >= 1:
            num_inter = 0.5
        else:
            num_inter = -0.5
        inter_feature.append(num_inter)
    del i, k, s, char, max_count, inter, Char, Inter, inter1, inter_rati, interjection, interjection1, num_str, str_coun, str_count, num_inte, num_inter

    # --- 12. ---
    Avglen_feature = []
    for i in range(len(Num_str)):
        sum = 0
        for j in range(len(Num_str[i])):
            sum += Num_str[i][j][0]
        Avglen_feature.append((sum/(j+1)) / 10)

    # --- 13. ---
    text_original_token = []
    for i in range(len(text_original_data)):
        # for j in range(len(text_original_data[i])):
        text_original_data[i] = text_original_data[i][0].strip()
        text_original_toke = nltk.word_tokenize(text_original_data[i])
        text_original_token.append(text_original_toke)

    fragmentation_feature = []
    for i in range(len(text_original_token)):
        label = 0
        # for j in range(len(text_original_token[i])):
        if '.' in text_original_token[i]:
            label += 1
        if ',' in text_original_token[i]:
            label += 1
        if ';' in text_original_token[i]:
            label += 1
        if '?' in text_original_token[i]:
            label += 1
        if '!' in text_original_token[i]:
            label += 1
        if label == 0:
            label += 1
        label = 1 / label
        fragmentation_feature.append(label)

    personalized_feature = [feature_https, feature_len, Noun, Verb, Prep, ADJV, Other, emotion, Thamatic1, Thamatic2, subjectivity, inter_feature, Avglen_feature, fragmentation_feature]
    personalized_feature = torch.tensor(personalized_feature).to(device)
    personalized_feature = personalized_feature.t()
    return personalized_feature


# ================================================================
#                      deeplearning_section
# ================================================================
class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.n_class = 2
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 64
        self.encode_layer = args.encode_layer
        self.num_filter_total = self.num_filters * len(self.filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total+768, self.n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([self.n_class]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, self.num_filters, kernel_size=(size, self.hidden_size)) for size in self.filter_sizes])

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]

        # ----------------------------
        cls_embedding = x[:, 0, :]
        x = x[:, 1:, :]
        x = x.unsqueeze(1)  # [batch_size, channel=1, seq_len, hidden_size]
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))  # [batch_size, channel=1, seq_len - kernel_size + 1, 1]
            maxpooling = nn.MaxPool2d(kernel_size=(31 - self.filter_sizes[i] + 1, 1))  # maxpooling: [batch_size, channel=3, weight, h]
            h_maxpooling = maxpooling(h)
            pooled = h_maxpooling.permute(0, 3, 2, 1)  # [batch_size, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, dim=1)  # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        h_pool_flat_feature = torch.cat((cls_embedding, h_pool_flat), dim=1)
        return h_pool_flat_feature


class Bert_CNN(nn.Module):
    def __init__(self, args):
        super(Bert_CNN, self).__init__()
        self.args = args
        self.bert = args.model
        self.textcnn = TextCNN(args=args)
        self.fc1 = nn.Linear(974, 2)
        self.projection = nn.Linear(984, 2, bias=False).cuda()
        self.softmax = nn.Softmax(dim=-1)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, text_data, text_original_data):

        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask, output_hidden_states=True)

        hidden_states = outputs[2]
        cls_embedding = hidden_states[0][:, 0, :].unsqueeze(1)
        for i in range(1, 13):
            cls_embedding = torch.cat((cls_embedding, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)  # [batch_size, encoder_layer(12), hidden_size]
        cls_embedding1 = torch.mean(cls_embedding[:, 1:, :], dim=1)
        cls_embedding2 = cls_embedding1.unsqueeze(1)
        bertoutput = outputs[2][-1]
        wordoutputs = torch.cat((cls_embedding2, bertoutput), dim=1)

        personalized_feature = per_feature(text_vec=x[0], text_data=text_data, text_original_data=text_original_data)  # 个性化特征
        logit = self.textcnn(wordoutputs)
        logi = torch.cat((logit, personalized_feature), dim=1)

        # logit1 = self.cro_att1(sem_3dim_linear2, att2sem_3dim_linear2)  # bertout: [batch_size, 50, 768]
        # logit2 = self.cro_att2(att_3dim_linear2, sem2att_3dim_linear2)
        # logit1_attn = self.softmax(torch.tensor(logit1[0]))
        # logit1_attn = logit1_attn[:, 0, :, :]
        # logit1 = torch.matmul(att_feature_3dim, attn2)
        # logit2 = torch.matmul(sem_feature_3dim, attn1)
        # logit1 = self.sem2att(logit1)
        # logit2 = self.cross_att(logit2)
        # logit2 = logit2.squeeze(1)
        # logit1 = logit1.squeeze(1)
        logits = self.fc1(logi)
        logits = self.softmax(logits)
        return logits


# ================================================================
#                           FCN + aware
# ================================================================
class FCN_aware(nn.Module):
    def __init__(self, args):
        super(FCN_aware, self).__init__()
        self.args = args
        self.bert = args.model
        self.fc1 = nn.Linear(782, 2)
        self.softmax = nn.Softmax(dim=-1)

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, text_data, text_original_data):
        context, mask = x[0], x[2]
        _, outputs = self.bert(context, attention_mask=mask)
        personalized_feature = per_feature(text_vec=x[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((outputs, personalized_feature), dim=1)
        logits = self.fc1(outputs)
        logits = self.softmax(logits)
        return logits


# ============================================================
#                      TS_BiRNN + aware
# ============================================================
class TSBiRNN_aware(nn.Module):
    def __init__(self, args):
        super(TSBiRNN_aware, self).__init__()
        self.args = args
        D = 768
        N = 2
        dropout = 0.5
        C = 2
        self.bert = args.model
        self.fc = nn.Linear(1550, C)
        self.fc1 = nn.Linear(782, 2)
        self.lstm = nn.LSTM(D, 768, num_layers=N, bidirectional=True, batch_first=True, dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, text_data, text_original_data):
        context, mask = x[0], x[2]
        _, outputs = self.bert(context, attention_mask=mask)
        personalized_feature = per_feature(text_vec=x[0], text_data=text_data, text_original_data=text_original_data)
        out, _ = self.lstm(_)
        outputs = torch.cat((out[:, -1, :], personalized_feature), dim=1)
        logits = self.fc(outputs)
        logits = self.softmax(logits)
        return logits


# ============================================================
#                     BiLSTM_Dense + aware
# ============================================================
class BiLSTMDense_aware(nn.Module):
    def __init__(self, args):
        super(BiLSTMDense_aware, self).__init__()
        self.args = args
        self.bert = args.model
        D = 768
        N = 3
        H = 768
        C = 2
        self.lstm1 = nn.LSTM(D, H, num_layers=N, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(2 * H, H, num_layers=N, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(4 * H, H, num_layers=N, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * H, C)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(1550, 2)
        self.softmax = nn.Softmax(dim=-1)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, text_data, text_original_data):
        context, mask = x[0], x[2]
        _, outputs = self.bert(context, attention_mask=mask)
        out1, _ = self.lstm1(_)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(torch.cat([out1, out2], 2))
        out = torch.add(torch.add(out1, out2), out3)
        logit = out[:, -1, :]
        personalized_feature = per_feature(text_vec=x[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((logit, personalized_feature), dim=1)
        logits = self.fc1(outputs)
        logits = self.softmax(logits)
        return logits


# ============================================================
#                        LS_CNN + aware
# ============================================================
class LSCNN_aware(nn.Module):
    def __init__(self, args):
        super(LSCNN_aware, self).__init__()
        self.args = args
        self.args = args
        self.bert = args.model
        D = 768
        C = 2
        Ci = 1
        Co = 128
        Ks = [3, 5, 7]

        self.conv1_D = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(1, D))
        self.convK_1 = nn.ModuleList([nn.Conv2d(Co, Co, (K, 1)) for K in Ks])
        self.conv3 = nn.Conv2d(Co, Co, (3, 1))
        self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1, 0))
        self.CNN_dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks) * Co + 14, C)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, text_data, text_original_data):
        context, mask = x1[0], x1[2]
        _, x = self.bert(context, attention_mask=mask)
        x = _.unsqueeze(1)  # x: [batch_size, 1, sen_len, H*2]
        x = self.conv1_D(x)  # x: [batch_size, kernel_num, sen_len, 1] // [sen_len, H*2] * [1, H*2] --> [sen_len, 1]
        x = [F.relu(conv(x)) for conv in self.convK_1]  # kernel_size = 3
        x3 = [F.relu(self.conv3(i)) for i in x]
        x4 = [F.relu(self.conv4(i)) for i in x3]

        inception = []
        for i in range(len(x4)):
            res = torch.add(x3[i], x4[i])
            inception.append(res)
        x = [i.squeeze(3) for i in inception]  # x: [batch_size, kernel_num, sen_len - kernel_num + 1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # x[0]: [batch_size, kernel_num]
        x = torch.cat(x, 1)  # x: [batch_size, kernel_num*n] (n = 3)
        x = self.CNN_dropout(x)
        personalized_feature = per_feature(text_vec=x1[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((x, personalized_feature), dim=1)
        logit = self.fc1(outputs)
        logit = self.softmax(logit)
        return logit


# ============================================================
#                       TS_CSW + aware
# ============================================================
class TSCSW_aware(nn.Module):
    def __init__(self, args):
        super(TSCSW_aware, self).__init__()
        self.args = args
        self.args = args
        self.bert = args.model
        D = 768
        C = 2
        Ci = 1
        Co = 128
        Ks = [3, 4, 5]

        self.conv1_D = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(1, D))
        self.convK_1 = nn.ModuleList([nn.Conv2d(Co, Co, (K, 1)) for K in Ks])
        self.conv3 = nn.Conv2d(Co, Co, (3, 1))
        self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1, 0))
        self.CNN_dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks) * Co + 14, 100)
        self.fc2 = nn.Linear(100, C)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, text_data, text_original_data):
        context, mask = x1[0], x1[2]
        _, x = self.bert(context, attention_mask=mask)
        x = _.unsqueeze(1)  # x: [batch_size, 1, sen_len, H*2]
        x = self.conv1_D(x)  # x: [batch_size, kernel_num, sen_len, 1] // [sen_len, H*2] * [1, H*2] --> [sen_len, 1]
        x = [F.relu(conv(x)) for conv in self.convK_1]  # kernel_size = 3
        x3 = [F.relu(self.conv3(i)) for i in x]
        x4 = [F.relu(self.conv4(i)) for i in x3]

        inception = []
        for i in range(len(x4)):
            res = torch.add(x3[i], x4[i])
            inception.append(res)
        x = [i.squeeze(3) for i in inception]  # x: [batch_size, kernel_num, sen_len - kernel_num + 1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # x[0]: [batch_size, kernel_num]
        x = torch.cat(x, 1)  # x: [batch_size, kernel_num*n] (n = 3)
        x = self.CNN_dropout(x)
        personalized_feature = per_feature(text_vec=x1[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((x, personalized_feature), dim=1)
        logit = self.fc1(outputs)
        logit = self.fc2(logit)
        logit = self.softmax(logit)
        return logit


# ============================================================
#                      R_BI_C + aware
# ============================================================
class RBIC_aware(nn.Module):
    def __init__(self, args):
        super(RBIC_aware, self).__init__()
        self.args = args
        self.bert = args.model

        D = 768  # 300
        C = 2
        N = 2
        H = 768
        Ci = 1
        Co = 128  # 128
        Ks =[3, 5, 7]

        self.lstm = nn.LSTM(D, H, num_layers=N, bidirectional=True, batch_first=True, dropout=0.5)
        self.conv1_D = nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(1, 2*H))
        self.convK_1 = nn.ModuleList([nn.Conv2d(Co, Co, (K, 1)) for K in Ks])
        self.conv3 = nn.Conv2d(Co, Co, (3, 1))
        self.conv4 = nn.Conv2d(Co, Co, (3, 1), padding=(1, 0))
        self.CNN_dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co+14, C)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, text_data, text_original_data):
        context, mask = x1[0], x1[2]
        _, x = self.bert(context, attention_mask=mask)
        out, _ = self.lstm(_)  # out: [batch_size, sen_len, H*2]
        x = out.unsqueeze(1)  # x: [batch_size, 1, sen_len, H*2]
        x = self.conv1_D(x)  # x: [batch_size, kernel_num, sen_len, 1] // [sen_len, H*2] * [1, H*2] --> [sen_len, 1]
        x = [F.relu(conv(x)) for conv in self.convK_1]  # kernel_size = 3
        x3 = [F.relu(self.conv3(i)) for i in x]
        x4 = [F.relu(self.conv4(i)) for i in x3]
        inception = []
        for i in range(len(x4)):
            res = torch.add(x3[i], x4[i])
            inception.append(res)
        x = [i.squeeze(3) for i in inception]  # x: [batch_size, kernel_num, sen_len-kernel_num+1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # x[0]: [batch_size, kernel_num]
        x = torch.cat(x, 1)  # x: [batch_size, kernel_num*n] (n = 3)
        x = self.CNN_dropout(x)
        personalized_feature = per_feature(text_vec=x1[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((x, personalized_feature), dim=1)
        logit = self.fc1(outputs)
        logit = self.softmax(logit)
        return logit


# ============================================================
#                           GNN + aware
# ============================================================
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, get_att=False):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.get_att = get_att
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=2)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=2)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj1):
        h = torch.matmul(input, self.W)
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj1 > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        B, M, E = Wh.shape  # (batch_zize, number_nodes, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(M, dim=1)  # (B, M*M, E)
        Wh_repeated_alternating = Wh.repeat(1, M, 1)  # (B, M*M, E)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)  # (B, M*M,2E)
        return all_combinations_matrix.view(B, M, M, 2 * E)


class GAT_aware(nn.Module):
    def __init__(self, args):
        super(GAT_aware, self).__init__()
        self.args = args
        self.bert = args.model
        self.alpha = 0.2
        self.hidden = 768
        self.out_features = 768
        self.dropout = 0.5
        self.dropout1 = nn.Dropout(self.dropout)
        self.n_heads = 1
        self.num_labels = 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.nnLinear = nn.Linear(768, 768)
        self.classifier = nn.Linear(self.out_features+14, self.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.out_att = GATLayer(self.hidden * self.n_heads, self.out_features, dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, x1, text_data, text_original_data, adj):
        context, mask = x1[0], x1[2]
        _, x_input = self.bert(context, attention_mask=mask)
        dim = _[0, :].shape
        dim = dim[0]
        adj = adj[0: dim, 0: dim]
        _ = self.nnLinear(_)
        x = F.elu(self.out_att(_, adj))
        clf_input = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        clf_input = self.dropout1(clf_input)
        personalized_feature = per_feature(text_vec=x1[0], text_data=text_data, text_original_data=text_original_data)
        clf_input = torch.cat((clf_input, personalized_feature), dim=1)
        logits = self.classifier(clf_input)
        logits = self.softmax(logits)
        return logits


# ================================================================
#                        Bert_LSTM_Attn
# ================================================================
class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(lstm, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.LNx = nn.LayerNorm(4 * self.hidden_size)
        self.LNh = nn.LayerNorm(4 * self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4 * self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4 * self.hidden_size, bias=True)

    def forward(self, x):
        def recurrence(xt, hidden):  # enhanced with layer norm
            # input: input to the current cell
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))
            return ht, ct

        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        inputs = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(inputs[t], hidden)
            output.append(hidden[0])
        output = torch.stack(output, 0).transpose(0, 1)
        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(inputs[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


class TC_base(nn.Module):
    def __init__(self,in_features, hidden_dim,  class_num, dropout_rate,bidirectional):
        super(TC_base, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num
        self.hidden_size = 768
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm = lstm(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.attn = Attention(
            embed_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            n_head=1,
            score_function='mlp',
            dropout=self.dropout_prob
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, features, input_ids_len):
        output = self.lstm(features)
        output = self.dropout(output)
        scc, scc1 = self.attn(output,output)
        t = input_ids_len.view(input_ids_len.size(0),1)
        scc_sen = torch.sum(scc,dim=2)
        scc_mean = torch.div(torch.sum(scc,dim=1),t)
        logits = self.classifier(scc_mean)
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )


class BertLstm(nn.Module):
    def __init__(self, args):
        super(BertLstm, self).__init__()
        self.args = args
        self.bert = args.model
        self.fc1 = nn.Linear(768, 2)
        self.hidden_size = 768
        self.num_labels = 2
        self.dropout = nn.Dropout(0.1)
        self.bidirectional = True
        self.embed_size = 768
        self.in_features = 768
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = lstm(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.attn = Attention(
            embed_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            n_head=1,
            score_function='mlp',
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        context, mask = x[0], x[2]
        input_ids_len = torch.sum(context != 0, dim=-1).float()
        outputs = self.bert(context, attention_mask=mask, output_hidden_states=True)
        output = self.lstm(outputs[2][12])
        output = self.dropout(output)
        scc, scc1 = self.attn(output, output)
        t = input_ids_len.view(input_ids_len.size(0), 1)
        scc_mean = torch.div(torch.sum(scc, dim=1), t)
        logits = self.classifier(scc_mean)
        logits = self.softmax(logits)
        return logits


# ================================================================
#                    Bert_LSTM_Attn + aware
# ================================================================
class Attention_aware(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention_aware, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]
        k_len = k.shape[1]
        q_len = q.shape[1]
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class lstm_aware(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(lstm_aware, self).__init__()
        self.input_size = input_size
        if bidirectional:
            self.hidden_size = hidden_size // 2
        else:
            self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.LNx = nn.LayerNorm(4 * self.hidden_size)
        self.LNh = nn.LayerNorm(4 * self.hidden_size)
        self.LNc = nn.LayerNorm(self.hidden_size)
        self.Wx = nn.Linear(in_features=self.input_size, out_features=4 * self.hidden_size, bias=True)
        self.Wh = nn.Linear(in_features=self.hidden_size, out_features=4 * self.hidden_size, bias=True)

    def forward(self, x):
        def recurrence(xt, hidden):  # enhanced with layer norm
            # input: input to the current cell
            htm1, ctm1 = hidden
            gates = self.LNx(self.Wx(xt)) + self.LNh(self.Wh(htm1))
            it, ft, gt, ot = gates.chunk(4, 1)
            it = torch.sigmoid(it)
            ft = torch.sigmoid(ft)
            gt = torch.tanh(gt)
            ot = torch.sigmoid(ot)
            ct = (ft * ctm1) + (it * gt)
            ht = ot * torch.tanh(self.LNc(ct))
            return ht, ct

        output = []
        steps = range(x.size(1))
        hidden = self.init_hidden(x.size(0))
        inputs = x.transpose(0, 1)
        for t in steps:
            hidden = recurrence(inputs[t], hidden)
            output.append(hidden[0])
        output = torch.stack(output, 0).transpose(0, 1)
        if self.bidirectional:
            hidden_b = self.init_hidden(x.size(0))
            output_b = []
            for t in steps[::-1]:
                hidden_b = recurrence(inputs[t], hidden_b)
                output_b.append(hidden_b[0])
            output_b = output_b[::-1]
            output_b = torch.stack(output_b, 0).transpose(0, 1)
            output = torch.cat([output, output_b], dim=-1)
        return output

    def init_hidden(self, bs):
        h_0 = torch.zeros(bs, self.hidden_size).cuda()
        c_0 = torch.zeros(bs, self.hidden_size).cuda()
        return h_0, c_0


class TC_basea(nn.Module):
    def __init__(self,in_features, hidden_dim,  class_num, dropout_rate,bidirectional):
        super(TC_basea, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num
        self.hidden_size = 768
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_prob)
        self.lstm = lstm_aware(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.attn = Attention_aware(
            embed_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            n_head=1,
            score_function='mlp',
            dropout=self.dropout_prob
        )
        self.classifier = nn.Linear(self.hidden_size+12, self.num_labels)

    def forward(self, features, input_ids_len):
        output = self.lstm(features)
        output = self.dropout(output)
        scc, scc1 = self.attn(output,output)
        t = input_ids_len.view(input_ids_len.size(0),1)
        scc_sen = torch.sum(scc, dim=2)
        scc_mean = torch.div(torch.sum(scc,dim=1),t)
        logits = self.classifier(scc_mean)
        return logits

    def extra_repr(self) -> str:
        return 'features {}->{},'.format(
            self.in_features, self.class_num
        )


class BertLstm_aware(nn.Module):
    def __init__(self, args):
        super(BertLstm_aware, self).__init__()
        self.args = args
        self.bert = args.model
        self.fc1 = nn.Linear(768, 2)
        self.hidden_size = 768
        self.num_labels = 2
        self.dropout = nn.Dropout(0.1)
        self.bidirectional = True
        self.embed_size = 768
        self.in_features = 768
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = lstm_aware(
            input_size=self.in_features,
            hidden_size=self.hidden_size,
            bidirectional=True
        )
        self.attn = Attention_aware(
            embed_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            n_head=1,
            score_function='mlp',
        )
        self.classifier = nn.Linear(self.hidden_size+14, self.num_labels)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x, text_data, text_original_data):
        context, mask = x[0], x[2]
        input_ids_len = torch.sum(context != 0, dim=-1).float()
        outputs = self.bert(context, attention_mask=mask, output_hidden_states=True)
        output = self.lstm(outputs[2][12])
        output = self.dropout(output)
        scc, scc1 = self.attn(output, output)
        t = input_ids_len.view(input_ids_len.size(0), 1)
        scc_mean = torch.div(torch.sum(scc, dim=1), t)
        personalized_feature = per_feature(text_vec=x[0], text_data=text_data, text_original_data=text_original_data)
        outputs = torch.cat((scc_mean, personalized_feature), dim=1)
        logits = self.classifier(outputs)
        logits = self.softmax(logits)
        return logits
