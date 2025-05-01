import torch
from torch import nn
import argparse
from logger import logger
import random
import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import time
import numpy as np
import heapq
from transformers import BertModel, BertTokenizer

import data_bert
import network_bert

# =======================================
# =======================================
parser = argparse.ArgumentParser(description='MyBert')

cover_data_name = "../data_cover/A_Overall.txt"
stego_data_name = "../data_stego/13b-32/A_Overall/A_Overall.txt"

parser.add_argument("--neg_filename", type=str, default=cover_data_name)
parser.add_argument("--pos_filename", type=str, default=stego_data_name)
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch [default: False]')
# parser.add_argument('-train-cover-dir', type=str, default=data_name + 'train_cover.txt', help='the path of train cover data. [default: tweets_cover.txt]')
# parser.add_argument('-train-stego-dir', type=str, default=data_name + 'train_stego.txt', help='the path of train stego data. [default: tweets_stego.txt]')
# parser.add_argument('-test-cover-dir', type=str, default=data_name + 'test_cover.txt', help='the path of test cover data. [default: test_cover.txt]')
# parser.add_argument('-test-stego-dir', type=str, default=data_name + 'test_stego.txt', help='the path of test stego data. [default: test_stego.txt]')

# learning
parser.add_argument('-lr', type=float, default=5e-5, help='initial learning rate [default:5e-5]')
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--stop", type=int, default=1000)
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--logdir", type=str, default="./cnnlog")
parser.add_argument("--sentence_num", type=int, default=1007)
parser.add_argument("--rand_seed", type=int, default=0)
parser.add_argument('-hidden-size', type=int, default=768, help='the number of hidden unit [default: 768]')
parser.add_argument('-encode-layer', type=int, default=12, help='the number of encoder layer [default: 12]')
parser.add_argument('-d_model', type=int, default=300, help='the number of encoder layer [default: 12]')

# device
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu [default:False]')
parser.add_argument('-device', type=str, default='cuda', help='device to use for training [default:cuda]')
parser.add_argument('-idx-gpu', type=str, default='0', help='the number of gpu for training [default:0]')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu

# bert_model
args.model = BertModel.from_pretrained('./BERT/bert-base-uncased')
args.tokenizer = BertTokenizer.from_pretrained('./BERT/bert-base-uncased')

log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir + "/cnn_{}.txt".format(os.path.basename(args.neg_filename) + "___" + os.path.basename(args.pos_filename))
random.seed(args.rand_seed)


# =======================================
# =======================================
def main(train_iter, test_iter, model, args):
    
    BATCH_SIZE = args.batch_size
    EMBED_SIZE = args.hidden_size
    CLASS_NUM = 2
    EPOCH = args.epoch
    FILTER_NUM = 128
    FILTER_SIZE = [3, 5, 7]
    DROPOUT_RATE = 0.5
    SAVE_EVERY = 20
    STOP = args.stop
    SENTENCE_NUM = args.sentence_num
    checkpoint_path = json_file = "".join(log_file.split(".txt")) + "_minloss.pth"
    all_var = locals()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class My_loss(nn.Module):
        def __init__(self, args, alpha=[0.5, 0.5], gamma=0, num_classes=2, size_average=True):
            super(My_loss, self).__init__()
            if 'unbalance_c' in args.train_cover_dir:
                alpha = [0.3, 0.7]
                gamma = 5
            if 'unbalance_s' in args.train_cover_dir:
                alpha = [0.7, 0.3]
                gamma = 5
            self.size_average = size_average
            if isinstance(alpha, list):
                assert len(alpha) == num_classes
                logger.info("My_loss: α = {}, γ = {}".format(alpha, gamma))
                self.alpha = torch.Tensor(alpha)
            else:
                assert alpha < 1 
                logger.info(" --- Focal_loss alpha = {}".format(alpha))
                self.alpha = torch.zeros(num_classes)
                self.alpha[0] += alpha
                self.alpha[1:] += (1 - alpha) 
            self.gamma = gamma

        def forward(self, preds_softmax, labels):
            self.alpha = self.alpha.to(preds_softmax.device)
            preds_softmax = preds_softmax.view(-1, preds_softmax.size(-1))
            preds_logsoft = torch.log(preds_softmax)

            preds_softmax = preds_softmax.gather(1, labels.view(-1, 1)) 
            preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
            alpha = self.alpha.gather(0, labels.view(-1))
            loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft) 

            loss = torch.mul(alpha, loss.t())
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
            return loss

    # criteration = nn.CrossEntropyLoss()
    criteration = My_loss(args=args)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    # 	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight': 0.01},
    # 	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight': 0.0}]
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05, t_total=len(train_iter)*args.epoch)
    
    early_stop = 0
    best_acc = 0
    best_test_loss = 1000
    Epoch, epoch_train_loss, epoch_train_Acc, train_correct, train_num, epoch_test_loss, epoch_test_Acc, \
    test_correct, test_num, epoch_test_P, epoch_test_R, epoch_test_F1 = [], [], [], [], [], [], [], [], [], [], [], []
    model.train()

    # =======================================
    #             train and test
    # =======================================
    for epoch in range(args.epoch):

        # -------------- train --------------
        idx1 = -1
        corrects = 0
        train_acc = []
        train_loss = []
        for batch_train in train_iter:
            batch_data, batch_original_data, all_data = [], [], []
            idx1 = idx1 + 1
            text, label = batch_train[0], batch_train[1]

            # for i in range(len(train_iter.batches)):
            # 	all_dat = train_iter.batches[i][7]
            # 	all_data.append(all_dat)

            
            for i in range(text[0].shape[0]):
                current_idx = BATCH_SIZE * idx1 + i 
                current_data = train_iter.batches[current_idx][4]
                batch_data.append(current_data)
                current_original_data = train_iter.batches[current_idx][7]
                batch_original_data.append(current_original_data)

            optimizer.zero_grad()
            # y = model(text)
            y = model(text, batch_data, batch_original_data)
            # y = model(text, batch_data, batch_original_data, adj)
            tr_loss = criteration(y, label)
            corrects += (torch.max(y, 1)[1].view(label.size()).data == label.data).sum()
            tr_loss.backward()
            optimizer.step()
            train_loss.append(tr_loss.item())
            y = y.cpu().detach().numpy()
            train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

        # -------------- test --------------
        test_loss = 0
        test_corrects = 0
        test_acc, test_tp, tfn, tpfn = [], [], [], []
        length_sum = 0
        idx1 = -1

        with torch.no_grad():
            for batch_test in test_iter:
                batch_data, all_data, batch_original_data = [], [], []
                idx1 = idx1 + 1
                text, label = batch_test[0], batch_test[1]

                # for i in range(len(test_iter.batches)):
                # 	all_dat = test_iter.batches[i][4]
                # 	all_data.append(all_dat)
                
                for i in range(text[0].shape[0]):
                    current_idx = BATCH_SIZE * idx1 + i 
                    current_data = test_iter.batches[current_idx][4]
                    batch_data.append(current_data)
                    current_original_data = test_iter.batches[current_idx][7]
                    batch_original_data.append(current_original_data)

                # y = model(text)
                y = model(text, batch_data, batch_original_data)
                # y = model(text, batch_data, batch_original_data, adj)
                loss = criteration(y, label)
                loss = loss.cpu().numpy()
                test_corrects += (torch.max(y, 1)[1].view(label.size()).data == label.data).sum()
                test_loss += loss * len(text)
                length_sum += len(text)
                y = y.cpu().numpy()
                label_pred = np.argmax(y, axis=-1)
                test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
                test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
                tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
                tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]

        test_loss = test_loss / length_sum
        acc = np.mean(test_acc)
        tpsum = np.sum(test_tp)
        test_precision = tpsum / np.sum(tfn)
        test_recall = tpsum / np.sum(tpfn)
        test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision)

        size_train = len(train_iter.batches)
        size_test = len(test_iter.batches)

        Epoch.append(epoch + 1)  # epoch
        epoch_train_loss.append(np.mean(train_loss))  # train_loss
        epoch_train_Acc.append(np.mean(train_acc))  # train_acc
        train_correct.append(corrects)  # train_correct_num
        train_num.append(size_train)  # train_total_num
        epoch_test_loss.append(test_loss)  # test_loss
        epoch_test_Acc.append(acc)  # test_acc
        test_correct.append(test_corrects)  # test_correct_num
        test_num.append(size_test)  # test_total_num
        epoch_test_P.append(test_precision)  # test_P
        epoch_test_R.append(test_recall)  # test_R
        epoch_test_F1.append(test_Fscore)  # test_F1

        logger.info("Epoch: {:d} || Train: Loss {:.4f}, Acc {:.4f}({}/{}) || Test: Loss {:.4f}, Acc {:.4f}({}/{}), P {:.4f}, R {:.4f}, F1 {:.4f}"
                    .format(epoch + 1, np.mean(train_loss), np.mean(train_acc), corrects, size_train, test_loss, acc, test_corrects, size_test, test_precision, test_recall, test_Fscore))

        # -------------- save --------------
        if test_loss < best_test_loss:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "loss": test_loss, "acc": np.mean(test_acc)}
            torch.save(state, checkpoint_path)
            best_test_loss = test_loss
            best_acc = max(epoch_test_Acc)
            precision = test_precision
            recall = test_recall
            F1 = test_Fscore
            early_stop = 0
        else:
            early_stop += 1

        # -------------- stop --------------
        if early_stop >= STOP:
            logger.info('Loss: {:.4f}, Acc: {:.4f}, P {:.4f}, R {:.4f}, F1 {:.4f}'.format(best_test_loss, best_acc, precision, recall, F1))
            return best_acc, precision, recall, F1

    
    best10_train_acc = list(map(epoch_train_Acc.index, heapq.nlargest(4, epoch_train_Acc)))
    best10_train_ACC = []
    for idx in list(best10_train_acc):
        best10_train_ACC.append(epoch_train_Acc[idx])

    best_test_acc1 = [epoch_test_Acc[best10_train_acc[0]], epoch_test_Acc[best10_train_acc[1]], epoch_test_Acc[best10_train_acc[2]], epoch_test_Acc[best10_train_acc[3]]]
    best_test_acc = max(best_test_acc1)
    best_test_acc_index = best_test_acc1.index(best_test_acc)
    best_test_pre = epoch_test_P[best10_train_acc[best_test_acc_index]]
    best_test_recall = epoch_test_R[best10_train_acc[best_test_acc_index]]
    best_test_F1 = epoch_test_F1[best10_train_acc[best_test_acc_index]]

    logger.info("=========================================")
    logger.info("The performance of the current time are:")
    for i in range(4):
        logger.info("Epoch: {:d} || Train: Loss {:.4f}, Acc {:.4f}({}/{}) || Test: Loss {:.4f}, Acc {:.4f}({}/{}), P {:.4f}, R {:.4f}, F1 {:.4f}"
                    .format(Epoch[i], epoch_train_loss[i], epoch_train_Acc[i], train_correct[i], train_num[i], epoch_test_loss[i], epoch_test_Acc[i], test_correct[i], test_num[i], epoch_test_P[i],
                            epoch_test_R[i], epoch_test_F1[i]))

    logger.info("The best performance of the current time is: Acc {:.4f}, P {:.4f}, R {:.4f}, F1 {:.4f}".format(best_test_acc, best_test_pre, best_test_recall, best_test_F1))
    return best_test_acc, best_test_pre, best_test_recall, best_test_F1


def select_random_samples(dataset, num_samples):
    return random.sample(dataset, num_samples)


if __name__ == '__main__':
    acc, preci, recall, F1 = [], [], [], []

    # =======================================
    #                   load
    # =======================================
    # -------------- load data --------------
    print('\nLoading data...')
    train_data, test_data = data_bert.build_dataset(args)

    if args.sentence_num < len(train_data):
        train_data = select_random_samples(train_data, args.sentence_num)
    if args.sentence_num < len(test_data):
        test_data = select_random_samples(test_data, args.sentence_num)

    train_iter = data_bert.build_iterator(train_data, args)
    test_iter = data_bert.build_iterator(test_data, args)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    del args.no_cuda

    #  -------------- load model and get params --------------
    # model = network_bert.BertLstm(args)
    # model = network_bert.Bert_CNN(args)

    model = network_bert.FCN_aware(args)
    # model = network_bert.TSBiRNN_aware(args)
    # model = network_bert.LSCNN_aware(args)
    # model = network_bert.RBIC_aware(args)
    # model = network_bert.TSCSW_aware(args)
    # model = network_bert.BiLSTMDense_aware(args)
    # model = network_bert.GAT_aware(args)
    # model = network_bert.BertLstm_aware(args)
    if args.cuda:
        torch.device(args.device)
        model = model.cuda()


    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


    print("The total and trainable parameters of this network is:", get_parameter_number(net=model))

    # =======================================
    # =======================================
    acc, preci, recall, F1 = [], [], [], []
    for i in range(5):
        random.seed(42)
        loss = 0
        logger.info("--------------------------------------------------------------------------------------")
        logger.info("i = {:.0f}".format(i))
        # logger.info("name = {}".format(name))
        start = time.time()
        random.seed(i)
        index = main(train_iter, test_iter, model, args)  # main
        acc.append(index[0])
        preci.append(index[1])
        recall.append(index[2])
        F1.append(index[3])
        end = time.time()
        logger.info('time: {:.4f}'.format(end - start))
    logger.info("--------------------------------------------------------------------------------------")

    acc_mean, acc_std, pre_mean, pre_std = np.mean(acc), np.std(acc), np.mean(preci), np.std(preci)
    recall_mean, recall_std, f1_mean, f1_std = np.mean(recall), np.std(recall), np.mean(F1), np.std(F1)

    logger.info("=============================================================================================")
    logger.info("Final: Acc {:.4f}±{:.4f}, P {:.4f}±{:.4f}, R {:.4f}±{:.4f}, F1 {:.4f}±{:.4f}"
                .format(acc_mean, acc_std, pre_mean, pre_std, recall_mean, recall_std, f1_mean, f1_std))
