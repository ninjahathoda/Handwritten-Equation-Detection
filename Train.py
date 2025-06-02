import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Densenet_torchvision import densenet121
from Attention_RNN import AttnDecoderRNN
import random
import matplotlib.pyplot as plt
from PIL import Image

# Compute the WER loss
def cmp_result(label, rec):
    dist_mat = numpy.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = range(len(rec) + 1)
    dist_mat[:, 0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i, j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

# Load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

# Custom dataset class
class custom_dset(data.Dataset):
    def __init__(self, train, train_label, batch_size):
        self.train = train
        self.train_label = train_label
        self.batch_size = batch_size

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)
        size = train_setting.size()
        train_setting = train_setting.view(1, size[2], size[3])
        label_setting = label_setting.view(-1)
        return train_setting, label_setting

    def __len__(self):
        return len(self.train)

# Collate function for padding images in batch
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0]) + 1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1, img_size_h, img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s * 255.0
        img_mask_sub = torch.cat((ii, img_mask_sub_s), dim=0)
        padding_h = aa1 - img_size_h
        padding_w = bb1 - img_size_w
        m = torch.nn.ZeroPad2d((0, padding_w, 0, padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k == 0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask, img_mask_sub_padding), dim=0)
        k = k + 1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0, max_len - ii1_len, 0, 0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding, ii1_padding), dim=0)
        k1 = k1 + 1

    img_padding_mask = img_padding_mask / 255.0
    return img_padding_mask, label_padding

# Training function
def my_train(target_length, attn_decoder1, output_highfeature, output_area, y, criterion, encoder_optimizer1, decoder_optimizer1, x_mean, dense_input, h_mask, w_mask, gpu, decoder_input, decoder_hidden, attention_sum, decoder_attention):
    loss = 0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    flag_z = [0] * batch_size

    if use_teacher_forcing:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention, attention_sum = attn_decoder1(decoder_input,
                                                                                           decoder_hidden,
                                                                                           output_highfeature,
                                                                                           output_area,
                                                                                           attention_sum,
                                                                                           decoder_attention,
                                                                                           dense_input, batch_size, h_mask, w_mask, gpu)
            y = y.unsqueeze(0)
            for i in range(batch_size):
                if int(y[0][i][di]) == 0:
                    flag_z[i] = flag_z[i] + 1
                    if flag_z[i] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[i], y[:, i, di])
                else:
                    loss += criterion(decoder_output[i], y[:, i, di])

            if int(y[0][0][di]) == 0:
                break
            decoder_input = y[:, :, di]
            decoder_input = decoder_input.squeeze(0)
            y = y.squeeze(0)

        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

    else:
        encoder_optimizer1.zero_grad()
        decoder_optimizer1.zero_grad()
        my_num = 0
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention, attention_sum = attn_decoder1(decoder_input, decoder_hidden,
                                                                                          output_highfeature, output_area,
                                                                                          attention_sum, decoder_attention, dense_input, batch_size,
                                                                                          h_mask, w_mask, gpu)
            topv, topi = torch.max(decoder_output, 2)
            decoder_input = topi
            decoder_input = decoder_input.view(batch_size)
            y = y.unsqueeze(0)

            for k in range(batch_size):
                if int(y[0][k][di]) == 0:
                    flag_z[k] = flag_z[k] + 1
                    if flag_z[k] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[k], y[:, k, di])
                else:
                    loss += criterion(decoder_output[k], y[:, k, di])

            y = y.squeeze(0)
        loss.backward()
        encoder_optimizer1.step()
        decoder_optimizer1.step()
        return loss.item()

def imresize(im, sz):
    pil_im = Image.fromarray(im)
    return numpy.array(pil_im.resize(sz))

if __name__ == '__main__':
    # Dynamic GPU/CPU handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    print(f"Using device: {device}, Available GPUs: {gpu}")

    datasets = ['offline-train.pkl', 'train_caption.txt']
    valid_datasets = ['offline-test.pkl', 'test_caption.txt']
    dictionaries = ['dictionary.txt']
    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000
    batch_size = 6
    batch_size_t = 6
    maxlen = 48
    maxImagesize = 100000
    hidden_size = 256
    teacher_forcing_ratio = 1
    lr_rate = 0.0001
    flag = 0
    exprate = 0

    worddicts = load_dict(dictionaries[0])
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    train, train_label = dataIterator(
        datasets[0], datasets[1], worddicts, batch_size=1,
        batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize
    )
    len_train = len(train)

    test, test_label = dataIterator(
        valid_datasets[0], valid_datasets[1], worddicts, batch_size=1,
        batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize
    )
    len_test = len(test)

    off_image_train = custom_dset(train, train_label, batch_size)
    off_image_test = custom_dset(test, test_label, batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset=off_image_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Windows
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=off_image_test,
        batch_size=batch_size_t,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Windows
    )

    encoder = densenet121()
    pthfile = r'densenet121-a639ec97.pt'
    pretrained_dict = torch.load(pthfile)
    encoder_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    encoder_dict.update(pretrained_dict)
    encoder.load_state_dict(encoder_dict)

    attn_decoder1 = AttnDecoderRNN(hidden_size, 112, dropout_p=0.5)

    # Move models to appropriate device
    encoder = encoder.to(device)
    attn_decoder1 = attn_decoder1.to(device)
    if gpu:
        encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
        attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)

    criterion = nn.NLLLoss()
    decoder_input_init = torch.LongTensor([111] * batch_size).to(device)
    decoder_hidden_init = torch.randn(batch_size, 1, hidden_size).to(device)
    nn.init.xavier_uniform_(decoder_hidden_init)

    for epoch in range(200):
        encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate, momentum=0.9)
        decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate, momentum=0.9)
        running_loss = 0
        whole_loss = 0
        encoder.train(mode=True)
        attn_decoder1.train(mode=True)

        for step, (x, y) in enumerate(train_loader):
            if x.size()[0] < batch_size:
                break
            h_mask = []
            w_mask = []
            for i in x:
                size_mask = i[1].size()
                s_w = str(i[1][0])
                s_h = str(i[1][:, 1])
                w = s_w.count('1')
                h = s_h.count('1')
                h_comp = int(h / 16) + 1
                w_comp = int(w / 16) + 1
                h_mask.append(h_comp)
                w_mask.append(w_comp)

            x = x.to(device)
            y = y.to(device)
            output_highfeature = encoder(x)
            x_mean = []
            for i in output_highfeature:
                x_mean.append(float(torch.mean(i)))

            for i in range(batch_size):
                decoder_hidden_init[i] = decoder_hidden_init[i] * x_mean[i]
                decoder_hidden_init[i] = torch.tanh(decoder_hidden_init[i])

            output_area1 = output_highfeature.size()
            output_area = output_area1[3]
            dense_input = output_area1[2]
            target_length = y.size()[1]
            attention_sum_init = torch.zeros(batch_size, 1, dense_input, output_area).to(device)
            decoder_attention_init = torch.zeros(batch_size, 1, dense_input, output_area).to(device)

            running_loss += my_train(target_length, attn_decoder1, output_highfeature,
                                    output_area, y, criterion, encoder_optimizer1, decoder_optimizer1, x_mean, dense_input, h_mask, w_mask, gpu,
                                    decoder_input_init, decoder_hidden_init, attention_sum_init, decoder_attention_init)

            if step % 20 == 19:
                pre = ((step + 1) / len_train) * 100 * batch_size
                whole_loss += running_loss
                running_loss = running_loss / (batch_size * 20)
                print('epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' % (epoch, lr_rate, teacher_forcing_ratio, batch_size, pre, running_loss))
                running_loss = 0

        loss_all_out = whole_loss / len_train
        print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))

        total_dist = 0
        total_label = 0
        total_line = 0
        total_line_rec = 0
        whole_loss_t = 0
        encoder.eval()
        attn_decoder1.eval()
        print('Now, begin testing!!')

        for step_t, (x_t, y_t) in enumerate(test_loader):
            x_real_high = x_t.size()[2]
            x_real_width = x_t.size()[3]
            if x_t.size()[0] < batch_size_t:
                break
            print('testing for %.3f%%' % (step_t * 100 * batch_size_t / len_test), end='\r')
            h_mask_t = []
            w_mask_t = []
            for i in x_t:
                size_mask_t = i[1].size()
                s_w_t = str(i[1][0])
                s_h_t = str(i[1][:, 1])
                w_t = s_w_t.count('1')
                h_t = s_h_t.count('1')
                h_comp_t = int(h_t / 16) + 1
                w_comp_t = int(w_t / 16) + 1
                h_mask_t.append(h_comp_t)
                w_mask_t.append(w_comp_t)

            x_t = x_t.to(device)
            y_t = y_t.to(device)
            output_highfeature_t = encoder(x_t)
            x_mean_t = torch.mean(output_highfeature_t)
            x_mean_t = float(x_mean_t)
            output_area_t1 = output_highfeature_t.size()
            output_area_t = output_area_t1[3]
            dense_input = output_area_t1[2]

            decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
            decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).to(device)
            nn.init.xavier_uniform_(decoder_hidden_t)

            x_mean_t = []
            for i in output_highfeature_t:
                x_mean_t.append(float(torch.mean(i)))

            for i in range(batch_size_t):
                decoder_hidden_t[i] = decoder_hidden_t[i] * x_mean_t[i]
                decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

            prediction = torch.zeros(batch_size_t, maxlen)
            prediction_sub = []
            label_sub = []
            decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).to(device)
            attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t).to(device)
            flag_z_t = [0] * batch_size_t
            loss_t = 0
            m = torch.nn.ZeroPad2d((0, maxlen - y_t.size()[1], 0, 0))
            y_t = m(y_t)
            for i in range(maxlen):
                decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                                    decoder_hidden_t,
                                                                                                    output_highfeature_t,
                                                                                                    output_area_t,
                                                                                                    attention_sum_t,
                                                                                                    decoder_attention_t, dense_input, batch_size_t, h_mask_t, w_mask_t, gpu)
                topv, topi = torch.max(decoder_output, 2)
                if torch.sum(topi) == 0:
                    break
                decoder_input_t = topi
                decoder_input_t = decoder_input_t.view(batch_size_t)
                prediction[:, i] = decoder_input_t

            for i in range(batch_size_t):
                for j in range(maxlen):
                    if int(prediction[i][j]) == 0:
                        break
                    else:
                        prediction_sub.append(int(prediction[i][j]))
                if len(prediction_sub) < maxlen:
                    prediction_sub.append(0)

                for k in range(y_t.size()[1]):
                    if int(y_t[i][k]) == 0:
                        break
                    else:
                        label_sub.append(int(y_t[i][k]))
                label_sub.append(0)

                dist, llen = cmp_result(label_sub, prediction_sub)
                total_dist += dist
                total_label += llen
                total_line += 1
                if dist == 0:
                    total_line_rec = total_line_rec + 1

                label_sub = []
                prediction_sub = []

        print('total_line_rec is', total_line_rec)
        wer = float(total_dist) / total_label
        sacc = float(total_line_rec) / total_line
        print('wer is %.5f' % (wer))
        print('sacc is %.5f ' % (sacc))

        if (sacc > exprate):
            exprate = sacc
            print(exprate)
            print("saving the model....")
            print('encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' % (lr_rate))
            torch.save(encoder.state_dict(), 'model/encoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' % (lr_rate))
            torch.save(attn_decoder1.state_dict(), 'model/attn_decoder_lr%.5f_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl' % (lr_rate))
            print("done")
            flag = 0
        else:
            flag = flag + 1
            print('the best is %f' % (exprate))
            print('the loss is bigger than before, so do not save the model')

        if flag == 10:
            lr_rate = lr_rate * 0.1
            flag = 0