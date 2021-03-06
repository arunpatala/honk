from collections import ChainMap
import argparse
import os
import random
import sys
import csv
import pdb

from torch.utils.data import ConcatDataset

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from tensorboardX import SummaryWriter

from . import model as mod
import torch.nn.functional as F

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).sum() / batch_size
    loss = loss.cpu().data.numpy()[0]
    #print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def submit(config, model=None, test_loader=None, output="sub.csv", log_dir=None):
    if not test_loader:
        config['test_pct'] = 100
        config['train_pct'] = 0
        config['dev_pct'] = 0
        config["group_speakers_by_id"] = False
        config["silence_prob"] = 0
        config["data_folder"] = config["test_folder"]
        output = os.path.join(config["log_dir"], output)
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set.sorted(), batch_size=config["batch_size"], shuffle=False)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    results = []
    total = 0

    print(len(test_set), len(test_set.audio_files))
    commands = ["silence", "unknown"]
    commands.extend(config["wanted_words"])

    with open(output, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['fname', 'label', 'score'])
        for model_in, labels in tqdm(test_loader):
            model_in = Variable(model_in, requires_grad=False)
            if not config["no_cuda"]:
                model_in = model_in.cuda()
            scores = model(model_in)
            predictions = F.softmax(scores.squeeze(0).cpu()).data
            #results.append(predictions) 
            scores, results = torch.max(predictions, 1)
            for i in range(len(results)):
                 csvwriter.writerow([test_set.audio_files[total+i].split("/")[-1], commands[results[i]], scores[i]] + predictions[i].numpy().tolist() )
            total += model_in.size(0)


def evaluate(config, model=None, test_loader=None, epoch=0, log_dir=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    #model = nn.Sequential(*list(model.children())[:-1])
    print(model)

    model.eval()
    total = 0
    all_scores = []
    all_fcs = []
    all_labels = []
    for model_in, labels in tqdm(test_loader):
        #print(model_in.size())
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        predictions = F.softmax(scores.squeeze(0).cpu()).data.numpy()
        all_scores.append(predictions)
        all_fcs.append(model.fc.squeeze(0).cpu().data.numpy())
        all_labels.append(labels.cpu().numpy())
        total += model_in.size(0)
    all_scores = np.concatenate(all_scores,axis=0)
    all_preds = np.argmax(all_scores, axis=1)
    all_fcs = np.concatenate(all_fcs,axis=0)    
    all_labels = np.concatenate(all_labels, axis=0)
    
    all_sl = np.concatenate([np.expand_dims(all_labels,axis=-1), np.expand_dims(all_preds,axis=-1), all_scores, all_fcs], axis=-1)
    np.savetxt(log_dir+'/fc{}.csv'.format(epoch), all_sl, delimiter=',', fmt='%.3f')
        
    #print("final test loss: {}".format(np.mean(losses)))
    #print("final test accuracy: {}".format(sum(results) / total))
    #return sum(results) / total


def evaluate2(config, model=None, test_loader=None, epoch=0, log_dir=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    #model = nn.Sequential(*list(model.children())[:-1])
    #print(model)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    losses = []
    all_scores = []
    all_labels = []
    for model_in, labels in tqdm(test_loader):
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        predictions = F.softmax(scores.squeeze(0).cpu()).data.numpy()
        all_scores.append(predictions)
        all_labels.append(labels.cpu().numpy())
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        losses.append(loss.cpu().data.numpy()[0])
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
    all_scores = np.concatenate(all_scores,axis=0)
    #print("all_scores", all_scores.shape)

    all_labels = np.concatenate(all_labels, axis=0)
    #print("all_labels", all_labels.shape)
    all_preds = np.argmax(all_scores, axis=1)

    all_sl = np.concatenate([np.expand_dims(all_labels,axis=-1), np.expand_dims(all_preds,axis=-1), all_scores], axis=-1)
    #print("all_sl", all_sl.shape)
    np.savetxt(log_dir+'/test{}.csv'.format(epoch), all_sl, delimiter=',', fmt='%.3f')
        
    print("final test loss: {}".format(np.mean(losses)))
    print("final test accuracy: {}".format(sum(results) / total))
    return sum(results) / total

def train(config):
    log_dir = config["log_dir"]
    if log_dir!="tmp" : os.mkdir(log_dir)
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)


    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    if config["psuedo"]:
        train_set1, dev_set1, test_set1 = mod.SpeechDataset.splits(config)
        dev_set1.set_predict(model, config)
        test_set1.set_predict(model, config)
        train_set = ConcatDataset([train_set, dev_set1, test_set1])

    print("len", len(train_set), len(dev_set), len(test_set))
    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=False)
    step_no = 0
    writer = SummaryWriter(log_dir)
    for epoch_idx in range(config["n_epochs"]):
        print("epoch", epoch_idx, config["n_epochs"])
        writer.add_scalar('data/iter', epoch_idx, epoch_idx)
        accs = []
        losses = []
        all_scores = []
        all_labels = []
        #print(model)
        for batch_idx, (model_in, labels) in enumerate(tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            predictions = F.softmax(scores.squeeze(0).cpu()).data.numpy()
            all_scores.append(predictions)
            #all_scores.append(scores.cpu().data.numpy())
            all_labels.append(labels.cpu().numpy())
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            accs.append(print_eval("train step #{}".format(step_no), scores, labels, loss))
            losses.append(loss.cpu().data.numpy()[0])
            #print(type( float(np.mean(losses) )) )
        all_scores = np.concatenate(all_scores,axis=0)
        #print("all_scores", all_scores.shape)

        all_labels = np.concatenate(all_labels, axis=0)
        print("train counts", np.bincount(all_labels))
        #print("all_labels", all_labels.shape)
        all_preds = np.argmax(all_scores, axis=1)

        all_sl = np.concatenate([np.expand_dims(all_labels,axis=-1), np.expand_dims(all_preds,axis=-1), all_scores], axis=-1)
        #print("all_sl", all_sl.shape)
        np.savetxt(log_dir+'/train.csv', all_sl, delimiter=',', fmt='%.3f')
            #print(all_sl)
            #print(all_sl.shape)
        tacc = float(np.mean(accs))
        tloss = float(np.mean(losses))
        print("train accuracy: {}".format(tacc))
        print("train loss: {}".format(tloss))
        writer.add_scalar('data/tacc', tacc, epoch_idx)
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            losses = []
            all_scores = []
            all_labels = []
            for model_in, labels in tqdm(dev_loader):
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                predictions = F.softmax(scores.squeeze(0).cpu()).data.numpy()
                all_scores.append(predictions)
                #all_scores.append(scores.cpu().data.numpy())
                all_labels.append(labels.cpu().numpy())
            
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.cpu().data.numpy()[0]
                accs.append(print_eval("dev", scores, labels, loss))
                losses.append(loss.cpu().data.numpy()[0])
            vacc = float(np.mean(accs))
            vloss= float(np.mean(losses))
            all_scores = np.concatenate(all_scores,axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            print("val counts", np.bincount(all_labels))
            all_preds = np.argmax(all_scores, axis=1)
            all_sl = np.concatenate([np.expand_dims(all_labels,axis=-1), np.expand_dims(all_preds, axis=-1), all_scores], axis=-1)
            np.savetxt(log_dir+'/val{}.csv'.format(epoch_idx), all_sl, delimiter=',', fmt='%.3f')
            
            print("final dev accuracy: {}".format(vacc))
            print("final dev loss: {}".format(vloss))
            writer.add_scalar('data/vacc', vacc, epoch_idx)
            if vacc > max_acc:
                print("saving best model...")
                max_acc = vacc
            model.save(log_dir+"/BEST_"+config["output_file"])
            model.save(log_dir+"/epoch_"+str(epoch_idx)+"_"+config["output_file"])
            ttacc = evaluate(config, model, test_loader, epoch=epoch_idx, log_dir=log_dir)
            writer.add_scalar('data/ttacc', ttacc, epoch_idx)
            writer.add_scalars('data/acc', {"tacc": tacc, "vacc": vacc, "ttacc": ttacc}, epoch_idx)
            #print(tloss, vloss)
            #print(type(vloss), type(tloss))
            writer.add_scalar('data/vloss', vloss, epoch_idx)
            writer.add_scalar('data/tloss', tloss, epoch_idx)
            
            writer.add_scalars('data/loss', {"tloss": tloss, "vloss": vloss}, epoch_idx)
            writer.export_scalars_to_json(log_dir+"/all_scalars.json")

def main():
    output_file = "model.pt"
    parser = argparse.ArgumentParser()
    print(parser)
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    parser.add_argument("--mode", choices=["train", "evaluate", "submit"], default="train", type=str)
    parser.add_argument("--psuedo",  default=False, type=bool)
    config, _ = parser.parse_known_args()
    print(config)

    global_config = dict(psuedo= config.psuedo, test_folder="test", mode=config.mode, log_dir="tmp", no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768, momentum=0.9, weight_decay=0.00001)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    #parser.add_argument("--mode", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    print(config)
    print("n_epochs", config["n_epochs"])
    print("seed", config["seed"])
    print("data_folder", config["data_folder"])
    print("gpu_no", config["gpu_no"])
    print("mode", config["mode"])
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "evaluate":
        log_dir = config["log_dir"]
        train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
        model = config["model_class"](config)
        #print(config["input_file"])
        if config["input_file"]:
            model.load(config["input_file"])
        else: model.load(log_dir+"/BEST_model.pt")
        if not config["no_cuda"]:
            torch.cuda.set_device(config["gpu_no"])
            model.cuda()
        #print(a+b)
        loader = data.DataLoader(dev_set, batch_size=min(len(test_set), 16), shuffle=False)
        evaluate(config, model, loader, epoch="deveval", log_dir=log_dir)
        #evaluate(config)
    elif config["mode"] == "submit":
        log_dir = config["log_dir"]
        if not config["input_file"]:
            config['input_file'] = (log_dir+"/BEST_model.pt")
        submit(config, output="submit.csv", log_dir=log_dir)

if __name__ == "__main__":
    main()
