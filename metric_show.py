import matplotlib.pyplot as plt
import json

'''
根据训练数据画图监测loss的变化

2020.05.26 -- CaoHuiBin
'''


def cls_metric_show():
    with open('metrics.txt') as metrics:
        lines = metrics.readlines()
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for line in lines:
            temp = line.strip().split(',')
            if temp[0] == 'train':
                train_loss.append(float(temp[1]))
                train_acc.append(float(temp[2]))
            elif temp[0] == 'val':
                val_loss.append(float(temp[1]))
                val_acc.append(float(temp[2]))
        epoch = [x for x in range(len(train_loss))]
        plt.plot(epoch, train_loss, label='train_loss')
        plt.plot(epoch, val_loss, label='val_loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        plt.plot(epoch, train_acc, label='train_acc')
        plt.plot(epoch, val_acc, label='val_acc')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()


def det_metric_show():
    iteration = []
    total_loss = []
    loss_cls = []
    loss_box_reg = []
    loss_rpn_cls = []
    loss_rpn_loc = []
    with open('./metrics/3/metrics.json') as metrics:
        for line in metrics.readlines():
            datas = json.loads(line)
            iteration.append(datas['iteration'])
            total_loss.append(datas['total_loss'])
            loss_cls.append(datas['loss_cls'])
            loss_box_reg.append(datas['loss_box_reg'])
            loss_rpn_cls.append(datas['loss_rpn_cls'])
            loss_rpn_loc.append(datas['loss_rpn_loc'])

    plt.plot(iteration, total_loss, label='total_loss')
    plt.plot(iteration, loss_cls, label='loss_cls')
    plt.plot(iteration, loss_box_reg, label='loss_box_reg')
    plt.plot(iteration, loss_rpn_cls, label='loss_rpn_cls')
    plt.plot(iteration, loss_rpn_loc, label='loss_rpn_loc')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title("loss-table")
    plt.legend()
    plt.show()


def log_ap_show():
    with open('log.txt') as log:
        lines = log.readlines()
        ap_nextline = False
        ap_class_nextline = 0
        ap50 = []
        ap = []
        ap1 = []
        ap2 = []
        ap3 = []
        ap4 = []
        ap5 = []
        ap6 = []
        ap7 = []
        for line in lines:
            if ap_nextline:
                aps = line.strip().split('|')
                ap.append(float(aps[1].strip()))
                ap50.append(float(aps[2].strip()))
                ap_nextline = False
            if ap_class_nextline > 0:
                aps_class = line.strip().split('|')
                if ap_class_nextline == 3:
                    ap1.append(float(aps_class[2].strip()))
                    ap2.append(float(aps_class[4].strip()))
                    ap3.append(float(aps_class[6].strip()))
                elif ap_class_nextline == 2:
                    ap4.append(float(aps_class[2].strip()))
                    ap5.append(float(aps_class[4].strip()))
                    ap6.append(float(aps_class[6].strip()))
                elif ap_class_nextline == 1:
                    ap7.append(float(aps_class[2].strip()))
                ap_class_nextline -= 1
            if line.strip() == '|:------:|:------:|:------:|:------:|:-----:|:-----:|':
                ap_nextline = True
            if line.strip() == '|:-----------|:-------|:-----------|:-------|:-----------|:-------|':
                ap_class_nextline = 3
        plt.plot([x + 1 for x in range(len(ap1))], ap1, label='surprise')
        plt.plot([x + 1 for x in range(len(ap2))], ap2, label='fear')
        plt.plot([x + 1 for x in range(len(ap3))], ap3, label='disgust')
        plt.plot([x + 1 for x in range(len(ap4))], ap4, label='happy')
        plt.plot([x + 1 for x in range(len(ap5))], ap5, label='sad')
        plt.plot([x + 1 for x in range(len(ap6))], ap6, label='angry')
        plt.plot([x + 1 for x in range(len(ap7))], ap7, label='nutural')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        #plt.plot([x + 1 for x in range(len(ap50))], ap50, label='AP50')
        plt.plot([x + 1 for x in range(len(ap))], ap, label='mAP')
        plt.xlabel('epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    log_ap_show()
