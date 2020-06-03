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


if __name__ == '__main__':
    cls_metric_show()