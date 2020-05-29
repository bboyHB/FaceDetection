import matplotlib.pyplot as plt
import json

'''
根据训练数据画图监测loss的变化

2020.05.26 -- CaoHuiBin
'''

iteration = []
total_loss = []
loss_cls = []
loss_box_reg = []
loss_rpn_cls = []
loss_rpn_loc = []
with open('./metrics/2/metrics.json') as metrics:
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


