import json

path = '/data4/jyh/hv2/logs/ulsam/01/stats.json'

with open(path,'r') as fp:
    hv_mes = []
    np_dice = []
    np_acc = []
    tp_dice_0 = []
    tp_dice_1 = []
    tp_dice_2 = []
    tp_dice_3 = []
    tp_dice_4 = []
    train_overall = []

    data = json.load(fp)
    print(data['1'].keys())
    for i in range(100):
        hv = data[str(i+1)]['valid-hv_mse']
        np = data[str(i+1)]['valid-np_dice']
        acc = data[str(i+1)]['valid-np_acc']
        dice_0 = data[str(i+1)]['valid-tp_dice_0']
        dice_1 = data[str(i + 1)]['valid-tp_dice_1']
        dice_2 = data[str(i + 1)]['valid-tp_dice_2']
        dice_3 = data[str(i + 1)]['valid-tp_dice_3']
        dice_4 = data[str(i + 1)]['valid-tp_dice_4']
        loss = data[str(i + 1)]['train-overall_loss']

        hv_mes.append(hv)
        np_dice.append(np)
        np_acc.append(acc)
        tp_dice_0.append(dice_0)
        tp_dice_1.append(dice_1)
        tp_dice_2.append(dice_2)
        tp_dice_3.append(dice_3)
        tp_dice_4.append(dice_4)
        train_overall.append(loss)



    print('-----------hv-mes------------')
    print(hv_mes.index(max(hv_mes)) + 1)
    print(max(hv_mes))
    print()
    print('-----------np-dice------------')
    print(np_dice.index(max(np_dice)) + 1)
    print(max(np_dice))
    print()
    print('-----------np_acc------------')
    print(np_acc.index(max(np_acc))+1)
    print(max(np_acc))
    print()
    print('-----------max_dice_0------------')
    print(tp_dice_0.index(max(tp_dice_0))+1)
    print(max(tp_dice_0))
    print()
    print('-----------max_dice_1------------')
    print(tp_dice_1.index(max(tp_dice_1)) + 1)
    print(max(tp_dice_1))
    print()
    print('-----------max_dice_2------------')
    print(tp_dice_2.index(max(tp_dice_2)) + 1)
    print(max(tp_dice_2))
    print()
    print('-----------max_dice_3------------')
    print(tp_dice_3.index(max(tp_dice_3)) + 1)
    print(max(tp_dice_3))
    print()
    print('-----------max_dice_4------------')
    print(tp_dice_4.index(max(tp_dice_4)) + 1)
    print(max(tp_dice_4))
    print()
    print('-----------loss------------')
    print(train_overall.index(min(train_overall)) + 1)
    print(min(train_overall))
    print()
