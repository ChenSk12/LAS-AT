from autoattack import AutoAttack

from TinyImageNet_models import *
from TinyImageNet_utils import *
import os
import argparse
import sys

sys.path.insert(0, '..')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/apdcephfs/share_1290939/jiaxiaojun/tiny-imagenet-200')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='/apdcephfs/private_xiaojunjia/LAS-AT/weights/LAS-AT/TinyImageNet/LAS_AWP/model.pth')
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='./data')

    arguments = parser.parse_args()
    return arguments


args = get_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
logfile1 = os.path.join(args.out_dir, 'log_file1.txt')
logfile2 = os.path.join(args.out_dir, 'log_file2.txt')
if os.path.exists(logfile1):
    os.remove(logfile1)
if os.path.exists(logfile2):
    os.remove(logfile2)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model == "VGG":
    target_model = VGG('VGG19')
elif args.model == "ResNet18":
    target_model = ResNet18()
elif args.model == "PreActResNest18":
    target_model = PreActResNet18()
elif args.model == "WideResNet":
    target_model = WideResNet()

target_model = target_model.to(device)
checkpoint = torch.load(args.model_path)
from collections import OrderedDict
try:
    target_model.load_state_dict(checkpoint)
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    target_model.load_state_dict(new_state_dict, False)

target_model.eval()



train_loader, test_loader =  New_ImageNet_get_loaders_64(args.data_dir, args.batch_size)
epsilon=args.epsilon
epsilon=float(epsilon)/255.
print(epsilon)

AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, target_model, 10, 1, epsilon / std)
AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, target_model, 20, 1, epsilon / std)
AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, target_model, 50, 1, epsilon / std)

AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, target_model, 20, 1)


AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, target_model)

print('AT_models_test_acc:', AT_models_test_acc)
print('AT_pgd_acc_10:', AT_pgd_acc_10)
print('AT_pgd_acc_20:', AT_pgd_acc_20)
print('AT_pgd_acc_50:', AT_pgd_acc_50)
print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)



adversary1 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile1)

#adversary2 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile2)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

adv_complete = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
# adv_complete1 = adversary2.run_standard_evaluation_individual(x_test[:args.n_ex], y_test[:args.n_ex],
#                 bs=args.batch_size)