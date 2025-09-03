import torch
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument('--policy_model_path', metavar='N',default=None, type=str,
                    help='an integer for the accumulator')
parser.add_argument('--output_model_path', metavar='N',default=None, type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()


current_dir = os.path.dirname(os.path.abspath(args.output_model_path))
os.makedirs(current_dir, exist_ok=True)

def find_first_above_beta(nums,beta):
    try:
        return next(idx for idx, num in enumerate(nums) if num > beta)
    except StopIteration:
        return None


print(args.policy_model_path)
policy=torch.load(args.policy_model_path)
policy_weight=policy['pi.net.0.weight']
pre_weight=policy['pi.net.2.weight'].T@policy['pi.net.4.weight'].T

U, S, Vt = torch.linalg.svd(pre_weight)
S2=[torch.sum(S[:i])/torch.sum(S) for i in range(S.shape[0])]
beta=0.98
S_ratio=[torch.sum(S[:i])/torch.sum(S) for i in range(S.shape[0])]
k=find_first_above_beta(S_ratio,beta)

U_k = U[:, :k]
S_k = S[:k]
V_k = Vt[:k, :]

Sigma_k = torch.diag(S_k)
A_approx = U_k @ Sigma_k @ V_k

policy['pi.net.2.weight']=U_k@torch.sqrt(Sigma_k)
policy["pi.net.2.weight"]=policy["pi.net.2.weight"].T
policy['pi.net.4.weight']=torch.sqrt(Sigma_k)@V_k
policy['pi.net.4.weight']=policy['pi.net.4.weight'].T

torch.save(policy,args.output_model_path)