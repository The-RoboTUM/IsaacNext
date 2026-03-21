import torch

x = torch.tensor([-1.0])
y = torch.tensor([0.0])
x.requires_grad_()
y.requires_grad_()
atan2_result = torch.atan2(y, x)

# gradient check
atan2_result.sum().backward()
print("atan2 results:", atan2_result)
print("Gradients dx:", x.grad)
print("Gradients dy:", y.grad)
