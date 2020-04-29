import torch
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
tensor1.requires_grad
tensor2.requires_grad
tensor1.requires_grad_()
tensor1.requires_grad
tensor2.requires_grad
print(tensor1.grad)
print(tensor1.grad_fn)
output_tensor = tensor1 * tensor2
output_tensor.requires_grad
print(output_tensor.grad_fn)
print(tensor1.grad_fn)
print(tensor2.grad_fn)
output_tensor = (tensor1 * tensor2).mean()
print(output_tensor.grad_fn)
print(tensor1.grad)
print(output_tensor.grad)
output_tensor.backward()
print(tensor1.grad)
print(tensor2.grad)
print(output_tensor.grad)
new_tensor = tensor1 * 3
print(new_tensor.requires_grad)
new_tensor
with torch.no_grad():
    new_tensor = tensor1 * 3
    print('requires_grad for tensor1 ', tensor1.requires_grad)
    print('requires_grad for tensor2 ', tensor2.requires_grad)
    print('requires_grad for new_tensor ', new_tensor.requires_grad)