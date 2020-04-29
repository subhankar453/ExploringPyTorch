import torch
print(torch.__version__)
torch.get_default_dtype()
torch.get_num_threads()
torch.set_default_dtype(torch.float64)
torch.get_default_dtype()
tensor_arr = torch.Tensor([[1, 2, 3], [4, 5, 6]])
torch.is_tensor(tensor_arr)
torch.numel(tensor_arr)  #Gives the number of elements in the tensor
tensor_uninitialized = torch.Tensor(2, 2)
tensor_uninitialized
tensor_initialized = torch.rand(2, 2)
tensor_initialized
temsor_cpu_int = torch.Tensor([5, 3]).type(torch.IntTensor)
tensor_cuda_int = torch.Tensor([5, 3]).type(torch.cuda.IntTensor) 
torch_cpu_short = torch.ShortTensor([1.0, 2.0, 3.0])
tensor_cuda_short = torch.cuda.ShortTensor([1.0, 2.0, 3.0]) #cuda commands create tensor in GPU
tensor_float = torch.cuda.HalfTensor([1.0, 2.0, 3.0]) #takes half the memory
torch_cpu_fill = torch.full((2, 6), fill_value = 10)
torch_cuda_fill = torch.full((2, 6), fill_value = 5).type(torch.cuda.FloatTensor)
tensor_of_ones_cpu = torch.ones([2, 4], dtype = torch.int32)
tensor_of_ones_cuda = torch.ones([2, 4], dtype = torch.int32).type(torch.cuda.IntTensor)
tensor_of_zeros__cpu = torch.zeros_like(tensor_of_ones_cpu)
tensor_of_zeros_gpu = torch.zeros_like(tensor_of_ones_cuda)
tensor_eye_cpu = torch.eye(5)
tensor_eye_gpu = torch.eye(5).type(torch.cuda.FloatTensor)
non_zero_indices = torch.nonzero(tensor_eye_cpu)
non_zero_indices = torch.nonzero(tensor_of_ones_cuda)
i = torch.tensor([[0, 1, 1],
                 [2, 2, 0]])
v = torch.tensor([3, 4, 5], dtype = torch.float32)
sparse_tensor = torch.sparse_coo_tensor(i, v, [2, 5])
sparse_tensor.data

#Simple operations on Tensors
initial_tensor = torch.rand(2, 3)
initial_tensor.fill_(10)
new_tensor = initial_tensor.add(5)
initial_tensor.add_(8)
new_tensor.sqrt_()
x = torch.linspace(start = 0.1, end = 10, steps = 15)
tensor_chunk = torch.chunk(x, 3, 0)
tensor_1 = tensor_chunk[0]
tensor_2 = tensor_chunk[1]
tensor_3 = torch.tensor([3.0, 4.0, 5.0])
torch.cat((tensor_1, tensor_2, tensor_3), 0)
random_tensor = torch.Tensor([[20, 8, 40], [5, 43, 6], [78, 54, 6]])
random_tensor[0, 1]
random_tensor[1:, 1:]
random_tensor.size()
resized_tensor = random_tensor.view(9)
resized_tensor.size()
random_tensor[2, 2] = 200.0
resized_tensor

#Elementwise and Matrix Operations
tensor_unsqueeze = torch.unsqueeze(random_tensor, 2)
tensor_unsqueeze.size()
tensor_transpose = torch.transpose(initial_tensor, 0, 1)
sorted_tensor, sorted_indices = torch.sort(random_tensor)
tensor_float = torch.FloatTensor([-1.1, -2.2, -3.3])
tensor_abs = torch.abs(tensor_float)
rand1 = torch.abs(torch.rand(2,3))
rand2 = torch.abs(torch.rand(2,3))
add1 = rand1 + rand2
tensor_just = torch.Tensor([[1, 2, 3], [-1, -2, -3]])
tensor_div = torch.div(tensor_just, tensor_just + 0.3)
tensor_mul = torch.mul(tensor_just, tensor_just)
tensor_clamp = torch.clamp(tensor_just, min = -0.2, max = 2)
t1 = torch.Tensor([2, 3])
t2 = torch.Tensor([4, 5])
dot_product = torch.dot(t1, t2)
matrix = torch.Tensor([[1, 2, 3], [4, 5, 6]])
vector = torch.Tensor([0, 1, 2])
matrix_vector = torch.mv(matrix, vector)
another_matrix = torch.Tensor([[10,20], [30, 0], [0,50]])
matrix_mul = torch.mm(matrix, another_matrix)
torch.argmax(matrix_mul, dim = 1)
torch.argmin(matrix_mul, dim = 1)


#Converting between PyTorch and numpy tensors
import numpy as np
tensor = torch.rand(4,3)
type(tensor)
numpy_from_tensor = tensor.numpy()
type(numpy_from_tensor)
torch.is_tensor(tensor)
torch.is_tensor(numpy_from_tensor)
numpy_from_tensor[0, 0] = 100.0
numpy_from_tensor
tensor
numpy_arr = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])
tensor_from_numpy = torch.from_numpy(numpy_arr)
type(tensor_from_numpy)
torch.is_tensor(tensor_from_numpy)
tensor_from_numpy_arr = torch.as_tensor(numpy_arr)
