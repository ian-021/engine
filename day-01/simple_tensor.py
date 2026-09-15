import torch

x = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])

# print("shape: ", x.shape)

flat = x.flatten()
# print(flat.shape)

back = flat.view(3,3)
print(back)
