import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([1.0])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])

w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True


def forward(xs, bs):
    return w1 * xs**2 + w2 * xs + bs


def loss(xs, ys):
    y_pred = forward(xs, b)
    return (y_pred - ys) ** 2


print("Predict (Before Training): ", 4, forward(4, b).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # forward
        l = loss(x, y)
        # backward
        l.backward()
        print("\tgrad: ", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        # update
        w1.data -= 0.01 * w1.grad.item()
        w2.data -= 0.01 * w2.grad.item()
        b.data -= 0.01 * b.grad.item()

        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()

    print("Progress: ", epoch, l.item())

print("Predict (After Training): ", 4, forward(4, b).item())
