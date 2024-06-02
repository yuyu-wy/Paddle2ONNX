import paddle
from onnxbase import APIOnnx
from onnxbase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, index, value):
        """
        forward
        """
        x = paddle.index_add(inputs,index, 0,value)
        return x


def test_index_add():
    """
    api: paddle.roll
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'index_add', [16])
    input_tensor = paddle.to_tensor(paddle.ones((3, 3)), dtype="float32")
    index = paddle.to_tensor([0, 2], dtype="int32")
    value = paddle.to_tensor([[1, 1, 1], [1, 1, 1]], dtype="float32")
    obj.set_input_data(
        "input_data",
        input_tensor,
        index,
        value
        )
    obj.run()

if __name__ == "__main__":
    test_index_add()