
import torch

from src.model.transformer import make_model

def test_make_model():
    model = make_model(10, 10, 2)

    # 导出模型
    torch.save(model, 'test_transformer.pth')

    model = torch.load('test_transformer.pth')

    model.eval()


# 测试清除
def test_clean():
    import os
    os.remove('test_transformer.pth')
