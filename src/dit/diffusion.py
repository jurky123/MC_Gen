class Diffusion():
    def __init__(self, config):
        self.config = config
        #初始化参数
    def add_noise(self, x, time_step):
        #添加噪声训练时使用
        #返回加噪后的图片和噪声
        pass
    def step(self, x, time_step,model):
        #推理时调用，用model计算出噪声
        #CFG融合后去噪
        #返回去噪后的图片
        pass
    def sample(self, model, text_encoding):
        #从纯噪声开始，迭代调用step去噪，直到得到最终图片
        #返回最终图片
        pass

