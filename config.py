# 配置文件 （包含模型定义，数据处理和训练过程中的变量）
class DefaultConfig(object):
    env = 'default' #visdom environment
    model = 'resnet101'

    root = r'C:\Users\y63qiu\PycharmProjects\Pure protein project\data\images'

    batch_size=10
    use_gpu=True
    num_workers = 6
    print_freq = 5 # print info every N batch

    result_file = 'AB40_Alphasyn_result.csv'

    max_epoch = 30
    lr = 0.01
    lr_decay = 0.8
