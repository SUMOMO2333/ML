from __future__ import print_function # 将python3中的print特性导入当前版本
import os
from PIL import Image # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid

def softmax_regression(img,label):
    """
    定义softmax分类器：
        一个一softmax为激活函数的全链接层
    :return:
        predict_image --分类结果
    """
    predict_img = fluid.layers.fc(input = img,size = 10,act = 'softmax')
    return predict_img,label


def multilayer_perception(img,label):
    """
        定义多层感知机分类器
        含有两个隐藏层（全连接层）的多层感知机
        其中钱两个隐藏层的激活函数是RULU，输出层的激活函数是softmax
    :return:
        predict_img --分类结果
    """
    # 建立第一个隐层（全链接层） ，激活函数为relu
    hidden_1 = fluid.layers.fc(input=img,size=200,act='relu')
    # 建立第二个隐层（全连接层） ， 激活函数为relu
    hidden_2 = fluid.layers.fc(input=hidden_1,size=200,act='relu')
    # # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    prediction = fluid.layers.fc(input=hidden_2,size=10,act='softmax')
    return  prediction , label


def converlutional_neural_network(img,label):
    """
    定义卷积神经网络分类器：
        输入的二维图像经过两个卷积-池化层，使用softmax为激活函数的全连接层作为输出层
    :return:
        predict --分类结果
    """

    #第一个卷积-池化层，使用20个5*5的滤波器，池化大小为2，池化步长为2，激活函数为RELU
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input = img, # 输入数据的名称
        filter_size = 5 , #滤波器尺寸
        num_filters = 20 , #滤波器数量
        pool_size = 2,   #池化大小
        pool_stride = 2, #池化步长
        act = 'relu'
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    #第二个卷积-池化层，使用50个5*5的滤波器，池化大小为2，池化步长为2，激活函数为relu
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input = conv_pool_1, #输入数据
        filter_size = 5,
        num_filters = 50,
        pool_size = 2,
        pool_stride = 2,
        act = 'relu'
    )

    prediction = fluid.layers.fc(input=conv_pool_2,size=10,act='softmax')

    return  prediction ,label


def train_progam(img,label):
    """
    配置 train_program
    :return:
    predict --分类结果
    avg_cost --平均损失
    acc      --准确率
    """
    # predict,label = softmax_regression(img,label) #取消注释将使用softmax回归
    # predict,label = multilayer_perception(img,label) #取消注释将使用多层感知机
    predict ,label = converlutional_neural_network(img,label) #取消注释将使用卷据-池化网络

    #使用交叉熵计算predit和label之间的损失函数
    cost = fluid.layers.cross_entropy(input = predict,label=label)
    avg_cost = fluid.layers.mean(cost)

    #计算分类准确率
    acc = fluid.layers.accuracy(input=predict,label=label)
    return predict, [avg_cost, acc]


def optimizer_funtion():
    """
    优化函数： Adam optimizer ，神经网络中常用的优化函数
    learning_rate 是学习率，它的大小与网络的训练收敛速度有关
    """
    return fluid.optimizer.Adam(learning_rate=0.001)


def set_Feeder():
    """
    下一步，我们开始训练过程。paddle.dataset.mnist.train()和paddle.dataset.mnist.test()分别做训练和测试数据集。这两个函数各自返回一个reader——PaddlePaddle中的reader是一个Python函数，每次调用的时候返回一个Python yield generator（生成器）。
下面shuffle是一个reader decorator，它接受一个reader A，返回另一个reader B。reader B 每次读入buffer_size条训练数据到一个buffer里，然后随机打乱其顺序，并且逐条输出。
batch是一个特殊的decorator，它的输入是一个reader，输出是一个batched reader。在PaddlePaddle里，一个reader每次yield一条训练数据，而一个batched reader每次yield一个minibatch。
    :return:
    """
    #一个minibatch中有64个数据
    BATCH_SIZE = 64

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader=paddle.dataset.mnist.train(), #下载训练集，一个一次只能yield一条数的生成器
            buf_size= 500                  #读取500条数据后打乱顺序再《逐条》输出
        ),
        batch_size = BATCH_SIZE          #把逐条得到的数据，每64个作为以后minibatch输出
    )

    test_reader = paddle.batch(
        reader=paddle.dataset.mnist.test(),  # 下载训练集，一个一次只能yield一条数的生成器,
        batch_size = BATCH_SIZE  # 把逐条得到的数据，每64个作为以后minibatch输出
    )
    return train_reader ,test_reader


def event_haddle(pass_id,batch_id,cost):
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass_id: %d, Batch_id_ %d, Cost: %f" % (pass_id, batch_id, cost))


def event_haddle_plot():
    #将训练过程绘图表示
    from paddle.utils.plot import Ploter

    train_prompt = "Train cost"
    test_prompt = "Test cost"
    cost_ploter = Ploter(train_prompt, test_prompt)

    # cost_ploter.append(ploter_title, step, cost)
    # cost_ploter.plot()
    # pass


def train_test(train_test_program,train_test_feed,train_test_reader,exe,avg_cost,acc):
    #将分类准确率存在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
         avg_loss_np ,acc_np= exe.run(
            program = train_test_program,
            feed = train_test_feed.feed(test_data),
            fetch_list = [avg_cost,acc]
        )

    acc_set.append(float(acc_np))
    avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean


def main(save_dirname):
    # 获取数据
    train_reader, test_reader = set_Feeder()
    # 输入原始图像数据，大小为28*28*1
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    # 标签层，名称为label，对应输入图片的类别标签
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    #定义损失函数、准确率、和预测算法
    predict, [avg_cost, acc] = train_progam(img,label)

    #定义优化函数,传入损失
    optimizer = optimizer_funtion()
    optimizer.minimize(avg_cost)

    #初始化参数
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()
    # test_program = main_program.clone(for_test=True)

    # 设置数据输入器
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    #设置训练过程的超参
    PASS_NUM = 5
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]



    #创建执行器
    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id , data in enumerate(train_reader()):
            metrics = exe.run(
                program = main_program,
                feed = feeder.feed(data),
                fetch_list=[avg_cost,acc]
            )
            if step % 100 ==0:  #每训练100次打印一次log
                print("step =  %d ; epoch_num = %d ; Cost =  %f ; acc = %f" % (step, epoch_id, metrics[0] , metrics[1]))

            step += 1

        # 测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(train_test_program = main_program,
                                           train_test_reader = test_reader,
                                           train_test_feed = feeder,
                                           exe = exe,
                                           avg_cost = avg_cost,
                                           acc=acc)
        print("Test with Epoch %d, avg_cost: %s, acc: %s" % (epoch_id, avg_loss_val, acc_val))
        # event_handler_plot(test_prompt, step, metrics[0])
        lists.append((epoch_id, avg_loss_val, acc_val))

        #保存训练好的模型进行预测
        if save_dirname is not None:
            fluid.io.save_inference_model(
                dirname=save_dirname,
                feeded_var_names=['img'],
                target_vars = [predict],
                executor = exe,
            )


def inference(save_dirname,im):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        """
        使用 fluid.io.load_inference_model 获取 inference program desc,
        feed_target_names 用于指定需要传入网络的变量名
        fetch_targets 指定希望从网络中fetch出的变量名
        """

        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
            save_dirname, exe, None, None)
        # 将feed构建成字典 {feed_target_name: feed_target_data}
        # 结果将包含一个与fetch_targets对应的数据列表
        print("feed_target_names:",feed_target_names)
        results = exe.run(
            program = inference_program,
            feed = {feed_target_names[0]:im},
            fetch_list = fetch_targets
        )

        print("results:",results)
        lab = np.argsort(results)
        print("lab = ",lab)


        return lab







def load_test_image(file):
    """读取自己手写数字的图片"""
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im



if __name__ == '__main__':
    save_dirname = r"result_train\recognize_digits.inference.model"
    #训练模型，储存model
    # main(save_dirname)

    #读取自己手写的数字图片
    file_im = r'traindata\image_cecognition\infer_8.png'
    im = load_test_image(file_im)

    # 对手写图片进行预测
    lab = inference(save_dirname,im)

    print("Inference result of image/infer_{}.png is: {}".format(file_im[-11:],lab[0][0][0]) ) #这里原文是[0][0][-1]是错的，选出来交叉熵最大的数，应该用[0][0][0]选交叉熵最小的数是预测结构。
