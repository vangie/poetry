# POETRY 项目

AI model serving 是函数计算一个比较典型的应用场景。数据科学家训练好模型以后往往需要找软件工程师把模型变成系统或者服务，通常把这个过程称之为 model serving。函数计算无需运维和弹性伸缩的特性，正好符合数据科学家对高可用分布式系统的诉求。本文将介绍把一个 TensorFlow CharRNN 训练的自动写五言绝句古诗的模型部署到函数计算的例子。由于 python TensorFlow 依赖库和训练的模型的文件有数百兆，即使压缩也远超了函数计算 50M 代码包大小的限制。对于这类超大体积的文件，采用 NAS 文件系统是最佳选择。本文会介绍一种 Fun + NAS 的方法来解决 tensorflow serving 问题。

## 快速开始

### 1. 克隆 poetry 项目

```bash
git clone https://github.com/vangie/poetry.git
```

### 2. 安装依赖

执行 `fun install` 安装依赖。

由于 fun.yml 文件里声明了 tensorflow 的依赖和包括了训练模型的脚本命令，所以执行会比较费时，请耐心等待。

这里有一个小细节需要注意，我们在 fun.yml 指定了 target 属性，因此会被安装到本地模拟的 [NAS 目录](https://yq.aliyun.com/articles/712700)。

动态演示（为了演示效果，我们将训练的 max_steps 参数设置为 100）：

![](https://tan-blog.oss-cn-hangzhou.aliyuncs.com/img/fun_nas_tensorflow_model_serving_demo_for_install.gif)

### 3. 运行函数

执行 `fun local invoke` 可以在本地运行函数，正确的返回内容如下：

```bash
$ fun local invoke poetry                                             
Reading event data from stdin, which can be ended with Enter then Ctrl+D
(you can also pass it from file with -e)
mouting local nas mock dir /Users/vangie/Desktop/poetry/.fun/nas/3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com/ into container /mnt/nas

skip pulling image aliyunfc/runtime-python3.6:1.5.2...
FunctionCompute python3 runtime inited.
FC Invoke Start RequestId: 938334c4-5407-4a72-93e1-6d59e52774d8
.......（省略了部分日志）
不见江中客，无言此别归。
江风秋雨落，山色夜山长。
不问江南客，孤舟在故乡。
一年如远别，何处是归人。
一夜无人

RequestId: 938334c4-5407-4a72-93e1-6d59e52774d8          Billed Duration: 14074 ms       Memory Size: 1998 MB    Max Memory Used: 226 MB
```

动态演示：
![](https://tan-blog.oss-cn-hangzhou.aliyuncs.com/img/fun_nas_tensorflow_model_serving_demo_for_local.gif)

### 4. 上传本地机器学习依赖、模型到 NAS 服务

我们在上一步通过 `fun local invoke` 命令在本地调通了 Fun NAS，但由于目前，我们的 NAS 资源只存在于本地，因此还要[上传到 NAS 服务上](https://yq.aliyun.com/articles/712700)。

下面，我们分别介绍 Fun Nas 以及 ECS 这两种上传方式。

#### 4.1 通过 Fun Nas 上传（推荐）

1. 执行 fun nas init 初始化 nas 的相关配置
2. 执行 fun nas info 可以查看本地 nas 的目录位置，因为我们前面 fun install target 直接指定了这个位置，因此不需要再做任何操作了
3. 执行 fun nas sync 将本地 nas 资源上传到云端 NAS 服务
4. 执行 fun nas ls nas://poetry:/mnt/auto/ 查看我们是否已经正确将文件上传到了 nas

fun nas sync 的动态演示（为了演示效果，我们缩减了依赖体积）：

![](https://tan-blog.oss-cn-hangzhou.aliyuncs.com/img/fun_nas_tensorflow_model_serving_demo_for_nas.gif)

#### 4.2 通过 ECS 上传

除了可以使用 Fun Nas 的方式上传外，我们还可以使用 ECS 的方式，将本地 NAS 资源上传。

假设我们已经拥有了一台 ECS，然后使用下面的命令拷贝文件到 nas 并解压：

```bash
# 挂载 nas 网盘
mount -t nfs -o vers=4.0 3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com:/ /mnt/nas

# 压缩本地要上传的目录
cd .fun/nas/3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com/
tar -czvf nas.tar.gz lib model

# 拷贝到 nas 目录
scp nas.tar.gz root@47.103.83.174:/mnt/nas

# 解压
tar -xvf nas.tar.gz
```

### 5. 部署函数

通过 fun deploy 部署函数到远端。

```bash
$ fun deploy             
using region: cn-shanghai
using accountId: ***********4733
using accessKeyId: ***********EUz3
using timeout: 60

Waiting for service poetry to be deployed...
        Waiting for function poetry to be deployed...
                Waiting for packaging function poetry code...
                package function poetry code done
        function poetry deploy success
service poetry deploy success
```

通过 fcli 调用远端函数（也可以通过[函数计算控制台](http://fc.console.aliyun.com)调用）：

```shell
$ fcli function invoke -s poetry -f poetry
换<unk>金龙瑁旒鸯垓疠萏萏瑁蟀瑁鸪雳萏萏萏蟀雳萏雳瑁雳瑁萏瑁瑁瑁鸪鸪蟀蟀蟀鸪蟀蟀萏萏萏蟀瑁萏蓉熳珑蟀熳萏缈皪惮萏萏皪惮皪琶萏萏珑琵疠缈轳寞雨风香。
春山无处处，秋色向江流。
不是东南望，孤山有一情。
春风不可见，一日向江流。
不见无人去，无时见白头。
相思一相见，相见一中风。
何日一相识，何人有此心。
何年不可识，相忆在江山。
一日一秋水，何来一山风。
相逢有何计，不见故人心。
不得何人去，无年不自同。
一来多此路，何处不堪寻。
此日多无事，无时自不知。
何年无此处，不是不相逢。
一日无如远，春风不自归。
何当有君客，不见旧人情。
此去相思处，不堪何处归。
不知青柳外，不得不堪亲。
不见青花去，无人
```

至此，已经将古诗创作程序成功部署到函数计算了。

## 更多参考

1. [21 个项目玩转深度学习——基于TensorFlow 的实践详解](https://book.douban.com/subject/30179607) 
2. [开发函数计算的正确姿势 —— Fun 自动化 NAS 配置](https://yq.aliyun.com/articles/712693)
3. [开发函数计算的正确姿势 —— 使用 Fun NAS 管理 NAS 资源](https://yq.aliyun.com/articles/712700)