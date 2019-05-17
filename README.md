# POETRY 项目

AI model serving 是函数计算一个比较典型的应用场景。数据科学家训练好模型以后往往需要找软件工程师把模型变成系统或者服务，通常把这个过程称之为 model serving。函数计算无需运维和弹性伸缩的特性，正好符合数据科学家对高可用分布式系统的诉求。本文将介绍把一个 TensorFlow CharRNN 训练的自动写五言绝句古诗的模型部署到函数计算的例子。由于 python TensorFlow 依赖库和训练的模型的文件有数百兆，即使压缩也远超了函数计算 50M 代码包大小的限制。对于这类超大体积的文件，采用 NAS 文件系统是最佳选择。本文会介绍一种 Fun + NAS 的方法来解决 tensorflow serving 问题。

## 快速开始

### 1. 克隆 poetry 项目

```bash
git clone https://github.com/vangie/poetry.git
```

### 2. 修改 template.yml 文件

修改下面的 VPC 配合和 NAS 配置，这部分配置需要分别去相应的控制台进行创建并把对应的值拷贝出来。

```yaml
VpcConfig:
  VpcId: 'vpc-uf6r2qatgfbdhgy2rhplo'
  VSwitchIds: [ 'vsw-uf669ekf9zser1hrmgru4' ]
  SecurityGroupId: 'sg-uf6jcqx1ogbr37hkvgxv'
NasConfig:
  UserId: 10003
  GroupId: 10003
  MountPoints:
    - ServerAddr: '3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com:/'
    MountDir: '/mnt/nas'
```

### 3. 安装依赖

1. 执行 `fun install` 安装依赖。fun.yml 文件里声明了 tensorflow 的依赖和包括了训练模型的脚本命令，所以执行会比较费时。
2. 执行 `fun local invoke poetry`,这一步主要是为了生成 .fun/nas/3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com 目录。
3. 执行 `cp -R .fun/python .fun/nas/3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com/lib/` 将第一步安装的 tensorflow 依赖包拷贝到本地的 nas 目录。
4. 执行 `cp -R model .fun/nas/3be7b4835d-pvs14.cn-shanghai.nas.aliyuncs.com/` 将第一步训练的模型拷贝到本地的 nas 目录

本地测试

执行 `fun local invoke poetry` ，正确的返回结果如下

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

### 5. 上传文件至 NAS

目前 NAS 服务尚未提供直接上传文件的 API 和命令行。NAS 控制台提供的上传方式是先上传到 OSS，再由 OSS 导入 NAS 的功能，该功能需要申请开通。此外还有两种方法：

1. [nas.sh](https://yq.aliyun.com/articles/685803) —— 在 FC 里部署一个拷贝函数。这个方法目前有个限制，无法拷贝大于 6M 的文件。
2. 购买一台最便宜的按周计费的 ECS（大概 9 元钱），借助 ECS 的把文件拷贝进去。

下面我们假设 ecs 已经买好了，然后使用下面的命令拷贝文件到 nas 并解压

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

### 6. 部署和调用

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

至此，已经成功的将古诗创作程序成功部署到函数计算了。

## 参考阅读

1. [《21 个项目玩转深度学习——基于TensorFlow 的实践详解》](https://book.douban.com/subject/30179607/)