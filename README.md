# POETRY 项目

AI model serving 是函数计算一个比较典型的应用场景。数据科学家训练好模型以后往往需要找软件工程师把模型变成系统或者服务，通常把这个过程称之为 model serving。函数计算无需运维和弹性伸缩的特性，正好符合数据科学家对高可用分布式系统的诉求。本文将介绍把一个 TensorFlow CharRNN 训练的自动写五言绝句古诗的模型部署到函数计算的例子。

基本上所有的 FaaS 平台为了减少平台的冷启动，都会设置代码包限制，函数计算也不例外。由于 python TensorFlow 依赖库和训练的模型的文件有数百兆，即使压缩也远超了函数计算 50M 代码包大小的限制。对于这类超大体积的文件，函数计算命令行 Fun 工具原生支持了这种大依赖部署（3.2.0 版本以上），只需根据向导不需额外操作。
<a name="c182e73c"></a>
## 快速开始

<a name="3f118699"></a>
### 1. 克隆 poetry 项目

```bash
git clone https://github.com/vangie/poetry.git
```

<a name="97f64b9b"></a>
### 2. 安装依赖

执行 `fun install` 安装 FunFile 中声明的 tensorflow 依赖。由于训练模型的脚本比较费时，所以提前将训练好的模型存放在 model 目录中。

```bash
$ fun install
using template: template.yml
start installing function dependencies without docker

building poetry/poetry
Funfile exist, Fun will use container to build forcely
Step 1/3 : FROM registry.cn-beijing.aliyuncs.com/aliyunfc/runtime-python3.6:build-1.7.7
 ---> 373f5819463b
Step 2/3 : WORKDIR /code
 ---> Using cache
 ---> f9f03330ddde
Step 3/3 : RUN fun-install pip install tensorflow
 ---> Using cache
 ---> af9e756d07c7
sha256:af9e756d07c77ac25548fa173997065c9ea8d92e98c760b1b12bab1f3f63b112
Successfully built af9e756d07c7
Successfully tagged fun-cache-1b39d414-0348-4823-b1ec-afb05e471666:latest
copying function artifact to /Users/ellison/poetry
copy from container /mnt/auto/. to localNasDir

Install Success

Tips for next step
======================
* Invoke Event Function: fun local invoke
* Invoke Http Function: fun local start
* Build Http Function: fun build
* Deploy Resources: fun deploy
```

<a name="2a6f38fa"></a>
### 3. 运行本地函数

执行 `fun local invoke` 可以在本地运行函数，正确的返回内容如下：

```bash
$ fun local invoke poetry
Missing invokeName argument, Fun will use the first function poetry/poetry as invokeName

skip pulling image aliyunfc/runtime-python3.6:1.7.7...
FunctionCompute python3 runtime inited.
FC Invoke Start RequestId: b125bd4b-0d23-447b-8d8c-df36808a458b
.......（省略了部分日志）
犬差花上水风，一月秋中时。
江水无人去，山山有不知。
江山一中路，不与一时还。
山水不知处，江阳无所逢。
山风吹水色，秋水入云中。
水月多相见，山城入水中。
江云无处处，春水不相归。
野寺春江远，秋风落月深。

RequestId: 938334c4-5407-4a72-93e1-6d59e52774d8          Billed Duration: 14074 ms       Memory Size: 1998 MB    Max Memory Used: 226 MB
```

<a name="e2f30c32"></a>
### 4. 部署函数

通过 `fun deploy` 部署函数到函数计算。

```bash
$ fun deploy
```

fun 会自动完成依赖部署，当 fun deploy 检测到打包的依赖超过了平台限制（50M），会进入到配置向导，帮助用户自动化的配置。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/502931/1577342619409-f43d3642-5526-458c-8519-96e4d61fbb4a.png#align=left&display=inline&height=60&name=image.png&originHeight=120&originWidth=1484&size=182866&status=done&style=none&width=742)

选择 "Y" 之后就不需要做其他事情，等到部署完成即可。

<a name="RYADM"></a>
### 5. 运行远端函数

通过 fun invoke 调用远端函数（也可以通过[函数计算控制台](http://fc.console.aliyun.com)调用）：

```shell
$ fun invoke

using template: template.yml

Missing invokeName argument, Fun will use the first function poetry/poetry as invokeName

========= FC invoke Logs begin =========
省略部分日志...
Restored from: /mnt/auto/model/poetry/model-10000
FC Invoke End RequestId: c0d7947d-7c44-428e-a5a0-30e6da6d1d0f

Duration: 18637.47 ms, Billed Duration: 18700 ms, Memory Size: 2048 MB, Max Memory Used: 201.10 MB
========= FC invoke Logs end =========

FC Invoke Result:
役不知此月，不是无年年。
何事无时去，谁堪得故年。
不知无限处，相思在山山。
何必不知客，何当不有时。
相知无所见，不得是人心。
不得无年日，何时在故乡。
不知山上路，不是故人人。
```

至此，已经将古诗创作程序成功部署到函数计算了。

<a name="5c76fe3e"></a>
## 更多参考

1. [21 个项目玩转深度学习——基于TensorFlow 的实践详解](https://book.douban.com/subject/30179607)