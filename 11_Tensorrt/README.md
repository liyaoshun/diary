当前仓库下的gather插件有如下缺点:
```
1. 没有编写half精度的cuda代码
2. 使用此插件的时候输入数据类型做了严格的限制，输入数据类型必须为float32,index 数据必须为int64类型.不然会出现如下错误提示（Could not find any supported formats consistent with input/output data types）
```
