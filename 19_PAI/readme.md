# **计算资源访问命令记录**

## 3090 服务器访问命令
### openvpn安装与配置
1. 安装openvpn
sudo apt update
sudo apt install openvpn openresolv
2. 在 OpenVPN 客户端的 profile 描述文件中找到 auth-user-pass 这行参数，在后面增加路径: auth-user-pass /etc/openvpn/passwd  (注意：profile文件就是下载的yaoshun.li_Smart.ovpn)。
3. 将修改好的文件移到openvpn文件夹下mv yaoshun.li_Smart.ovpn /etc/openvpn/HARI_OPT.conf 
4. 在/etc/openvpn/目录下创建文件 passwd，文件内容两行，第一行用户名，第二行为密码。 (密码申请的时候会发邮件过来)
5. 启动OpenVPN服务,启动服务的命令和配置文件名称有关，比如配置文件为HARI_OPT.conf，则服务应该openvpn@HARI_OPT.service，如果Profile文件为其他名称的，比如 Science-SGP，则服务为openvpn@Science-SGP.service。
```
systemctl start openvpn@HARI_OPT
systemctl status openvpn@HARI_OPT
systemctl restart openvpn@HARI_OPT
```
如果不希望使用服务的方式管理openVPN，则可以直接命令启动：
```
openvpn --daemon ovpn-HARI_OPT  --cd /etc/openvpn --config /etc/openvpn/HARI_OPT.conf
# --daemon 程序是以守护进程的方式运行，即在后台运行，而不是在前台运行
# 或者：
openvpn --config /etc/openvpn/HARI_OPT.conf &
```
### ssh命令
ssh -l yaoshun.li 172.16.23.120 -p 10022

### 运行conda命令
<!-- source /data1/yaoshun.li/Anaconda/anaconda3/bin/activate
conda init -->

### 需要使用跳板机进行连接

http://172.16.31.180/core/auth/login/
登录后使用authentucator中的验证码进行验证