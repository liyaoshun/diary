# **git各种操作合集**

git branch -a   #查看当前git存在哪些分支
git checkout -b  cityscapes  origin/master   #切换origin/master分支到cityscapes分支


git 使用命令：
git clone -b branch-name --recursive http://github.xxxxx.xxx.git
git clone xxx  . 克隆仓库文件到本地
git add .  |  添加当前仓库下所有修改
git commit -m "说明"   | 将需要添加文件的信息写入git本地数据库中
创建分支： git checkout -b branch1
更新代码： git pull

//新建分支并提交
git checkout -b dev
git add *
git commit -m "info"
git push --set-upstream origin dev  |推送本地分支到远程仓库
git push origin dev:dev

//修改代码后上传
git status命令： 查看代码修改情况。
git add . 命令: 将修改代码的添加至暂存区。
git commit -m 'xxx' ：将暂存区的代码提交至本地仓库。
git push 命令：本地git仓库代码上传到远程仓库。

将远程git仓库里的指定分支拉取到本地（本地不存在的分支）
　　　git checkout -b 本地分支名 origin/远程分支名
    例如远程仓库里有个分支dev2,我本地没有该分支，我要把dev2拉到我本地
    git checkout -b dev2 origin/dev2


gitlab: name:yaoshun   pwd:liys123456


//切换branch  
    git checkout dev  切换到dev分支
    git checkout master    切换回主分支

//恢复本地误删除文件
git status 得到被删除的文件
git reset HEAD [被删除文件名称]
git checkout [被删除文件名称]

//下载子模块命令： git submodule update --init --recursive



# 修改为ssh 访问github：
1. cd ~/.ssh
2. ssh-keygen -t rsa -C "798226544@qq.com"   |  1. 输入保存秘钥名 githubssh.txt  2. 输入私钥秘钥  (不能忘记).  之后会在~/.ssh 文件夹下输出githubssh.txt 和githubssh.txt.pub两个文件。githubssh.txt.pub为保存的公钥。
3. cat githubssh.txt.pub
4. 复制界面显示的公钥到github的SSH key生成位置生成ssh key
5. 然后到自己的本地github文件下,   ssh -T gti@github.com 来配置当前文件使用ssh。  然后输入git remote add diary git@github.com:liyaoshun/diary.git 来链接本地和远端仓库.  最后使用  git remote -v 验证链接成功与否
6. git add . |  git commit -m ""  |  git push diary master

切换master为dev的上游分支，更新master中的文件到dev中：
1. git branch --set-upstream-to=origin/master  dev
2. git pull
然后切换回dev为自己的上游分支.
git branch --set-upstream-to=origin/dev  dev


## git rebase  | git stash | 

branch master
git cherry-pick commitID # 合并dev到master分支



本地有修改，需要更新其他人master的代码：
git stash : 缓存自己代码到自己分支，
git rebase
git add -U # 添加修改了的，之后还需要git status 查看一下add的状态
合并多次commit 中间别人没有commit的情况：
1. 确定是几次commit需要合并，假如是5，
git rebase -i HEAD~5
然后进入编辑器中将除了第一行的pick不变外，其他的pick都变为f. nano 保存使用Ctrl+X
git push 
取消rebase命令 git rebase --abort

git rebase 有冲突时，且本地没有需要修改的，可以使用：
1. git reset --hard commitID
2. git rebase users/leim/add_deep_learning_feature // 再进行rebase操作
3. git push -f 强制将rebase后的代码推到远端服务器



gerrit http 密码：OvQ/rcvQv1qKx96HwVFHkdYkmr0eJotROpit7Aydvg


gerrit:上传
git add .
git commit -m ""
git commit --amend  （ctrl+X） 重复提交只需要进行这一步及之后的
git push origin HEAD:refs/for/develop