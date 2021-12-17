#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 导入模块
from wxpy import *
import pdb



# 初始化机器人，扫码登陆
bot = Bot()


# 搜索名称含有 "游否" 的男性深圳好友
my_friend = bot.friends().search('A大腿')[0]


# 发送文本给好友
OBmy_friend.send('Hello WeChat!')
# 发送图片
my_friend.send_image('my_picture.jpg')



@bot.register()
def print_others(msg):
    print(msg)


# 回复 my_friend 的消息 (优先匹配后注册的函数!)
@bot.register(my_friend)
def reply_my_friend(msg):
    return 'received: {} ({})'.format(msg.text, msg.type)


# 自动接受新的好友请求
@bot.register(msg_types=FRIENDS)
def auto_accept_friends(msg):
    # 接受好友请求
    new_friend = msg.card.accept()
    # 向新的好友发送消息OA
    new_friend.send('哈哈，我自动接受了你的好友请求')
