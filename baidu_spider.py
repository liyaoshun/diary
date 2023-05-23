#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version: 0.0.1
@author: ShenTuZhiGang
@time: 2021/03/08 19:44
@file: imagetest.py
@function:
@last modified by: ShenTuZhiGang
@last modified time: 2021/03/08 19:44
"""
import json
import os
import re
 
import cv2
import requests
from urllib.parse import urlparse, parse_qs
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from lxml import etree
import time
browser = webdriver.Chrome()

data = {
    'image':open("/media/robot/4T/18_seg_datesets/mseg_dataset/MsegV2/escalator/horizontal.elevator/istockphoto-1387521822-612x612.jpg",'rb')
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
}
r = requests.post('https://graph.baidu.com/upload?tn=pc&from=pc&image_source=PC_UPLOAD_IMAGE_MOVE&range={%22page_from%22:%20%22shituIndex%22}&extUiData%5bisLogoShow%5d=1', files=data, headers = headers).text
url = json.loads(r)["data"]["url"]
o = urlparse(url)
q = parse_qs(o.query, True)
sign = q['sign'][0]
# r1 = requests.get(url,headers = headers).text
# r0 = requests.get("https://graph.baidu.com/ajax/pcsimi?sign={}".format(sign)).text

url_q = "https://graph.baidu.com/ajax/pcsimi?sign={}".format(sign)

class Crawler_google_images:
    # 初始化
    def __init__(self):
        self.url = url

    # 获得Chrome驱动，并访问url
    def init_browser(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-infobars")
        browser = webdriver.Chrome(chrome_options=chrome_options)
        # 访问url
        browser.get(self.url)
        # 最大化窗口，之后需要爬取窗口中所见的所有图片
        browser.maximize_window()
        return browser

    #下载图片
    def download_images(self, browser,round=2):
        picpath = './google_sim_escalator'
        # 路径不存在时创建一个
        if not os.path.exists(picpath): os.makedirs(picpath)
        # 记录下载过的图片地址，避免重复下载
        img_url_dic = []

        count = 0 #图片序号
        pos = 0
        len_cur_imgs = 0
        for i in range(round):
            pos += 500
            # 向下滑动
            js = 'var q=document.documentElement.scrollTop=' + str(pos)
            browser.execute_script(js)
            time.sleep(1)
            # 找到图片
            # html = browser.page_source#也可以抓取当前页面的html文本，然后用beautifulsoup来抓取
            #直接通过tag_name来抓取是最简单的，比较方便
            # class="rg_i Q4LuWd"

            # img_elements = browser.find_elements('img') # img?
            img_elements = browser.find_elements(By.CSS_SELECTOR, "img")
            len_cur_imgs=0
            #遍历抓到的webElement
            for img_element in img_elements:
                # img_element.click()
                time.sleep(0.5)
                len_cur_imgs+=1
                img_url = img_element.get_attribute('src')
                # 前几个图片的url太长，不是图片的url，先过滤掉，爬后面的
                if isinstance(img_url, str):
                    # if len(img_url) <= 2000000:
                        #将干扰的goole图标筛去
                    # if 'images' in img_url:
                        #判断是否已经爬过，因为每次爬取当前窗口，或许会重复
                        #实际上这里可以修改一下，将列表只存储上一次的url，这样可以节省开销，不过我懒得写了···
                    if img_url not in img_url_dic:
                        try:
                            img_url_dic.append(img_url)
                            #下载并保存图片到当前目录下
                            filename = "./google_sim_escalator/" + str(count) + "_2.jpg"
                            r = requests.get(img_url)

                            imgDataNp = np.frombuffer(r.content, dtype='uint8')
                            img = cv2.imdecode(imgDataNp, cv2.IMREAD_UNCHANGED)
                            
                            cv2.imwrite(filename, img)
                            
                            count += 1
                            print('this is '+str(count)+'st img')
                            #防止反爬机制
                            time.sleep(0.2)
                        except:
                            print('failure')

    def run(self):
        self.__init__()
        browser = self.init_browser()
        self.download_images(browser, 100)#可以修改爬取的页面数，基本10页是100多张图片
        browser.close()
        print("爬取完成")
        
if __name__ == '__main__':
    craw = Crawler_google_images()
    craw.run()
