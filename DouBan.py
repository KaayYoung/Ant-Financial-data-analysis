
import time
import requests
import os
import csv
import re
import pathlib
import sys
import threading
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from selenium.common.exceptions import ElementNotVisibleException
from queue import Queue
from urllib import request
from webptools import webplib as webp
from selenium.common.exceptions import NoSuchElementException


class DBDY:
    def __init__(self):  # 进行初始化操作
        self.driver = webdriver.Chrome('C:/Users/KayLa/Downloads/chromedriver.exe')
        self.wait = WebDriverWait(self.driver, 20)
        js = 'window.open("https://www.baidu.com");'
        self.driver.execute_script(js)
 
    def open_page(self, url):
        driver = self.driver
        driver.switch_to.window(driver.window_handles[0])  # 定位至总览标签页
        driver.get(url)
        driver.refresh()
        time.sleep(5)

 
    def get_response(self):
        driver = self.driver
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')
        count = 0
 
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')

        items = soup.find_all('img', class_='cover') 
        print('information of the current web')
        print("number of items:" + str(len(items)))
        return items

    def producer(self, movie_q, movies):
        print("producing")
        for movie in movies:
            item = {
                'img':movie['src'],
                'title':movie['alt']
            }
            print(item['title'])

            movie_q.put(item)
        return movie_q

    def download_poster_img(self, movie_q):
        cur_path = pathlib.Path().absolute()
        file_path = str(cur_path) + '/poster_img/'

        while True:
            movie = movie_q.get()

            url = movie['img']
            title = movie['title']
            res = requests.get(url)
                
            new_title = title
            if '/' in title:
                new_title = title.replace('/', '')
                target = os.path.join(file_path, new_title)
            else:
                target = os.path.join(file_path, new_title)
            is_Exist = os.path.exists(target)
            if not is_Exist:
                print('download img file_path = ', target)
                jpg = new_title + '.jpg'
                request.urlretrieve(url, jpg)
                #webp.dwebp(title,,"-o")


if __name__ == '__main__':

    db = DBDY()
    
    pre = 'https://search.douban.com/movie/subject_search?search_text=成龙&cat=1002&start='

    movie_queue = Queue(maxsize=50) # A queue with size of 50
    movie_thread = []
    for i in range(5):
        thread2 = threading.Thread(target=db.download_poster_img, args=(movie_queue,))
        movie_thread.append(thread2) # five threads are used to download pictures

    # loop over 50 pages
    for i in range(50):
        print(i)
        target = pre + str(i * 15)
        #target = 'https://search.douban.com/movie/subject_search?search_text=成龙&cat=1002&start=0'
        
        db.open_page(target)
        movies = db.get_response()
        thread = threading.Thread(target=db.producer, args=(movie_queue,movies))

        # if the return result is none, that means we reach the end
        if not movies:
            print("reach the end")
            break
        if i == 0:
            start_time = time.time()
            thread.start()
        
        for i in range(5):
            movie_thread[i].start()
        
        # wait for all threads shutting down
        thread.join()    
        for i in range(5):
            movie_thread[i].join()

    db.driver.quit()
