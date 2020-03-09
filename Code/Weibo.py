from selenium import webdriver
import time
import sys

class Weibo:
    def __init__(self):
        self.driver = webdriver.Chrome(executable_path='C:/Users/KayLa/Downloads/chromedriver.exe')

    # Login by Weibo's account and password
    def login(self, browser, login_u, login_p):
        # fill username and password
        browser.find_element_by_id("loginName").send_keys(login_u) # find the input where id is "loginName"
        browser.find_element_by_id("loginPassword").send_keys(login_p) # find the input where id is "loginPassword"
        time.sleep(3)

        # click login
        browser.find_element_by_id("loginAction").click() 
        time.sleep(3)

    # Login by QQ's account and password
    def qq_login(self, browser, acc, password):
        other_login = browser.find_element_by_xpath('//*[@id="loginWrapper"]/p/a')
        time.sleep(2)
        other_login.click()
        time.sleep(2)
        qq_login = browser.find_element_by_xpath('/html/body/ul/li/a')
        qq_login.click()
        time.sleep(2)
        browser.switch_to.frame("ptlogin_iframe")
        acc_pas = browser.find_element_by_xpath('//*[@id="switcher_plogin"]')
        acc_pas.click()
        time.sleep(2)
        
        # input account and password
        browser.find_element_by_id("u").send_keys(acc)
        browser.find_element_by_id("p").send_keys(password)
        time.sleep(2)
        browser.find_element_by_id("login_button").click()
        time.sleep(2)
    
    # slip verify 
    def slip_verify(self):
        d = self.driver
        iframe = d.find_element_by_xpath('//iframe')
        d.switch_to.frame(iframe)
        span_background = d.find_element_by_id('slide')
        span_background_size = span_background.size
        print(span_background_size)
        
        # Find the drag button
        button = d.find_element_by_id('tcaptcha_drag_button')
        button_location = button.location
        print(button_location)

        x_location = span_background_size["width"] * 5 / 7 # The drag button's destination
        y_location = 0

        while 1:
            time.sleep(5)
            print(x_location, y_location)
            try:
                # d.switch_to.frame('tcaptcha-iframe')
                alert = d.find_element_by_xpath('//*[@id="tcOperation"]/div[1]/div[6]/div[1]')
                webdriver.ActionChains(d).drag_and_drop_by_offset(button, x_location, y_location).perform()
            except Exception as e:
                print("get alert error: %s" %e)
                alert = None
            if alert:
                print(alert)
                x_location = x_location * 0.93 # each time decrease the distance from the starting point 
            else:
                break # Verify successfully
    
    # Follow the object
    def follow(self):
        d = self.driver
        
        # The location of the follow button
        follow_button = d.find_element_by_xpath('//*[@id="app"]/div[1]/div[3]/div[2]/div/div[1]/div')

        # check whether the user has been followed
        pop = d.find_element_by_xpath('//*[@id="app"]/div[1]/div[3]/div[2]/div/div[1]/div/span/h4')
        print(pop.text)

        if pop.text == "已关注":
            print("The object has been followed")
            return
        else:
            follow_button.click()
            time.sleep(3)
            group_button = d.find_element_by_xpath('//*[@id="app"]/div[1]/div[2]/div[2]/footer/div[2]')
            print("group the current followed object")
            time.sleep(3)
            group_button.click()
            time.sleep(3)


    # Thumbs up one weibo under the object
    def thumbs_up(self):
        # Thumbs up the second Weibo
        a = self.driver.find_element_by_xpath('//*[@id="app"]/div[1]/div[1]/div[7]/div/div/footer/div[3]/i')
        a.click()
        time.sleep(3)


if __name__ == '__main__':

    wb = Weibo()

    qq_user = '535946078'
    qq_pas = 'yjs19980401'
    
    uids = ['1223178222'] # store objects' uid

    wb.driver.get('https://passport.weibo.cn/signin/login') # open the weibo's login page
    wb.driver.implicitly_wait(5) # raise exception if cannot find elements in the given time

    wb.qq_login(wb.driver, qq_user, qq_pas)
    print("qq login successfully")
    wb.slip_verify()
    print("verify successfully")

    for u in uids:
        target = 'https://m.weibo.cn/u/' + u
        wb.driver.get(target) # open a user's page
        time.sleep(3)

        # follow the current object
        wb.follow()

        # Find the second Weibo
        second_weibo = wb.driver.find_element_by_xpath('//*[@id="app"]/div[1]/div[1]/div[7]/div/div/article/div/div[1]')
        second_weibo.text
        js = "arguments[0].scrollIntoView();" 
        # scroll down to that div region
        wb.driver.execute_script(js, second_weibo)  

        # Thumbs up the current Weibo
        wb.thumbs_up()
        
        # locate to the comment place, and click
        selector='//*[@id="app"]/div[1]/div[1]/div[7]/div/div/footer/div[2]/i'
        a = wb.driver.find_element_by_xpath(selector)
        text=a.text
        a.click()

        sys.exit() # For test

        wishes="Everything goes fine!"
        if text=='评论':
            # locate the cursor to comment
            comment = wb.driver.find_element_by_tag_name('textarea')
            comment.click()
            # input comment
            comment.send_keys(wishes)
            time.sleep(1)
            # locate the button "send"
            sendBtn = wb.driver.find_element_by_class_name('m-send-btn')
        else:
            # locate the cursor to comment
            focus = wb.driver.find_element_by_css_selector('span[class="m-box-center-a main-text m-text-cut focus"]')
            focus.click()
            # click the comment
            comment = wb.driver.find_element_by_tag_name('textarea')
            comment.click()
            # input comment
            comment.send_keys(wishes)
            # locate the button "send"
            sendBtn = wb.driver.find_element_by_class_name('btn-send')
            
        # send the comment
        sendBtn.click()