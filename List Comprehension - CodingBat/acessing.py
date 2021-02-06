from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pyautogui
import time
import re
import csv
import string
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import nbformat as nbf
import json
import yaml


def opening_url(page_link):
    
    driver_path = 'D:\Projetos\chromedriver'

    driver = webdriver.Chrome(driver_path)
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    driver.get(page_link)
    time.sleep(3)
    driver.maximize_window()
    time.sleep(2)
    
    return driver


def credentials(driver):
    
    conf = yaml.load(open('credentials.yml'))
    
#    credentials.yml:

#    text file with a structure like below:
     
#    "user:
#     email: your_email@**mail.com
#     password: yourpassword"
    
    email = conf['user']['email']
    pwd = conf['user']['password']
    driver.find_element_by_xpath("//input[@name='uname']").send_keys(email)
    driver.find_element_by_xpath("//input[@name='pw']").send_keys(pwd)
    driver.find_element_by_xpath("//input[@name='pw']").send_keys(Keys.ENTER)
    time.sleep(3)

    driver.refresh()
    time.sleep(3)
	
	return driver



page_link = "https://codingbat.com/python"

driver = opening_url(page_link)

driver = credentials(driver)

