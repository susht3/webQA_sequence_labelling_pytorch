import requests  
from bs4 import BeautifulSoup  
import time  
import jieba
import urllib  
from tqdm import *
import time, threading
from time import ctime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
 
stopwords = ['是', '的', '谁', '什么', '和', '了', '我', '你', '知道', '哪', '？', '?', '，', ',', '.', '。', '：', ':']

def clean_question(question):
    ques = list(jieba.cut(question))
    for w in stopwords: 
        if w in ques: ques.remove(w)
    return ques        

def match_key_words(main_ques, other):
    #if len(other) < 8:
    #    return True
    for word in main_ques:
        if word in other:
            return True
    return False
    
    
def get_page(ques, one, url):  
    evidences = []
    
    page_question_No = 1 + one   
    wb_data = requests.get(url)  
    wb_data.encoding = ('gbk')  
    soup = BeautifulSoup(wb_data.text, 'lxml')  
    webdata = soup.select('a.ti')  

    for title,url in zip(webdata, webdata):  
        #data = [title.get('title'), url.get('href')]   
        #print(page_question_No, ' ------------------------------------ \n')
        #print ('Question: ', title.get_text(), '\n') 
        
        url_sub = url.get('href')  
        wb_data_sub = requests.get(url_sub)  
        wb_data_sub.encoding = ('gbk')        
        soup_sub = BeautifulSoup(wb_data_sub.text, 'lxml')  
        best_answer = soup_sub.find('pre', class_ = "best-text mb-10")  
                          
        if best_answer != None:  
            best = best_answer.get_text(strip = True)  
            if match_key_words(ques, best):
                evidences.append(best)
 
        else:  
            better_answer = soup_sub.find_all('div', class_ = "answer-text line")  

            if better_answer != None:  
                for i_better, better_answer_sub in enumerate(better_answer):  
                    better = better_answer_sub.get_text(strip = True)  
                    if match_key_words(ques, better):
                        evidences.append(better) 
                    
        page_question_No += 1
        
    return evidences
                     
evidencess = []    
def get_evidences(question, pages = 20): 
    print('Getting eivdences from baiduzhidao....')
    url = "https://zhidao.baidu.com/search?word=" + urllib.parse.quote(question) + "&pn="  
    
    ques = clean_question(question)
    evidences_list = []
    for one in tqdm(range(0, pages, 10)):  
        evidencess = []
        #evidences = get_multi_thread_page(ques, one, url + str(one))
        evidences = get_page(ques, one, url + str(one))  
        if evidences != []:
            evidences_list.extend(evidences)
        time.sleep(2)  
        
    print('evidences: ', len(evidences_list))   
    #evidences_list = rank(evidneces_list)
    return evidences_list

# ---------------------------------


#evidencess = []
lock = threading.Lock()
def get_href(ques, title, url):
    url_sub = url.get('href')  
    wb_data_sub = requests.get(url_sub)  
    wb_data_sub.encoding = ('gbk')        
    soup_sub = BeautifulSoup(wb_data_sub.text, 'lxml')  
    best_answer = soup_sub.find('pre', class_ = "best-text mb-10")  
    
    evidences = ['no_answer']
    if best_answer != None:  
        best = best_answer.get_text(strip = True)  
        if match_key_words(ques, best):
            if lock.acquire():
                evidencess.append(best)
                lock.release()
                    #print(evidencess)
    else:  
        better_answer = soup_sub.find_all('div', class_ = "answer-text line")  

        if better_answer != None:  
            for i_better, better_answer_sub in enumerate(better_answer):  
                better = better_answer_sub.get_text(strip = True)  
                if match_key_words(ques, better):
                    if lock.acquire():
                        evidencess.append(better) 
                        lock.release()
                        #print(evidencess)
    #return 1 #evidences                  

def get_multi_thread_page(ques, one, url):  
    threads = []
    #evidences = []
    
    page_question_No = 1 + one   
    wb_data = requests.get(url)  
    wb_data.encoding = ('gbk')  
    soup = BeautifulSoup(wb_data.text, 'lxml')  
    webdata = soup.select('a.ti')  
    nb_thread = len(webdata)
    
    for i in range(nb_thread):
        t = threading.Thread(target=get_href(ques, webdata[i], webdata[i]), name='LoopThread')
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
        #href_evidences = t.get_result()
        #evidneces.extend(href_evidences)
 
    return evidencess

if __name__ == '__main__':
    question = '三生三世十里桃花女主角是谁？'
    evidences = get_evidences(question) 
    #print(evidences)
    
    
    
    