import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random


def extract_paragraphs(tempsoup):
    casepara=[]
    case=0
    discusiion=0
    discussionpara=[]
    paragraphs = []
    diagnosis=""
    fetchtrue=False
    founddiagnosis=False
    fetchoptiontrue=False
    chooseoption=""
    if tempsoup:
        article_para=tempsoup.find('div', class_='article-full-text')
        stop_condition="Article Information"
        for paragraph in article_para.find_all(['div', 'p']):
            if paragraph.name == 'div':
                if stop_condition in paragraph.get_text():
                    break
            else:
                if fetchoptiontrue==True:
                    chooseoption=paragraph.get_text()
                    fetchoptiontrue=False
                elif fetchtrue==True:
                    diagnosis=paragraph.get_text()
                    fetchtrue=False
                    founddiagnosis=True
                elif paragraph.get_text()=="Diagnosis":
                    fetchtrue=True
                # or paragraph.get_text()== "What To Do Next"
                elif paragraph.get_text()=="What to Do Next" or paragraph.get_text()== "What To Do Next" or paragraph.get_text()== "Answer":
                    if paragraph.get_text()=="Answer" or paragraph.get_text()== "What To Do Next":
                        print(paragraph.get_text())
                    fetchoptiontrue=True
                    founddiagnosis=False
                else:
                    if paragraph.get_text()=="Case":
                        case=1
                    if paragraph.get_text()=="Discussion":
                        case=0
                        discusiion=1
                    if np.char.count(paragraph.get_text(), ' ') + 1 <8:
                        continue
                    if case==1:
                        casepara.append(paragraph.get_text())
                    if discusiion==1:
                        discussionpara.append(paragraph.get_text())
                    paragraphs.append(paragraph.get_text())

        if founddiagnosis==True and fetchoptiontrue==False:
            chooseoption=diagnosis
    return paragraphs, diagnosis, chooseoption, casepara, discussionpara


def hasImage(tempsoup):
    article_para = tempsoup.find('div', class_='article-full-text')

    if article_para:
        # Check if there is an image in the article-full-text div
        image_div = article_para.find('div', class_='figure-table-image')
        if image_div and image_div.find('img'):
            return True 
    return False

def tellfield(tempsoup):
    article_para = tempsoup.find('div', class_='meta-article-type thm-col')
    super_class = tempsoup.find('div', class_='meta-super-class')
    if super_class:
        return article_para.get_text(),super_class.get_text()
    return article_para.get_text(),None
    

def extractMCQ(tempsoup):
    ques=None
    ans=None
    if tempsoup:
        #article_mcq=tempsoup.find('div',class='box-section online-quiz clip-last-child thm-bg')
        div_element = tempsoup.find('div', class_='box-section online-quiz clip-last-child thm-bg')
        if div_element==None:
            return None, ques,ans
        question_element = div_element.find('h4', class_='box-section--title')

        # Find all the p elements within the div (answers)
        p_elements = div_element.find_all('p', class_='para')

        # Extract and print the question and answers
        question = question_element.text
        answers = [p.text for p in p_elements]

        whetherTable=1
        ques=question
        return whetherTable,ques,answers


if __name__ == '__main__':
    url_df = pd.read_json('jama_links.json')
    df = pd.DataFrame(columns = ['URL', 'Title','Case','Discussion','MCQ_question','Option1','Option2','Option3','Option4','Diagnosis','Correct_option','HasImage','MedicalField','Superclass'])
    cnt = 0
    print("Start Scraping...")
    for index, row in url_df.iterrows():
        url = row['link']
        time.sleep(random.uniform(1, 2))
        scraper = cloudscraper.create_scraper(delay=2, browser="chrome")
        content = scraper.get(url).text
        soup = BeautifulSoup(content, 'html.parser')
        results = soup.findAll("div",{"class": "article-content"})
        title_value=row['title']
        checkimage=False

        whethermcq,question,answers=extractMCQ(soup)
        if whethermcq==None:
            print("No MCQ found or some other issue....trying again ")
            time.sleep(random.uniform(1, 2))
            scraper = cloudscraper.create_scraper(delay=2, browser="chrome")
            content = scraper.get(url).text
            soup = BeautifulSoup(content, 'html.parser')
            whethermcq,question,answers=extractMCQ(soup)

            if whethermcq==None:
                print("Again No MCQ found...Move to next")
                continue

        paragraphs,diagnosis,chooseoption,casepara,discussionpara = extract_paragraphs(soup)

        checkImage=hasImage(soup)
        HasImage="No"

        if checkImage==True:
            HasImage="Yes"

        articleType,superclass=tellfield(soup)

        combineCasepara=""
        combinediscussionpara=""

        for para in casepara:
            combineCasepara+=para

        for para in discussionpara:
            combinediscussionpara+=para

        dff={'URL': url, 'Title': title_value, 'Case': combineCasepara, 'Discussion': combinediscussionpara, 'MCQ_question': question, 'Option1': answers[0], 'Option2': answers[1], 'Option3': answers[2], 'Option4': answers[3], 'Diagnosis': diagnosis, 'Correct_option': chooseoption, 'HasImage': HasImage, 'MedicalField': articleType, 'Superclass': superclass}
        cnt += 1
        if cnt%10==0:
            print(f"{cnt} Links are Successfully Fetched")
        df = df._append(dff, ignore_index=True)

    print("Scraping Finished")
    df.to_csv("jama_raw.csv", index=False)
    df.to_json("jama_raw.json")
    print("Files Saved")
