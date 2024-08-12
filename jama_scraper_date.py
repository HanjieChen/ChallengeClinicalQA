import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random
import json
import pdb

# Sample list of user agents for random selection
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/537.36 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Linux; Android 11; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Linux; Android 12; SM-S906N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Mobile Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
    "Mozilla/5.0 (Linux; Android 9; SM-J530F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; OnePlus 8T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; AS; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Linux; Android 8.0.0; Nexus 5X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Mobile Safari/537.36",
]

# Function to append data to a JSON file incrementally
def append_to_json(data, filename='jama_raw_date.json'):
    try:
        # Try loading existing data if file exists
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []  # Start with an empty list if file doesn't exist

    # Append new data
    existing_data.extend(data)

    # Write back to the JSON file
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=4)


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
    # Load the URLs from the JSON file
    url_df = pd.read_json('retry_jama_clinical_cases_ALPHA.json', orient='records')
    # url_df = pd.read_json('jama_clinical_test.json', orient='records')
    
    dff = []
    cnt = 0
    print("Start Scraping...")
    max_retries = 16
    failed_urls = []  # List to store failed URLs
    for index, row in url_df.iterrows():
        url = row['link']
        date = row['publication_year']  # Assuming this field exists
        time.sleep(random.uniform(1, 2))
        
        scraper = cloudscraper.create_scraper(delay=2, browser="chrome")

        # Set a maximum number of retries
        
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Create a new scraper instance for each URL
                scraper = cloudscraper.create_scraper(delay=2, browser="chrome")
                
                # Randomize user agent for each request
                scraper.headers.update({
                    'User-Agent': random.choice(user_agents)
                })
                
                content = scraper.get(url).text
                soup = BeautifulSoup(content, 'html.parser')
                results = soup.findAll("div",{"class": "article-content"})
                checkimage=False

                whethermcq, mcqquestion, answers = extractMCQ(soup)
                
                if whethermcq is not None:
                    # counter = 0
                    break  # Exit the loop if MCQ is found
                else:
                    # pdb.set_trace()
                    scraper = cloudscraper.create_scraper(delay=2, browser="chrome")
                    retry_count += 1
                    print(f"No MCQ found....trying again for the {retry_count} time")
                    time.sleep(random.uniform(3, 5))
            except Exception as e:
                print(f"Error occurred while scraping {url}: {e}")
                retry_count += 1
                time.sleep(random.uniform(3, 5))
        
        if retry_count == max_retries:
            print(f"Failed to scrape {url} after {max_retries} retries.")
            failed_urls.append(url)  # Add the failed URL to the list
            continue
        # Extract additional information
        paragraphs, diagnosis, chooseoption, casepara, discussionpara = extract_paragraphs(soup)

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
        question = combineCasepara + ' ' + mcqquestion
        # We directly copy the answer from jama_links.json to make sure they are correct

        # Prepare the data to be appended
        # [url, question, answers[0], answers[1], answers[2], answers[3], diagnosis, row['answer_idx'], row['answer'], combinediscussionpara, articleType]
        # data_to_append = [[url, date, mcqquestion, answers[0], answers[1], answers[2], answers[3], diagnosis, row['answer_idx'], row['answer']]]
        data_to_append = [[url, date, question, answers[0], answers[1], answers[2], answers[3], diagnosis, row['answer_idx'], row['answer'], combinediscussionpara, articleType]]
        # Append data to JSON incrementally
        append_to_json(data_to_append)

        # Prepare the DataFrame for CSV
        # dff.append([url, date, mcqquestion, answers[0], answers[1], answers[2], answers[3], diagnosis, row['answer_idx'], row['answer']])
        dff.append([url, date, question, answers[0], answers[1], answers[2], answers[3], diagnosis, row['answer_idx'], row['answer'], combinediscussionpara, articleType])
        cnt += 1
        if cnt % 10 == 0:
            print(f"{cnt} Links are Successfully Fetched")

    print("Scraping Finished")

    # Create DataFrame and write to CSV
    df = pd.DataFrame(dff, columns=['link', 'date', 'question', 'opa', 'opb','opc','opd','diagnosis', 'answer_idx','answer','explanation','field'])
    df.to_csv("jama_raw_date.csv", index=False)

    print("Data written to CSV.")

    # Write final DataFrame to JSON
    df.index.name = 'id'
    df = df.reset_index()
    json_dict = df.to_dict(orient='records')
    
    with open('jama_raw_date.json', 'w') as f:
        json.dump(json_dict, f, indent=4)

    # Write failed URLs to a text file
    with open('failed_urls_date.txt', 'w') as f:
        for url in failed_urls:
            f.write(url + '\n')

    print("Files Saved")
