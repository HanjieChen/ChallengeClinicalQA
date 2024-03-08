# ChallengeClinicalQA
Repo for the paper [Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions](https://arxiv.org/pdf/2402.18060.pdf)

### Datasets
- [Medbullets](https://github.com/HanjieChen/ChallengeClinicalQA/tree/main/medbullets)
- [JAMA Clinical Challenge](https://jamanetwork.com/collections/44038/clinical-challenge)

We do not publicably release the JAMA Clinical Challenge data due to the license constrain. Instead, we provide URLs to the articles and a scraper that you can use to get the data with license.

To install all the dependencies, run the following command

````
pip install -r requirements.txt
````

To scrape the data, run the following command

````
python jama_scraper.py
````

We have also provided the correct answer ids in `jama_links.json` for double check 

After scraping, the dataset are saved as `jama_raw.csv`  and `jama_raw.json`
