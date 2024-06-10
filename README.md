# ChallengeClinicalQA
Repo for the paper [Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions](https://arxiv.org/pdf/2402.18060.pdf)

### Datasets
- [Medbullets](https://github.com/HanjieChen/ChallengeClinicalQA/tree/main/medbullets)
- [JAMA Clinical Challenge](https://jamanetwork.com/collections/44038/clinical-challenge)

We do not publicably release the JAMA Clinical Challenge data due to license constrains. Instead, we provide [URLs](https://github.com/HanjieChen/ChallengeClinicalQA/blob/main/jama_links.json) to the articles and a [scraper](https://github.com/HanjieChen/ChallengeClinicalQA/blob/main/jama_scraper.py) that you can use to obtain the data with the appropriate license. Please check your access to JAMA articles before you run the script.

Install the required dependencies
````
pip install -r requirements.txt
````

Scrape the data
````
python jama_scraper.py
````

The data will be saved in `jama_raw.csv` and `jama_raw.json` files.

### Reference
If you find this repository helpful, please cite our paper:
```bibtex
@article{chen2024benchmarking,
  title={Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions},
  author={Chen, Hanjie and Fang, Zhouxiang and Singla, Yash and Dredze, Mark},
  journal={arXiv preprint arXiv:2402.18060},
  year={2024}
}
```
