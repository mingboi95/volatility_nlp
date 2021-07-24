# Stock Market Price Volatility Prediction
## Using Natural Language Processing and Sentiment Analysis techniques

Feature engineering is achieved through Sentiment Analysis techniques:
- Augmented VADER with Loughan-Mcdonald Financial Sentiment dictionary for the finance-focused slant of our social media tweets
- Bidirectional RNN with character-level embeddings (Colneric and Demsar - 2018) to handle OOV hashtags and words problem in social media tweets

Automatic feature and model selection modules has been implemented, achieved through Granger causality analysis and time-series cross-validation. Modelling is done through traditional models as well as ensembling methods, with model interpreptability handled by SHAP
## Results
| Metric      | Sentiment only | VIX only | Combined |
|-------------|----------------|----------|----------|
| Accuracy    | 0.6154         | 0.6603   | 0.7051   |
| F1-Weighted | 0.6041         | 0.5812   | 0.5864   |
| F2-Weighted | 0.6105         | 0.6244   | 0.6515   |

## Project Structure 
```
volatility_nlp/
┣ data/                         # Raw data files
┣ lexicon/                      # LM and VADER Lexicon
┣ models/                       # Bidirectional RNN model
┣ scrape/                       # Twitter scraper
┣ code.ipynb
┣ demo.ipynb
┣ emotion_predictor.py
┗ utils.py
```
- `code` contains the notebook for preprocessing, sentiment analysis and modelling
- `demo` contains our deliverable showing how to utilise the code for your own needs
- `scrape/scrape_twitter` is our code for scraping Twitter data

## Setup
Please ensure all other files in the folder are in the same directory as `demo`, and the environment is setup with preferably `environments.yml` or `requirements.txt`. Crucially, Theano backend for Keras must be enabled.

1) Create a [virtual environment](https://docs.python.org/3/library/venv.html) within your project directory. Setup the environment and activate it in your terminal:

Conda with `environments.yml` (preferred)
```
conda env create -f environment.yml
conda activate bt4222
```

Pip with `requirements.txt`
```bash
python3 -m venv /path/to/new/virtual/environment
pip install -r requirements.txt
$ C:\Users\...\project_folder> venv\Scripts\activate
```

2) Clone repository:
```bash
git clone https://github.com/mingboi95/volatility_nlp.git
```

## Contributors
1. [Clara](https://www.github.com/claratay)
2. [Glenn](https://www.github.com/glennljs) 
3. [Brian](https://www.github.com/wongchunghowbrian)
4. [Hui Lin](https://www.github.com/huilinloo)
5. [Yang Ming](https://www.github.com/glennljs)
