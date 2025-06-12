import re
import emoji
import contractions
import langdetect
from nltk.tokenize import word_tokenize, TweetTokenizer, WhitespaceTokenizer, RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm


# Initialize tools
lemma = WordNetLemmatizer()
porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
snowball_stemmer = SnowballStemmer('english')
stop = set(stopwords.words('english'))

# Define tokenizers
tokenizers = {
    'word_tokenize': word_tokenize,
    'tweet_tokenizer': TweetTokenizer().tokenize,
    'whitespace_tokenizer': WhitespaceTokenizer().tokenize,
    'regexp_tokenizer': RegexpTokenizer(r'\w+').tokenize
}

def emoji_to_text(text):
    """Convert emojis to their textual descriptions"""
    return emoji.demojize(text, delimiters=(" ", " "))

def find_monetary_values(text):
    """Find monetary values and replace with MONETARY_VALUE_STAMP"""
    # Pattern for currency symbols followed by numbers or numbers followed by currency codes
    pattern = r'[$€£¥₹]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:USD|EUR|GBP|JPY|INR|dollars|euros|pounds)'
    return re.sub(pattern, " MONETARY_VALUE_STAMP ", text, flags=re.IGNORECASE)

def find_dates(text):
    """Find dates and replace with DATE_STAMP"""
    # Various date formats: MM/DD/YYYY, Month DD YYYY, DD-MM-YYYY, etc.
    pattern = r'\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b'
    return re.sub(pattern, " DATE_STAMP ", text, flags=re.IGNORECASE)

def find_time_references(text):
    """Find time references and replace with TIME_STAMP"""
    # Various time formats: 12:30, 12:30 PM, 12h30, etc.
    pattern = r'\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?|\d{1,2}[Hh]\d{2}(?:[Mm])?)\b'
    return re.sub(pattern, " TIME_STAMP ", text, flags=re.IGNORECASE)

def expand_all_contractions(text):
    """Expand contractions like don't to do not"""
    return contractions.fix(text)

def remove_x000d(text):
    """Remove x000d characters that appear in some text data"""
    return re.sub(r'x000d', '', text)

def detect_language(text):
    """Detect language of the text and return it along with the text"""
    try:
        lang = langdetect.detect(text)
    except:
        lang = 'unknown'
    return text, lang

#------------------------------------MAIN FUNCTION-------------------------------------

def clean(text_list, tokenizer_name='word_tokenize', lemmatize=False, stemmer_type=None, 
          remove_html=True, convert_emoji=True, replace_monetary=True, 
          replace_dates=True, replace_time=True, expand_contractions=True):
    """
    Clean text using multiple preprocessing steps in specific order.
    
    Args:
        text_list: List of text to clean
        tokenizer_name: Name of tokenizer to use ('word_tokenize', 'tweet_tokenizer', etc.)
        lemmatize: Whether to apply lemmatization
        stemmer_type: Type of stemmer to use (None, 'porter', 'lancaster', 'snowball')
        remove_html: Whether to remove HTML tags
        convert_emoji: Whether to convert emojis to text
        replace_monetary: Whether to replace monetary values
        replace_dates: Whether to replace dates
        replace_time: Whether to replace time references
        expand_contractions: Whether to expand contractions
    
    Returns:
        List of cleaned text
    """
    updates = []
    
    for j in tqdm(text_list):
        text = j
        
        # 1. Language detection (will not filter, just detect)
        text, lang = detect_language(text)
        
        # 2. Converting emojis to text descriptions
        if convert_emoji:
            text = emoji_to_text(text)
        
        # 3. LOWERCASE TEXT (kept from original function)
        text = text.lower()
        
        # 4. Remove HTML tags
        if remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # 5. Replacing monetary values with "MONETARY_VALUE_STAMP"
        if replace_monetary:
            text = find_monetary_values(text)
        
        # 6. Replacing dates with "DATE_STAMP"
        if replace_dates:
            text = find_dates(text)
        
        # 7. Replacing time references with "TIME_STAMP"
        if replace_time:
            text = find_time_references(text)
        
        # 8. Expanding contractions (if they exist)
        if expand_contractions:
            text = expand_all_contractions(text)
        
        # 9. REMOVE URLs (kept from original function)
        text = re.sub(r'http\S+', '', text)
        
        # 10. Removing "x000d" characters
        text = remove_x000d(text)
        
        # 11. REMOVE NUMERICAL DATA and PUNCTUATION (kept from original function)
        text = re.sub("[^a-zA-Z]", " ", text)
        text = re.sub("br", "", text)  # remove "br"
        
        # 12. Tokenization (with selected tokenizer)
        if tokenizer_name in tokenizers:
            tokens = tokenizers[tokenizer_name](text)
        else:
            tokens = text.split()  # Default to simple split
        
        # 13. REMOVE STOPWORDS
        tokens = [word for word in tokens if word not in stop]
        
        # 14. Lemmatize
        if lemmatize:
            tokens = [lemma.lemmatize(word) for word in tokens]
        
        # 15. Stemming
        if stemmer_type == 'porter':
            tokens = [porter_stemmer.stem(word) for word in tokens]
        elif stemmer_type == 'lancaster':
            tokens = [lancaster_stemmer.stem(word) for word in tokens]
        elif stemmer_type == 'snowball':
            tokens = [snowball_stemmer.stem(word) for word in tokens]
        
        # 16. Rejoining the words back into a string
        text = " ".join(tokens)
        
        updates.append(text)
    
    return updates



