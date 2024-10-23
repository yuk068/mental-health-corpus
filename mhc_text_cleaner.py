import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if you haven't
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_document(text):
    # 1. Lowercase the text
    text = text.lower()
    
    # 2. Remove all characters that are not lowercase a-z or spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Normalize all whitespace to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Remove links (tokens containing http or www)
    text = ' '.join([word for word in text.split() if not ('http' in word or 'www' in word)])
    
    # 5. Remove words with length >= 17
    text = ' '.join([word for word in text.split() if len(word) < 17])
    
    # 6. Handle characters repeated 3 or more times (e.g., 'aaa' -> 'a')
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # 7. Lemmatize the words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # 8. Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # 9. Remove tokens with char length = 1
    text = ' '.join([word for word in text.split() if len(word) > 1])
    
    return text