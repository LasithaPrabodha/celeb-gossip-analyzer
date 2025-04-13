import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    nltk_available = True
except Exception as e:
    print(f"Warning: NLTK initialization failed: {e}")
    print("Will continue without sentiment analysis")
    nltk_available = False

# Try to load spaCy - with fallback if not available
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")  # Try loading the smaller model first
        spacy_available = True
        print("Loaded spaCy 'en_core_web_sm' model")
    except OSError:
        try:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
            spacy_available = True
            print("Successfully downloaded and loaded spaCy model")
        except Exception as e:
            print(f"Warning: Could not download spaCy model: {e}")
            spacy_available = False
except ImportError:
    print("Warning: spaCy not installed. NER features will be disabled.")
    spacy_available = False

# Sri Lankan celebrity names (sample list - would need to be expanded)
SL_CELEBRITIES = [
    "Jacqueline Fernandez", "Yohani", "Bathiya and Santhush", "Pooja Umashankar", 
    "Sangakkara", "Malinga", "Mahela", "Angelo Mathews", "Upul Tharanga",
    "Roshan Mahanama", "Sanath Jayasuriya", "Mahinda Rajapaksa", "Gotabaya Rajapaksa",
    "Sajith Premadasa", "Hiru Gossip", "Nehara Peiris", "Menaka Rajapaksa",
    "Jackson Anthony", "Iraj", "Dinakshie Priyasad", "Shanudrie Priyasad",
    "Ureshi Rajasinghe", "Santhush Weeraman", "Umaria Sinhawansa", "Bhathiya Jayakody",
    "Natasha Rathnayake", "Hirunika Premachandra", "Stephanie Siriwardhana",
    "Vinu Udani Siriwardana", "Sheshadri Priyasad", "Sangeetha Weeraratne",
    "Daisy Achchi", "Namal Rajapaksa", "Sudu Aiya", "Peshala", "Ryan Van Rooyen",
    "Piumi Hansamali", "Sachini Nipunsala", "Nadeesha Hemamali", "Samanalee Fonseka"
]

# Sri Lankan media companies, studios, production houses
SL_COMPANIES = [
    "Derana", "Sirasa", "Hiru", "Swarnavahini", "Rupavahini", "ITN", "TV Derana",
    "Sirasa TV", "Hiru TV", "Swarnavahini TV", "Rupavahini TV", "Shakthi TV",
    "EAP Films", "Torana Video", "Sarasaviya", "Maharaja Organization", "Dialog",
    "SLT Mobitel", "Etisalat", "Airtel", "Laxapana Studios", "Ceylon Theatres",
    "EAP Holdings", "Wijeya Newspapers", "Lanka Films", "Ranmihitenna Film Studio",
    "Unlimited Entertainment", "Power House", "Red Hen Productions",
    "Exceptional Entertainment", "Lyca Productions"
]

# Sri Lankan locations and venues
SL_LOCATIONS = [
    "Colombo", "Kandy", "Galle", "Negombo", "Jaffna", "Anuradhapura", "Batticaloa",
    "Trincomalee", "Nuwara Eliya", "Matara", "Odel", "Cinnamon Grand", "Taj Samudra",
    "Kingsbury", "Liberty Cinema", "Majestic City", "Nelum Pokuna", "BMICH",
    "Shangri-La", "Hilton Colombo", "Viharamahadevi Park", "Mount Lavinia Hotel",
    "Independence Square", "Arcade Independence Square", "Race Course", "Dutch Hospital",
    "Water's Edge", "Excel World", "Galle Face Hotel", "Galadari Hotel", "Jetwing Hotels",
    "Colombo City Centre", "Crescat Boulevard", "Marino Mall", "One Galle Face",
    "Bandaranaike Memorial International Conference Hall", "Lotus Pond Theatre"
]

# Sensationalist language specific to Sri Lankan gossip
SENSATIONALIST_TERMS = [
    "shocking", "bombshell", "scandal", "exclusive", "secret affair", "caught in the act",
    "breaking news", "exposed", "you won't believe", "leaked", "insider information",
    "drama", "controversy", "disaster", "disgrace", "shame", "infamous", "rumors",
    "sources close to", "anonymous source", "unnamed insider", "relationship drama",
    # Local terms and transliterated terms
    "keliya", "rasthiyadu", "hoda show", "hada", "pissu", "bayanak", "kunu harapa",
    "kohulan maththa", "jalaya", "baduwa", "katakaranawa", "asammanai"
]

# Specific categories of gossip based on your requirements
GOSSIP_CATEGORIES = {
    'secret_affair': [
        "affair", "secret relationship", "romance", "dating", "seeing each other", 
        "together", "intimate", "lovers", "cheating", "two-timing", "unfaithful", 
        "behind closed doors", "private meetings", "secret rendezvous", "hidden relationship",
        "meeting secretly", "sneaking around", "secret lover", "affair with", "caught together",
        "spotted together", "spending time together", "close relationship", "more than friends"
    ],
    
    'sexting': [
        "sexting", "explicit messages", "nude", "private photos", "intimate pictures", 
        "inappropriate texts", "leaked messages", "personal chat", "explicit content",
        "private conversation", "WhatsApp leak", "Instagram DM", "private DM", "messenger chat",
        "compromising photos", "revealing photos", "dirty messages", "flirty texts",
        "suggestive messages", "bedroom photos", "shower pictures", "inappropriate content",
        "private video call", "video scandal"
    ],
    
    'boss_employee': [
        "boss", "director", "producer", "manager", "supervisor", "senior", "junior",
        "assistant", "staff", "employee", "workplace relationship", "office romance",
        "professional relationship", "working together", "professional boundaries",
        "power imbalance", "casting couch", "favoritism", "special treatment", 
        "promotion", "career advancement", "mentor", "protege", "contract", "work relationship",
        "film set", "production", "studio", "company", "agency", "talent agency"
    ],
    
    'celebrity_fan': [
        "fan", "follower", "admirer", "supporter", "devotee", "fan club", "fan meeting",
        "meet and greet", "selfie", "autograph", "photo with fan", "fan encounter",
        "fan interaction", "obsessed fan", "stalker", "fan mail", "message from fan",
        "fan gift", "meeting fan", "private meeting", "special fan", "loyal fan",
        "dedicated fan", "fan relationship", "social media fan"
    ]
}

# Privacy related terms
PRIVACY_TERMS = [
    "private", "personal", "confidential", "intimate", "secret", "leaked", "exposed",
    "hacked", "unauthorized", "not meant to be shared", "hidden", "sensitive",
    "not for public", "behind closed doors", "off the record", "anonymous",
    "private account", "personal device", "personal chat", "locked", "password protected"
]

class SriLankanCelebrityGossipAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = None
    
    def load_data(self, csv_path):
        """Load and prepare data from CSV file"""
        try:
            data = pd.read_csv(csv_path)
            # Check for required column
            if 'comment_text' not in data.columns:
                raise ValueError("CSV must contain 'comment_text' column")
            
            print(f"Loaded {len(data)} comments from {csv_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def extract_features(self, df, training=False):
        """Extract features for gossip verification focusing on relationship secrets"""
        print("Extracting features...")
        
        # Create a copy to avoid modifying the original dataframe
        feature_df = df.copy()
        
        # Basic text features
        feature_df['text_length'] = feature_df['comment_text'].apply(len)
        feature_df['word_count'] = feature_df['comment_text'].apply(lambda x: len(str(x).split()))
        
        # Check for specific details
        feature_df['has_specific_date'] = feature_df['comment_text'].apply(
            lambda x: bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b', str(x)))
        )
        feature_df['has_time'] = feature_df['comment_text'].apply(
            lambda x: bool(re.search(r'\b\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b|\b\d{1,2}\s*o\'clock\b', str(x)))
        )
        feature_df['has_location'] = feature_df['comment_text'].apply(
            lambda x: any(location.lower() in str(x).lower() for location in SL_LOCATIONS)
        )
        
        # Celebrity mentions
        feature_df['celebrity_mention_count'] = feature_df['comment_text'].apply(
            lambda x: sum(1 for celeb in SL_CELEBRITIES if celeb.lower() in str(x).lower())
        )
        feature_df['has_celebrity_mention'] = feature_df['celebrity_mention_count'] > 0
        
        # Company/Studio mentions - important for boss-employee relationships
        feature_df['company_mention_count'] = feature_df['comment_text'].apply(
            lambda x: sum(1 for company in SL_COMPANIES if company.lower() in str(x).lower())
        )
        feature_df['has_company_mention'] = feature_df['company_mention_count'] > 0
        
        # Privacy indicators
        feature_df['privacy_terms_count'] = feature_df['comment_text'].apply(
            lambda x: sum(1 for term in PRIVACY_TERMS if term.lower() in str(x).lower())
        )
        feature_df['has_privacy_terms'] = feature_df['privacy_terms_count'] > 0
        
        # Credibility signals
        feature_df['uses_firsthand_language'] = feature_df['comment_text'].apply(
            lambda x: bool(re.search(r'\bI saw\b|\bI heard\b|\bI was there\b|\bI witnessed\b|\bI know\b|\bmy friend\b|\bmy colleague\b', str(x)))
        )
        feature_df['contains_sensationalism'] = feature_df['comment_text'].apply(
            lambda x: any(term.lower() in str(x).lower() for term in SENSATIONALIST_TERMS)
        )
        
        # Check for relationship patterns between two entities
        feature_df['has_relationship_pattern'] = feature_df['comment_text'].apply(
            lambda x: bool(re.search(r'\b\w+\b.{1,30}\b(?:and|with)\b.{1,30}\b\w+\b', str(x)))
        )
        
        # Check for specific messaging platforms (common in sexting allegations)
        feature_df['mentions_messaging_platform'] = feature_df['comment_text'].apply(
            lambda x: bool(re.search(r'\b(?:WhatsApp|Messenger|Instagram|DM|text message|SMS|iMessage|Telegram|Signal|Snapchat|TikTok)\b', str(x), re.IGNORECASE))
        )
        
        # NLP-based features - only if available
        if spacy_available:
            print("Processing NLP features with spaCy...")
            # Applying spaCy can be slow for large datasets, sample if needed
            if len(feature_df) > 1000 and training:
                temp_df = feature_df.sample(1000, random_state=42)
                print(f"Sampling 1000 comments for NLP processing during training")
            else:
                temp_df = feature_df
                
            # Named entity recognition
            temp_df['named_entity_count'] = temp_df['comment_text'].apply(
                lambda x: len([ent for ent in nlp(str(x)[:1000]).ents]) if pd.notnull(x) else 0
            )
            
            # Only update the processed rows if we sampled
            if len(feature_df) > 1000 and training:
                feature_df.loc[temp_df.index, 'named_entity_count'] = temp_df['named_entity_count']
                # Fill missing values
                feature_df['named_entity_count'].fillna(0, inplace=True)
            else:
                feature_df['named_entity_count'] = temp_df['named_entity_count']
        else:
            print("SpaCy not available - skipping entity recognition")
            feature_df['named_entity_count'] = 0
        
        # Sentiment analysis - only if NLTK is available
        if nltk_available:
            print("Processing sentiment analysis with NLTK...")
            feature_df['sentiment_score'] = feature_df['comment_text'].apply(
                lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
            )
        else:
            print("NLTK not available - skipping sentiment analysis")
            feature_df['sentiment_score'] = 0
        
        # Categorize gossip types
        for category, terms in GOSSIP_CATEGORIES.items():
            feature_df[f'{category}_score'] = feature_df['comment_text'].apply(
                lambda x: sum(1 for term in terms if term.lower() in str(x).lower()) / len(terms)
            )
        
        # Extract numerical features for model training
        numeric_features = [
            'text_length', 'word_count', 'has_specific_date', 'has_time', 'has_location',
            'celebrity_mention_count', 'has_celebrity_mention', 'company_mention_count',
            'has_company_mention', 'privacy_terms_count', 'has_privacy_terms',
            'uses_firsthand_language', 'contains_sensationalism', 'has_relationship_pattern',
            'mentions_messaging_platform', 'named_entity_count', 'sentiment_score'
        ] + [f'{category}_score' for category in GOSSIP_CATEGORIES.keys()]
        
        self.feature_names = numeric_features
        
        # Handle text features separately with TF-IDF
        if training and self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=3000, 
                ngram_range=(1, 2),
                stop_words='english',
                min_df=3
            )
            text_features = self.vectorizer.fit_transform(feature_df['comment_text'].fillna(''))
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer must be trained before prediction")
            text_features = self.vectorizer.transform(feature_df['comment_text'].fillna(''))
        
        # Create a DataFrame for non-text features
        numeric_feature_df = feature_df[numeric_features].fillna(0)
        
        return text_features, numeric_feature_df
    
    def train_model(self, df, label_column='is_true'):
        """Train a model to identify true celebrity gossip"""
        if label_column not in df.columns:
            raise ValueError(f"Training data must contain label column '{label_column}'")
        
        print(f"Training model with {len(df)} labeled examples...")
        
        # Extract features
        text_features, numeric_features = self.extract_features(df, training=True)
        
        # Convert to numpy for model training
        numeric_array = numeric_features.values
        
        # Scale numeric features
        scaler = StandardScaler()
        numeric_array_scaled = scaler.fit_transform(numeric_array)
        
        # Combine text and numeric features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, numeric_array_scaled])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, df[label_column], test_size=0.3, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save preprocessing components
        self.scaler = scaler
        
        return self.model
    
    def predict(self, df):
        """Predict which comments are likely true"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Extract features
        text_features, numeric_features = self.extract_features(df)
        
        # Scale numeric features
        numeric_array = numeric_features.values
        numeric_array_scaled = self.scaler.transform(numeric_array)
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, numeric_array_scaled])
        
        # Make predictions
        predictions = self.model.predict(combined_features)
        probabilities = self.model.predict_proba(combined_features)[:, 1]
        
        # Add predictions to the original dataframe
        result_df = df.copy()
        result_df['predicted_true'] = predictions
        result_df['confidence_score'] = probabilities
        
        # Add category labels
        for category, terms in GOSSIP_CATEGORIES.items():
            result_df[f'is_{category}'] = result_df['comment_text'].apply(
                lambda x: any(term.lower() in str(x).lower() for term in terms)
            )
        
        return result_df
    
    def save_model(self, filepath='sl_celebrity_gossip_model.pkl'):
        """Save the trained model and preprocessing components"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath='sl_celebrity_gossip_model.pkl'):
        """Load a previously trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.feature_names = model_data['feature_names']
            self.scaler = model_data['scaler']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_comments(self, df, threshold=0.8):
        """Full analysis of comments with categorization"""
        # Predict true/false
        result_df = self.predict(df)
        
        # Apply confidence threshold
        result_df['likely_true'] = result_df['confidence_score'] >= threshold
        
        # Summarize results
        true_count = sum(result_df['likely_true'])
        total = len(result_df)
        
        print(f"\nAnalysis Complete:")
        print(f"Total comments: {total}")
        print(f"Likely true comments: {true_count} ({true_count/total*100:.1f}%)")
        
        # Category breakdown
        for category in GOSSIP_CATEGORIES.keys():
            cat_count = sum(result_df[f'is_{category}'])
            true_in_cat = sum(result_df[f'is_{category}'] & result_df['likely_true'])
            
            if cat_count > 0:
                print(f"{category.replace('_', '-').capitalize()} comments: {cat_count} ({cat_count/total*100:.1f}%)")
                print(f"  - Likely true: {true_in_cat} ({true_in_cat/cat_count*100:.1f}% of {category.replace('_', '-')})")
        
        return result_df
    
    def extract_relationship_pairs(self, df):
        """Extract potential relationship pairs from comments"""
        pairs = []
        
        # Function to find pairs of names in text
        def find_name_pairs(text):
            text = str(text)
            found_pairs = []
            
            # Check each celebrity against others
            for celeb1 in SL_CELEBRITIES:
                if celeb1.lower() in text.lower():
                    for celeb2 in SL_CELEBRITIES:
                        # Don't match the same person
                        if celeb1 != celeb2 and celeb2.lower() in text.lower():
                            # Check if they appear close to each other with relationship terms
                            celeb1_pos = text.lower().find(celeb1.lower())
                            celeb2_pos = text.lower().find(celeb2.lower())
                            
                            # Calculate the distance between mentions
                            distance = abs(celeb1_pos - celeb2_pos)
                            
                            # If they're mentioned within reasonable distance
                            if distance < 100:
                                # Check for relationship terms between them
                                between_text = text[min(celeb1_pos, celeb2_pos):max(celeb1_pos, celeb2_pos)]
                                relationship_terms = ["and", "with", "dating", "affair", "together", "relationship", "romance"]
                                
                                if any(term in between_text.lower() for term in relationship_terms):
                                    found_pairs.append((celeb1, celeb2))
            
            return found_pairs
        
        # Process each comment
        for idx, row in df.iterrows():
            text = row['comment_text']
            comment_pairs = find_name_pairs(text)
            
            for pair in comment_pairs:
                pairs.append({
                    'person1': pair[0],
                    'person2': pair[1],
                    'confidence_score': row.get('confidence_score', 0),
                    'likely_true': row.get('likely_true', False),
                    'comment': text
                })
        
        # Convert to DataFrame
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            return pairs_df
        else:
            return pd.DataFrame(columns=['person1', 'person2', 'confidence_score', 'likely_true', 'comment'])

    def detect_private_content_sharing(self, df):
        """Detect comments that suggest private content has been shared"""
        private_content_indicators = [
            "leaked", "private photos", "nude", "personal chat", "hacked", 
            "sexting", "WhatsApp", "screenshot", "personal messages", "private messages",
            "intimate photos", "bedroom photos", "shower photos", "exposed"
        ]
        
        def has_private_content_indicators(text):
            return any(indicator.lower() in str(text).lower() for indicator in private_content_indicators)
        
        # Filter comments that mention private content
        result_df = df.copy()
        result_df['suggests_private_content_shared'] = result_df['comment_text'].apply(has_private_content_indicators)
        
        # Filter high-risk comments (suggests private content + likely true)
        high_risk = result_df[result_df['suggests_private_content_shared'] & result_df.get('likely_true', False)]
        
        print(f"\nDetected {len(high_risk)} high-risk comments that may involve sharing of private content")
        
        return high_risk

def sample_labeled_data():
    """Create a sample dataset for demonstration purposes focusing on relationship gossip"""
    comments = [
        # True comments (more specific details, balanced language)
        "I saw Yohani at Odel in Colombo yesterday around 3pm. She was shopping with her friends.",
        "Bathiya and Santhush were performing at Nelum Pokuna on January 15. I was there and the crowd went wild!",
        "Jacqueline Fernandez visited her family in Colombo last weekend. My friend works at Cinnamon Grand and confirmed she stayed there.",
        "Angelo Mathews was having dinner at Taj Samudra on Friday night. He was with his family and was kind enough to take photos with fans.",
        "I met Sangakkara at the airport on Tuesday. He was very humble and signed an autograph for my son.",
        
        # Secret affair comments (potentially true)
        "My cousin works at Hilton Colombo and saw Dinakshie Priyasad and that new actor from Derana spending time together at the hotel restaurant three times last week. They seemed very close.",
        "I have a friend who works at Cinnamon Grand who told me they've reserved a private dining room for Santhush Weeraman and Umaria Sinhawansa twice this month. They're definitely seeing each other.",
        "My colleague spotted Stephanie Siriwardhana and Iraj having lunch at Water's Edge on Tuesday around 1pm. They were sitting in a corner and looked very intimate.",
        
        # Secret affair comments (likely exaggerated)
        "OMG Yohani is having a SECRET AFFAIR with Bathiya!! My friend's cousin saw them kissing behind the stage after the show!!",
        "BREAKING NEWS!!! Jacqueline and Namal are having a secret relationship! You won't believe how long this has been going on!",
        "Dinakshie Priyasad is CHEATING on her boyfriend with a married producer! This is absolutely SHOCKING!",
        
        # Sexting comments (potentially true)
        "My friend who works at Dialog saw Umaria Sinhawansa's private messages when her phone was being repaired last week. She's been exchanging flirty texts with someone from the industry.",
        "A technician I know had to recover data from Iraj's phone and noticed several personal exchanges with a female celebrity. Nothing explicit, but definitely flirtatious.",
        "I work in IT support and had to help a celebrity with their phone issues. I noticed they had several private conversations with fans that seemed inappropriate.",
        
        # Sexting comments (likely false)
        "OMG!! NUDE PHOTOS of Natasha Rathnayake have been LEAKED on WhatsApp groups! The scandal is INSANE!!",
        "Explicit messages between Sachini Nipunsala and that married actor have been leaked! SHOCKING CONTENT!!!",
        "You won't believe the sexting scandal involving Yohani! The messages are being shared in private groups!!",
        
        # Boss-employee comments (potentially true)
        "My sister works at Sirasa and mentioned that one of the directors has been spending a lot of time with a new presenter. They often have private meetings after hours.",
        "I have a friend at Derana who told me that one of the producers has been giving special treatment to a particular actress. They're often seen together outside work.",
        "A colleague who works at Hiru TV told me there's a relationship between a senior producer and one of the new anchors. It's not a secret among the staff.",
        
        # Boss-employee comments (likely exaggerated)
        "BREAKING: Famous director at Rupavahini is FORCING female employees into relationships for CAREER ADVANCEMENT!!! SHOCKING details inside!",
        "That big Hiru producer is having affairs with THREE different actresses at the same time! All to give them better roles! SCANDAL!",
        "Famous film director is EXPLOITING young actresses! The industry is CORRUPT and EVIL!",
        
        # Celebrity-fan comments (potentially true)
        "I know someone who's been exchanging private messages with Malinga on Instagram for months. He's actually very friendly and responds to her regularly.",
        "My cousin has been getting personal replies from Jacqueline on Instagram. She even wished her happy birthday in a private message last month.",
        "A friend of mine met Angelo Mathews at a private event and they exchanged numbers. They've been texting occasionally since then.",
        
        # Celebrity-fan comments (likely exaggerated)
        "SHOCKING! Famous cricket player is having SECRET RELATIONSHIPS with multiple fans! Sends them INAPPROPRIATE messages!",
        "Celebrity actress has been caught sending EXPLICIT photos to a teenage fan! DISGUSTING behavior!!",
        "Famous singer is OBSESSED with a fan and keeps sending them creepy messages! This is INSANE behavior!"
    ]
    
    # Create labels (first 5 generic true, next 9 potentially true relationship comments, rest are likely exaggerated)
    labels = [1, 1, 1, 1, 1] + [1, 1, 1] + [0, 0, 0] + [1, 1, 1] + [0, 0, 0] + [1, 1, 1] + [0, 0, 0] + [1, 1, 1] + [0, 0, 0]
    
    # Create DataFrame
    df = pd.DataFrame({
        'comment_text': comments,
        'is_true': labels
    })
    
    return df

# Main execution function
def main():
    print("Sri Lankan Celebrity Gossip Analyzer - Relationship Secrets Edition")
    print("=" * 60)
    
    analyzer = SriLankanCelebrityGossipAnalyzer()
    
    # Ask user if they want to train a new model or use existing
    mode = input("\nDo you want to train a new model or use an existing one? (train/use): ").strip().lower()
    
    model_folder = "model"

    if mode == 'train':
        # Ask for training data file
        train_file = input("Enter path to training data CSV file (or press Enter to use sample data): ").strip()
        
        if train_file:
            try:
                train_data = analyzer.load_data(train_file)
                if 'is_true' not in train_data.columns:
                    print("Warning: Training data must contain 'is_true' column. Adding sample labels...")
                    # Add random labels for demonstration
                    import numpy as np
                    np.random.seed(42)
                    train_data['is_true'] = np.random.randint(0, 2, size=len(train_data))
            except Exception as e:
                print(f"Error loading training file: {e}")
                print("Using sample data instead.")
                train_data = sample_labeled_data()
        else:
            print("Using sample data for training...")
            train_data = sample_labeled_data()
        
        # Train the model
        print("\nTraining model...")
        analyzer.train_model(train_data)
        
        # Save the model
        model_path = input("Enter path to save model (or press Enter for default 'model/sl_celebrity_gossip_model.pkl'): ").strip()
        if not model_path:
            os.makedirs(model_folder, exist_ok=True)
            model_path = os.path.join(model_folder, 'sl_celebrity_gossip_model.pkl')
        analyzer.save_model(model_path)
        print(f"Model saved to {model_path}")
        
    else:  # Use existing model
        # Load the model
        model_path = input("Enter path to model file (or press Enter for default 'model/sl_celebrity_gossip_model.pkl'): ").strip()
        if not model_path:
            model_path = os.path.join(model_folder, 'sl_celebrity_gossip_model.pkl')
        
        try:
            success = analyzer.load_model(model_path)
            if not success:
                print("Could not load model. Please train a new model first.")
                return
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a new model first.")
            return
    

    output_folder = "output"
    input_folder = "input"

    # Ask for data to analyze
    analyze_file = input("\nEnter path to data CSV for analysis (or press Enter for 'input/tiktok_comments_final.csv'): ").strip()
    if not analyze_file:
        os.makedirs(input_folder, exist_ok=True)
        analyze_file = os.path.join(input_folder, 'tiktok_comments_final.csv')
    
    try:
        data = analyzer.load_data(analyze_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nAnalyzing data from {analyze_file}...")
    
    # Analyze the data
    results = analyzer.analyze_comments(data)
    
    # Save results
    output_file = input("\nEnter path to save results (or press Enter for 'output/analyzed_results.csv'): ").strip()
    if not output_file:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'analyzed_results.csv')
    
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Show some example results
    print("\nExample Results (Top 5):")
    columns_to_show = ['comment_text', 'likely_true', 'confidence_score'] + [f'is_{cat}' for cat in GOSSIP_CATEGORIES.keys()]
    print(results[columns_to_show].head())
    
    # Example of relationship extraction
    print("\nExtracting potential relationship pairs...")
    relationship_pairs = analyzer.extract_relationship_pairs(results)
    if not relationship_pairs.empty:
        print(f"Found {len(relationship_pairs)} potential relationship pairs")
        print("\nTop relationships by confidence:")
        print(relationship_pairs.sort_values('confidence_score', ascending=False)[['person1', 'person2', 'confidence_score', 'likely_true']].head())
        
        # Save relationship pairs
        pairs_file = input("\nEnter path to save relationship pairs (or press Enter for 'output/relationship_pairs.csv'): ").strip()
        if not pairs_file:
            os.makedirs(output_folder, exist_ok=True)
            pairs_file = os.path.join(output_folder, 'relationship_pairs.csv')
        relationship_pairs.to_csv(pairs_file, index=False)
        print(f"Relationship pairs saved to {pairs_file}")
    else:
        print("No relationship pairs detected in data")
    
    # Example of private content detection
    print("\nDetecting possible private content sharing...")
    private_content = analyzer.detect_private_content_sharing(results)
    if not private_content.empty:
        print(f"Found {len(private_content)} comments suggesting private content sharing")
        print("\nMost concerning (by confidence):")
        print(private_content.sort_values('confidence_score', ascending=False)[['comment_text', 'confidence_score']].head())
        
        # Save private content findings
        private_file = input("\nEnter path to save private content findings (or press Enter for 'output/private_content_findings.csv'): ").strip()
        if not private_file:
            os.makedirs(output_folder, exist_ok=True)
            private_file = os.path.join(output_folder, 'private_content_findings.csv')
        
        private_content.to_csv(private_file, index=False)
        print(f"Private content findings saved to {private_file}")
    else:
        print("No high-risk private content sharing detected in data")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()