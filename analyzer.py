import pandas as pd
import re
import os
import numpy as np
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

from sl_celebrity_data import (
    SL_CELEBRITIES, SL_CELEBRITY_NICKNAMES, SL_COMPANIES, SL_LOCATIONS,
    SENSATIONALIST_TERMS, GOSSIP_CATEGORIES, PRIVACY_TERMS, 
    RELATIONSHIP_PROXIMITY_TERMS, TIME_PATTERNS, DATE_PATTERNS,
    CREDIBILITY_INDICATORS, MESSAGING_PLATFORMS, RELATIONSHIP_PATTERNS,
    LOCATION_PATTERNS, SENTIMENT_WORDS, EVENT_TYPES
)

class SriLankanCelebrityGossipAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.scaler = None
    
    def load_data(self, csv_path):
        """Load and prepare data from CSV file"""
        try:
            data = pd.read_csv(csv_path)
            # Check for required column
            if 'Comment' not in data.columns:
                raise ValueError("CSV must contain 'Comment' column")
            
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
        feature_df['text_length'] = feature_df['Comment'].apply(len)
        feature_df['word_count'] = feature_df['Comment'].apply(lambda x: len(str(x).split()))
        
        # Check for specific dates using enhanced date patterns
        feature_df['has_specific_date'] = feature_df['Comment'].apply(
            lambda x: any(bool(re.search(pattern, str(x), re.IGNORECASE)) for pattern in DATE_PATTERNS)
        )
        
        # Check for specific times using enhanced time patterns
        feature_df['has_time'] = feature_df['Comment'].apply(
            lambda x: any(bool(re.search(pattern, str(x), re.IGNORECASE)) for pattern in TIME_PATTERNS)
        )
        
        # Location detection
        feature_df['has_location'] = feature_df['Comment'].apply(
            lambda x: any(location.lower() in str(x).lower() for location in SL_LOCATIONS) or 
                     any(bool(re.search(pattern, str(x), re.IGNORECASE)) for pattern in LOCATION_PATTERNS)
        )
        
        # Celebrity mentions including nickname resolution
        def count_celebrity_mentions(text):
            text = str(text).lower()
            direct_mentions = sum(1 for celeb in SL_CELEBRITIES if celeb.lower() in text)
            
            # Check for nicknames and count them
            nickname_mentions = 0
            for nickname, full_name in SL_CELEBRITY_NICKNAMES.items():
                if nickname.lower() in text:
                    nickname_mentions += 1
            
            return direct_mentions + nickname_mentions
        
        feature_df['celebrity_mention_count'] = feature_df['Comment'].apply(count_celebrity_mentions)
        feature_df['has_celebrity_mention'] = feature_df['celebrity_mention_count'] > 0
        
        # Company/Studio mentions - important for boss-employee relationships
        feature_df['company_mention_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for company in SL_COMPANIES if company.lower() in str(x).lower())
        )
        feature_df['has_company_mention'] = feature_df['company_mention_count'] > 0
        
        # Privacy indicators using enhanced PRIVACY_TERMS
        feature_df['privacy_terms_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in PRIVACY_TERMS if term.lower() in str(x).lower())
        )
        feature_df['has_privacy_terms'] = feature_df['privacy_terms_count'] > 0
        
        # Credibility signals using enhanced CREDIBILITY_INDICATORS
        feature_df['firsthand_language_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in CREDIBILITY_INDICATORS['firsthand'] if term.lower() in str(x).lower())
        )
        feature_df['hearsay_language_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in CREDIBILITY_INDICATORS['hearsay'] if term.lower() in str(x).lower())
        )
        feature_df['speculation_language_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in CREDIBILITY_INDICATORS['speculation'] if term.lower() in str(x).lower())
        )
        
        feature_df['credibility_score'] = feature_df.apply(
            lambda row: (row['firsthand_language_count'] - row['hearsay_language_count'] - 
                         0.5 * row['speculation_language_count']), axis=1
        )
        
        # Sensationalism using enhanced SENSATIONALIST_TERMS
        feature_df['sensationalism_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in SENSATIONALIST_TERMS if term.lower() in str(x).lower())
        )
        feature_df['contains_sensationalism'] = feature_df['sensationalism_count'] > 0
        
        # Check for relationship patterns between two entities using enhanced RELATIONSHIP_PATTERNS
        feature_df['has_relationship_pattern'] = feature_df['Comment'].apply(
            lambda x: any(bool(re.search(pattern, str(x), re.IGNORECASE)) for pattern in RELATIONSHIP_PATTERNS)
        )
        
        # Check for relationship proximity terms
        feature_df['relationship_proximity_score'] = feature_df['Comment'].apply(
            lambda x: sum(1 for term in RELATIONSHIP_PROXIMITY_TERMS if term.lower() in str(x).lower()) / 
                      len(RELATIONSHIP_PROXIMITY_TERMS)
        )
        
        # Check for specific messaging platforms (common in sexting allegations)
        feature_df['messaging_platform_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for platform in MESSAGING_PLATFORMS if platform.lower() in str(x).lower())
        )
        feature_df['mentions_messaging_platform'] = feature_df['messaging_platform_count'] > 0
        
        # Event type detection
        feature_df['event_type_count'] = feature_df['Comment'].apply(
            lambda x: sum(1 for event_type in EVENT_TYPES if event_type.lower() in str(x).lower())
        )
        feature_df['mentions_event'] = feature_df['event_type_count'] > 0
        
        # Sentiment analysis using built-in dictionaries if NLTK is not available
        if not nltk_available:
            def basic_sentiment_score(text):
                text = str(text).lower()
                positive_count = sum(1 for term in SENTIMENT_WORDS['positive'] if term.lower() in text)
                negative_count = sum(1 for term in SENTIMENT_WORDS['negative'] if term.lower() in text)
                
                total_words = len(text.split())
                if total_words == 0:
                    return 0
                
                # Return normalized difference between positive and negative
                return (positive_count - negative_count) / (positive_count + negative_count + 1)
            
            feature_df['sentiment_score'] = feature_df['Comment'].apply(basic_sentiment_score)
        else:
            print("Processing sentiment analysis with NLTK...")
            feature_df['sentiment_score'] = feature_df['Comment'].apply(
                lambda x: sia.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0
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
            temp_df['named_entity_count'] = temp_df['Comment'].apply(
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
        
        # Categorize gossip types with more detailed scoring
        for category, terms in GOSSIP_CATEGORIES.items():
            # Count raw term occurrences
            feature_df[f'{category}_term_count'] = feature_df['Comment'].apply(
                lambda x: sum(1 for term in terms if term.lower() in str(x).lower())
            )
            
            # Calculate normalized score (0-1 range)
            feature_df[f'{category}_score'] = feature_df[f'{category}_term_count'] / max(len(terms) / 3, 1)
            # Cap at 1.0
            feature_df[f'{category}_score'] = feature_df[f'{category}_score'].clip(upper=1.0)
        
        # Extract numerical features for model training
        numeric_features = [
            'text_length', 'word_count', 'has_specific_date', 'has_time', 'has_location',
            'celebrity_mention_count', 'has_celebrity_mention', 'company_mention_count',
            'has_company_mention', 'privacy_terms_count', 'has_privacy_terms',
            'firsthand_language_count', 'hearsay_language_count', 'speculation_language_count',
            'credibility_score', 'sensationalism_count', 'contains_sensationalism', 
            'has_relationship_pattern', 'relationship_proximity_score',
            'messaging_platform_count', 'mentions_messaging_platform',
            'event_type_count', 'mentions_event', 'named_entity_count', 'sentiment_score'
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
            text_features = self.vectorizer.fit_transform(feature_df['Comment'].fillna(''))
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer must be trained before prediction")
            text_features = self.vectorizer.transform(feature_df['Comment'].fillna(''))
        
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
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            # Get text feature names
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                text_feature_names = self.vectorizer.get_feature_names_out()
            else:
                text_feature_names = self.vectorizer.get_feature_names()
                
            # Combine with numeric feature names
            all_feature_names = list(text_feature_names) + list(self.feature_names)
            
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            print("\nTop 20 most important features:")
            for i in range(min(20, len(indices))):
                if indices[i] < len(text_feature_names):
                    print(f"{all_feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
                else:
                    print(f"{all_feature_names[indices[i]]} (numeric): {importances[indices[i]]:.4f}")
        
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
            result_df[f'is_{category}'] = result_df['Comment'].apply(
                lambda x: any(term.lower() in str(x).lower() for term in terms)
            )
        
        return result_df
    
    def save_model(self, filepath):
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
    
    def load_model(self, filepath):
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
        """Extract potential relationship pairs from comments with enhanced detection"""
        pairs = []
        
        # Function to find pairs of names in text
        def find_name_pairs(text, comment_score=0, row_id=None):
            text = str(text)
            found_pairs = []
            
            # Check for direct matches using enhanced RELATIONSHIP_PATTERNS
            for pattern in RELATIONSHIP_PATTERNS:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract the potential person names from the match
                    person1 = match.group(1)
                    person2 = match.group(2)
                    
                    # Verify if these are actual celebrities in our list
                    celeb1 = next((c for c in SL_CELEBRITIES if c.lower() in person1.lower()), None)
                    celeb2 = next((c for c in SL_CELEBRITIES if c.lower() in person2.lower()), None)
                    
                    # If both are celebrities, add the pair
                    if celeb1 and celeb2 and celeb1 != celeb2:
                        context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
                        found_pairs.append((celeb1, celeb2, context))
            
            # If no pairs found through patterns, use the original approach
            if not found_pairs:
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
                                    # Get context around the names
                                    context_start = max(0, min(celeb1_pos, celeb2_pos) - 50)
                                    context_end = min(len(text), max(celeb1_pos, celeb2_pos) + 50)
                                    context = text[context_start:context_end]
                                    
                                    # Check for relationship terms between them
                                    between_text = text[min(celeb1_pos, celeb2_pos):max(celeb1_pos, celeb2_pos)]
                                    
                                    # Check for relationship proximity terms
                                    if any(term.lower() in between_text.lower() for term in RELATIONSHIP_PROXIMITY_TERMS):
                                        found_pairs.append((celeb1, celeb2, context))
            
            return found_pairs
        
        # Process each comment
        for idx, row in df.iterrows():
            text = row['Comment']
            comment_pairs = find_name_pairs(text, row.get('confidence_score', 0), idx)
            
            for pair in comment_pairs:
                celeb1, celeb2, context = pair
                pairs.append({
                    'person1': celeb1,
                    'person2': celeb2,
                    'context': context,
                    'confidence_score': row.get('confidence_score', 0),
                    'likely_true': row.get('likely_true', False),
                    'comment_id': idx,
                    'comment': text
                })
        
        # Convert to DataFrame
        if pairs:
            pairs_df = pd.DataFrame(pairs)
            return pairs_df
        else:
            return pd.DataFrame(columns=['person1', 'person2', 'context', 'confidence_score', 'likely_true', 'comment_id', 'comment'])

    def detect_private_content_sharing(self, df):
        """
        Enhanced detection of comments that suggest private content has been shared
        using the expanded privacy terms
        """
        # Use the enhanced PRIVACY_TERMS combined with GOSSIP_CATEGORIES['sexting']
        private_content_indicators = PRIVACY_TERMS + GOSSIP_CATEGORIES['sexting']
        
        def private_content_score(text):
            text = str(text).lower()
            matches = sum(1 for indicator in private_content_indicators if indicator.lower() in text)
            # Normalize score
            return min(matches / 5, 1.0)  # Cap at 1.0
        
        # Filter comments that mention private content
        result_df = df.copy()
        result_df['private_content_score'] = result_df['Comment'].apply(private_content_score)
        result_df['suggests_private_content_shared'] = result_df['private_content_score'] > 0.2
        
        # Check for specific messaging platforms
        result_df['mentions_messaging_platform'] = result_df['Comment'].apply(
            lambda x: any(platform.lower() in str(x).lower() for platform in MESSAGING_PLATFORMS)
        )
        
        # Combined risk score
        result_df['privacy_risk_score'] = result_df.apply(
            lambda row: row['private_content_score'] * (1.5 if row['mentions_messaging_platform'] else 1.0), 
            axis=1
        )
        
        # Filter high-risk comments (suggests private content + likely true)
        high_risk = result_df[
            (result_df['privacy_risk_score'] > 0.4) & 
            result_df.get('likely_true', pd.Series([True] * len(result_df)))
        ]
        
        # Sort by risk score
        high_risk = high_risk.sort_values('privacy_risk_score', ascending=False)
        
        print(f"\nDetected {len(high_risk)} high-risk comments that may involve sharing of private content")
        
        return high_risk
    
    def analyze_celebrity_network(self, pairs_df, min_confidence=0.5):
        """
        Analyze the network of relationship connections between celebrities
        based on extracted pairs
        """
        # Filter for pairs with sufficient confidence
        filtered_pairs = pairs_df[pairs_df['confidence_score'] >= min_confidence]
        
        # Count co-mentions for each pair
        pair_counts = {}
        for _, row in filtered_pairs.iterrows():
            # Ensure consistent ordering of pairs
            pair = tuple(sorted([row['person1'], row['person2']]))
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1
        
        # Convert to DataFrame
        network_df = pd.DataFrame([
            {'person1': pair[0], 'person2': pair[1], 'mentions': count}
            for pair, count in pair_counts.items()
        ])
        
        # Sort by mention count
        if not network_df.empty:
            network_df = network_df.sort_values('mentions', ascending=False)
            
            print("\nCelebrity Relationship Network Analysis:")
            print(f"Found {len(network_df)} potential relationship connections")
            print("\nTop relationship connections:")
            for i, row in network_df.head(10).iterrows():
                print(f"{row['person1']} - {row['person2']}: {row['mentions']} mentions")
            
            return network_df
        else:
            print("\nNo significant relationship connections found")
            return pd.DataFrame(columns=['person1', 'person2', 'mentions'])
    
    def identify_trending_gossip(self, df, threshold=0.7):
        """
        Identify trending gossip topics and most mentioned celebrities
        """
        # Filter for high-confidence comments
        high_conf = df[df['confidence_score'] >= threshold].copy()
        
        # Count celebrity mentions
        celeb_counts = {}
        for _, row in high_conf.iterrows():
            comment = str(row['Comment']).lower()
            for celeb in SL_CELEBRITIES:
                if celeb.lower() in comment:
                    if celeb in celeb_counts:
                        celeb_counts[celeb] += 1
                    else:
                        celeb_counts[celeb] = 1
            
            # Also check nicknames
            for nickname, full_name in SL_CELEBRITY_NICKNAMES.items():
                if nickname.lower() in comment:
                    if full_name in celeb_counts:
                        celeb_counts[full_name] += 1
                    else:
                        celeb_counts[full_name] = 1
        
        # Convert to DataFrame
        celeb_df = pd.DataFrame([
            {'celebrity': celeb, 'mentions': count}
            for celeb, count in celeb_counts.items()
        ])
        
        # Sort by mention count
        if not celeb_df.empty:
            celeb_df = celeb_df.sort_values('mentions', ascending=False)
            
            # Category analysis for each celebrity
            top_celebs = celeb_df.head(10)['celebrity'].tolist()
            celeb_categories = {}
            
            for celeb in top_celebs:
                celeb_comments = high_conf[high_conf['Comment'].str.lower().str.contains(celeb.lower())]
                category_counts = {}
                
                for category in GOSSIP_CATEGORIES.keys():
                    cat_count = sum(celeb_comments[f'is_{category}']) if f'is_{category}' in celeb_comments.columns else 0
                    if cat_count > 0:
                        category_counts[category] = cat_count
                
                celeb_categories[celeb] = category_counts
            
            print("\nTrending Celebrity Gossip Analysis:")
            print(f"Analyzed {len(high_conf)} high-confidence comments")
            print("\nMost mentioned celebrities:")
            for i, row in top_celebs[:10]:
                print(f"{i+1}. {row['celebrity']}: {row['mentions']} mentions")
                if row['celebrity'] in celeb_categories:
                    cat_str = ", ".join([f"{cat.replace('_', ' ')}: {count}" 
                                       for cat, count in celeb_categories[row['celebrity']].items()])
                    print(f"   Topics: {cat_str}")
            
            return celeb_df, celeb_categories
        else:
            print("\nNo significant celebrity mentions found")
            return pd.DataFrame(columns=['celebrity', 'mentions']), {}
        
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
        'Comment': comments,
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
                    rng = np.random.default_rng(seed=42)  # create a random number generator
                    train_data['is_true'] = rng.integers(0, 2, size=len(train_data))
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
    columns_to_show = ['Comment', 'likely_true', 'confidence_score'] + [f'is_{cat}' for cat in GOSSIP_CATEGORIES.keys()]
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
        print(private_content.sort_values('confidence_score', ascending=False)[['Comment', 'confidence_score']].head())
        
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