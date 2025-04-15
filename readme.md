pip3 install numpy==1.26.4
pip3 install scipy==1.12.0
pip3 install scikit-learn==1.4.1.post1

# Downgrade spaCy and thinc to be compatible with numpy < 2.0
pip3 install spacy==3.7.2 thinc==8.2.2

# Then download the model
python3 -m spacy download en_core_web_sm

pip3 install pandas nltk openpyxl
