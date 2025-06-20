pip install --upgrade pip
pip install scipy==1.10.1
pip install protobuf==3.20.0
pip install stanza
pip install rapidfuzz
pip install matplotlib
python -c 'import stanza; stanza.download("en")'
pip install -U git+https://github.com/recski/tuw-nlp.git@dev_newpotato
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm