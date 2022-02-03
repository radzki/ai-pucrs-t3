#!/bin/bash
export TREETAGGER_HOME='/home/radzki/Documents/Faculdade/Inteligencia_Artificial/T3/TreeTagger/'
python run.py weka Testes/not_mergeado_sem_hashtag.norm; mv out.arff not_mergeado_sem_hashtag.arff; mv not_mergeado_sem_hashtag.arff Testes/
python run.py weka Testes/not_mergeado_com_hashtag.norm; mv out.arff not_mergeado_com_hashtag.arff; mv not_mergeado_com_hashtag.arff Testes/
python run.py weka Testes/com_not_sem_hashtag.norm; mv out.arff com_not_sem_hashtag.arff; mv com_not_sem_hashtag.arff Testes/
python run.py weka Testes/com_not_com_hashtag.norm; mv out.arff com_not_com_hashtag.arff; mv com_not_com_hashtag.arff Testes/
