#!/bin/bash
python deleteModel.py
python bmsTrain.py walter_data/4-10 1 0.00 1.00
python bmsPredict.py walter_data/4-11 1.00 predictions_walter/walter_on_walter_4-11.txt
python bmsPredict.py stephen_data/4-11 1.00 predictions_walter/stephen_on_walter_4-11.txt
python bmsPredict.py arun_data/4-11 1.00 predictions_walter/arun_on_walter_4-11.txt
python bmsPredict.py arthur_data/4-11 1.00 predictions_walter/arthur_on_walter_4-11.txt

python bmsTrain.py walter_data/4-11 1 0.00 1.00
python bmsPredict.py walter_data/4-12 1.00 predictions_walter/walter_on_walter_4-12.txt
python bmsPredict.py stephen_data/4-12 1.00 predictions_walter/stephen_on_walter_4-12.txt
python bmsPredict.py arun_data/4-12 1.00 predictions_walter/arun_on_walter_4-12.txt
python bmsPredict.py arthur_data/4-12 1.00 predictions_walter/arthur_on_walter_4-12.txt

python bmsTrain.py walter_data/4-12 1 0.00 1.00
python bmsPredict.py walter_data/4-13 1.00 predictions_walter/walter_on_walter_4-13.txt
python bmsPredict.py stephen_data/4-13 1.00 predictions_walter/stephen_on_walter_4-13.txt
python bmsPredict.py arun_data/4-13 1.00 predictions_walter/arun_on_walter_4-13.txt
python bmsPredict.py arthur_data/4-13 1.00 predictions_walter/arthur_on_walter_4-13.txt

python bmsTrain.py walter_data/4-13 1 0.00 1.00
python bmsPredict.py walter_data/4-14 1.00 predictions_walter/walter_on_walter_4-14.txt
python bmsPredict.py stephen_data/4-14 1.00 predictions_walter/stephen_on_walter_4-14.txt
python bmsPredict.py arun_data/4-14 1.00 predictions_walter/arun_on_walter_4-14.txt
python bmsPredict.py arthur_data/4-14 1.00 predictions_walter/arthur_on_walter_4-14.txt

python bmsTrain.py walter_data/4-14 1 0.00 1.00
python bmsPredict.py walter_data/4-15 1.00 predictions_walter/walter_on_walter_4-15.txt
python bmsPredict.py stephen_data/4-15 1.00 predictions_walter/stephen_on_walter_4-15.txt
python bmsPredict.py arun_data/4-15 1.00 predictions_walter/arun_on_walter_4-15.txt
python bmsPredict.py arthur_data/4-15 1.00 predictions_walter/arthur_on_walter_4-15.txt

python bmsTrain.py walter_data/4-15 1 0.00 1.00
python bmsPredict.py walter_data/4-16 1.00 predictions_walter/walter_on_walter_4-16.txt
python bmsPredict.py stephen_data/4-16 1.00 predictions_walter/stephen_on_walter_4-16.txt
python bmsPredict.py arun_data/4-16 1.00 predictions_walter/arun_on_walter_4-16.txt
python bmsPredict.py arthur_data/4-16 1.00 predictions_walter/arthur_on_walter_4-16.txt

python bmsTrain.py walter_data/4-16 1 0.00 1.00
python bmsPredict.py walter_data/4-17 1.00 predictions_walter/walter_on_walter_4-17.txt
python bmsPredict.py stephen_data/4-17 1.00 predictions_walter/stephen_on_walter_4-17.txt
python bmsPredict.py arun_data/4-17 1.00 predictions_walter/arun_on_walter_4-17.txt
python bmsPredict.py arthur_data/4-17 1.00 predictions_walter/arthur_on_walter_4-17.txt

python bmsTrain.py walter_data/4-17 1 0.00 1.00
python bmsPredict.py walter_data/4-18 1.00 predictions_walter/walter_on_walter_4-18.txt
python bmsPredict.py stephen_data/4-18 1.00 predictions_walter/stephen_on_walter_4-18.txt
python bmsPredict.py arun_data/4-18 1.00 predictions_walter/arun_on_walter_4-18.txt
python bmsPredict.py arthur_data/4-18 1.00 predictions_walter/arthur_on_walter_4-18.txt

python bmsTrain.py walter_data/4-18 1 0.00 1.00
python bmsPredict.py walter_data/4-19 1.00 predictions_walter/walter_on_walter_4-19.txt
python bmsPredict.py stephen_data/4-19 1.00 predictions_walter/stephen_on_walter_4-19.txt
python bmsPredict.py arun_data/4-19 1.00 predictions_walter/arun_on_walter_4-19.txt
python bmsPredict.py arthur_data/4-19 1.00 predictions_walter/arthur_on_walter_4-19.txt

python bmsTrain.py walter_data/4-19 1 0.00 1.00
python bmsPredict.py walter_data/4-20 1.00 predictions_walter/walter_on_walter_4-20.txt
python bmsPredict.py stephen_data/4-20 1.00 predictions_walter/stephen_on_walter_4-20.txt
python bmsPredict.py arun_data/4-20 1.00 predictions_walter/arun_on_walter_4-20.txt
python bmsPredict.py arthur_data/4-20 1.00 predictions_walter/arthur_on_walter_4-20.txt

python bmsTrain.py walter_data/4-20 1 0.00 1.00
python bmsPredict.py walter_data/4-21 1.00 predictions_walter/walter_on_walter_4-21.txt
python bmsPredict.py stephen_data/4-21 1.00 predictions_walter/stephen_on_walter_4-21.txt
python bmsPredict.py arun_data/4-21 1.00 predictions_walter/arun_on_walter_4-21.txt
python bmsPredict.py arthur_data/4-21 1.00 predictions_walter/arthur_on_walter_4-21.txt

python bmsTrain.py walter_data/4-21 1 0.00 1.00
python bmsPredict.py walter_data/4-22 1.00 predictions_walter/walter_on_walter_4-22.txt
python bmsPredict.py stephen_data/4-22 1.00 predictions_walter/stephen_on_walter_4-22.txt
python bmsPredict.py arthur_data/4-22 1.00 predictions_walter/arthur_on_walter_4-22.txt
