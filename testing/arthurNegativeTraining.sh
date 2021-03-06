#!/bin/bash
python deleteModel.py

python bmsTrain.py arun_data/4-08 0 0.00 1.00

python bmsTrain.py stephen_data/4-09 0 0.00 1.00
python bmsTrain.py arun_data/4-09 0 0.00 1.00
python bmsTrain.py arthur_data/4-09 1 0.00 1.00
python bmsPredict.py arthur_data/4-10 1.00 predictions_arthur_cross/arthur_on_arthur_4-10.txt
python bmsPredict.py stephen_data/4-10 1.00 predictions_arthur_cross/stephen_on_arthur_4-10.txt
python bmsPredict.py arun_data/4-10 1.00 predictions_arthur_cross/arun_on_arthur_4-10.txt
python bmsPredict.py walter_data/4-10 1.00 predictions_arthur_cross/walter_on_arthur_4-10.txt

python bmsTrain.py stephen_data/4-10 0 0.00 1.00
python bmsTrain.py arun_data/4-10 0 0.00 1.00
python bmsTrain.py walter_data/4-10 0 0.00 1.00
python bmsTrain.py arthur_data/4-10 1 0.00 1.00
python bmsPredict.py arthur_data/4-11 1.00 predictions_arthur_cross/arthur_on_arthur_4-11.txt
python bmsPredict.py stephen_data/4-11 1.00 predictions_arthur_cross/stephen_on_arthur_4-11.txt
python bmsPredict.py arun_data/4-11 1.00 predictions_arthur_cross/arun_on_arthur_4-11.txt
python bmsPredict.py walter_data/4-11 1.00 predictions_arthur_cross/walter_on_arthur_4-11.txt

python bmsTrain.py stephen_data/4-11 0 0.00 1.00
python bmsTrain.py arun_data/4-11 0 0.00 1.00
python bmsTrain.py walter_data/4-11 0 0.00 1.00
python bmsTrain.py arthur_data/4-11 1 0.00 1.00
python bmsPredict.py arthur_data/4-12 1.00 predictions_arthur_cross/arthur_on_arthur_4-12.txt
python bmsPredict.py stephen_data/4-12 1.00 predictions_arthur_cross/stephen_on_arthur_4-12.txt
python bmsPredict.py arun_data/4-12 1.00 predictions_arthur_cross/arun_on_arthur_4-12.txt
python bmsPredict.py walter_data/4-12 1.00 predictions_arthur_cross/walter_on_arthur_4-12.txt

python bmsTrain.py stephen_data/4-12 0 0.00 1.00
python bmsTrain.py arun_data/4-12 0 0.00 1.00
python bmsTrain.py walter_data/4-12 0 0.00 1.00
python bmsTrain.py arthur_data/4-12 1 0.00 1.00
python bmsPredict.py arthur_data/4-13 1.00 predictions_arthur_cross/arthur_on_arthur_4-13.txt
python bmsPredict.py stephen_data/4-13 1.00 predictions_arthur_cross/stephen_on_arthur_4-13.txt
python bmsPredict.py arun_data/4-13 1.00 predictions_arthur_cross/arun_on_arthur_4-13.txt
python bmsPredict.py walter_data/4-13 1.00 predictions_arthur_cross/walter_on_arthur_4-13.txt

python bmsTrain.py stephen_data/4-13 0 0.00 1.00
python bmsTrain.py arun_data/4-13 0 0.00 1.00
python bmsTrain.py walter_data/4-13 0 0.00 1.00
python bmsTrain.py arthur_data/4-13 1 0.00 1.00
python bmsPredict.py arthur_data/4-14 1.00 predictions_arthur_cross/arthur_on_arthur_4-14.txt
python bmsPredict.py stephen_data/4-14 1.00 predictions_arthur_cross/stephen_on_arthur_4-14.txt
python bmsPredict.py arun_data/4-14 1.00 predictions_arthur_cross/arun_on_arthur_4-14.txt
python bmsPredict.py walter_data/4-14 1.00 predictions_arthur_cross/walter_on_arthur_4-14.txt

python bmsTrain.py stephen_data/4-14 0 0.00 1.00
python bmsTrain.py arun_data/4-14 0 0.00 1.00
python bmsTrain.py walter_data/4-14 0 0.00 1.00
python bmsTrain.py arthur_data/4-14 1 0.00 1.00
python bmsPredict.py arthur_data/4-15 1.00 predictions_arthur_cross/arthur_on_arthur_4-15.txt
python bmsPredict.py stephen_data/4-15 1.00 predictions_arthur_cross/stephen_on_arthur_4-15.txt
python bmsPredict.py arun_data/4-15 1.00 predictions_arthur_cross/arun_on_arthur_4-15.txt
python bmsPredict.py walter_data/4-15 1.00 predictions_arthur_cross/walter_on_arthur_4-15.txt

python bmsTrain.py stephen_data/4-15 0 0.00 1.00
python bmsTrain.py arun_data/4-15 0 0.00 1.00
python bmsTrain.py walter_data/4-15 0 0.00 1.00
python bmsTrain.py arthur_data/4-15 1 0.00 1.00
python bmsPredict.py arthur_data/4-16 1.00 predictions_arthur_cross/arthur_on_arthur_4-16.txt
python bmsPredict.py stephen_data/4-16 1.00 predictions_arthur_cross/stephen_on_arthur_4-16.txt
python bmsPredict.py arun_data/4-16 1.00 predictions_arthur_cross/arun_on_arthur_4-16.txt
python bmsPredict.py walter_data/4-16 1.00 predictions_arthur_cross/walter_on_arthur_4-16.txt

python bmsTrain.py stephen_data/4-16 0 0.00 1.00
python bmsTrain.py arun_data/4-16 0 0.00 1.00
python bmsTrain.py walter_data/4-16 0 0.00 1.00
python bmsTrain.py arthur_data/4-16 1 0.00 1.00
python bmsPredict.py arthur_data/4-17 1.00 predictions_arthur_cross/arthur_on_arthur_4-17.txt
python bmsPredict.py stephen_data/4-17 1.00 predictions_arthur_cross/stephen_on_arthur_4-17.txt
python bmsPredict.py arun_data/4-17 1.00 predictions_arthur_cross/arun_on_arthur_4-17.txt
python bmsPredict.py walter_data/4-17 1.00 predictions_arthur_cross/walter_on_arthur_4-17.txt

python bmsTrain.py stephen_data/4-17 0 0.00 1.00
python bmsTrain.py arun_data/4-17 0 0.00 1.00
python bmsTrain.py walter_data/4-17 0 0.00 1.00
python bmsTrain.py arthur_data/4-17 1 0.00 1.00
python bmsPredict.py arthur_data/4-18 1.00 predictions_arthur_cross/arthur_on_arthur_4-18.txt
python bmsPredict.py stephen_data/4-18 1.00 predictions_arthur_cross/stephen_on_arthur_4-18.txt
python bmsPredict.py arun_data/4-18 1.00 predictions_arthur_cross/arun_on_arthur_4-18.txt
python bmsPredict.py walter_data/4-18 1.00 predictions_arthur_cross/walter_on_arthur_4-18.txt

python bmsTrain.py stephen_data/4-18 0 0.00 1.00
python bmsTrain.py arun_data/4-18 0 0.00 1.00
python bmsTrain.py walter_data/4-18 0 0.00 1.00
python bmsTrain.py arthur_data/4-18 1 0.00 1.00
python bmsPredict.py arthur_data/4-19 1.00 predictions_arthur_cross/arthur_on_arthur_4-19.txt
python bmsPredict.py stephen_data/4-19 1.00 predictions_arthur_cross/stephen_on_arthur_4-19.txt
python bmsPredict.py arun_data/4-19 1.00 predictions_arthur_cross/arun_on_arthur_4-19.txt
python bmsPredict.py walter_data/4-19 1.00 predictions_arthur_cross/walter_on_arthur_4-19.txt

python bmsTrain.py stephen_data/4-19 0 0.00 1.00
python bmsTrain.py arun_data/4-19 0 0.00 1.00
python bmsTrain.py walter_data/4-19 0 0.00 1.00
python bmsTrain.py arthur_data/4-19 1 0.00 1.00
python bmsPredict.py arthur_data/4-20 1.00 predictions_arthur_cross/arthur_on_arthur_4-20.txt
python bmsPredict.py stephen_data/4-20 1.00 predictions_arthur_cross/stephen_on_arthur_4-20.txt
python bmsPredict.py arun_data/4-20 1.00 predictions_arthur_cross/arun_on_arthur_4-20.txt
python bmsPredict.py walter_data/4-20 1.00 predictions_arthur_cross/walter_on_arthur_4-20.txt

python bmsTrain.py stephen_data/4-20 0 0.00 1.00
python bmsTrain.py arun_data/4-20 0 0.00 1.00
python bmsTrain.py walter_data/4-20 0 0.00 1.00
python bmsTrain.py arthur_data/4-20 1 0.00 1.00
python bmsPredict.py arthur_data/4-21 1.00 predictions_arthur_cross/arthur_on_arthur_4-21.txt
python bmsPredict.py stephen_data/4-21 1.00 predictions_arthur_cross/stephen_on_arthur_4-21.txt
python bmsPredict.py arun_data/4-21 1.00 predictions_arthur_cross/arun_on_arthur_4-21.txt
python bmsPredict.py walter_data/4-21 1.00 predictions_arthur_cross/walter_on_arthur_4-21.txt

python bmsTrain.py stephen_data/4-21 0 0.00 1.00
python bmsTrain.py arun_data/4-21 0 0.00 1.00
python bmsTrain.py walter_data/4-21 0 0.00 1.00
python bmsTrain.py arthur_data/4-21 1 0.00 1.00
python bmsPredict.py arthur_data/4-22 1.00 predictions_arthur_cross/arthur_on_arthur_4-22.txt
python bmsPredict.py stephen_data/4-22 1.00 predictions_arthur_cross/stephen_on_arthur_4-22.txt
python bmsPredict.py walter_data/4-22 1.00 predictions_arthur_cross/walter_on_arthur_4-22.txt
