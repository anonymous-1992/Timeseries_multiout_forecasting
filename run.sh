python lib/multiout_forecast_rnn.py --name LSTM --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name BiLSTM --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name EdLSTM --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name BiEdLSTM --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name CNN --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name GRU --save results.txt --horizon 8
python lib/multiout_forecast_rnn.py --name BiGRU --save results.txt --horizon 8

python lib/multiout_forecast_reg.py --name LR --save results.txt --horizon 8
python lib/multiout_forecast_reg.py --name SVR --save results.txt --horizon 8
python lib/multiout_forecast_reg.py --name Lasso --save results.txt --horizon 8
python lib/multiout_forecast_reg.py --name GP --save results.txt --horizon 8

python lib/multiout_forecast_com.py --save results.txt --horizon 8


