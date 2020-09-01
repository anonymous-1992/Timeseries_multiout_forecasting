python lib/multiout_forecast_rnn.py --name LSTM --save results/LSTM.txt
python lib/multiout_forecast_rnn.py --name BiLSTM --save results/BiLSTM.txt
python lib/multiout_forecast_rnn.py --name EdLSTM --save results/EdLSTM.txt
python lib/multiout_forecast_rnn.py --name BiEdLSTM --save results/BiEdLSTM.txt
python lib/multiout_forecast_rnn.py --name CNN --save results/CNN.txt
python lib/multiout_forecast_rnn.py --name GRU --save results/GRU.txt
python lib/multiout_forecast_rnn.py --name BiGRU --save results/BiGRU.txt

python lib/multiout_forecast_reg.py --name LR --save results/LR.txt
python lib/multiout_forecast_reg.py --name SVR --save results/SVR.txt
python lib/multiout_forecast_reg.py --name Lasso --save results/Lasso.txt
python lib/multiout_forecast_reg.py --name GP --save results/GP.txt

python lib/multiout_forecast_stats.py --name ARIMA --save results/ARIMA.txt
python lib/multiout_forecast_stats.py --name VAR --save results/VAR.txt
