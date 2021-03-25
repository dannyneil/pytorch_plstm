# Port of Phased LSTM to Pytorch

This is a quick port of Phased LSTM to Pytorch.  Original is here:
https://github.com/dannyneil/public_plstm

## Rule of Thumb
In general, if you are using ~1000 timesteps or more in your input sequence, or you have asynchronously sampled data, you can benefit from PLSTM.

![Performance](/comparison.png)

# To run

To run, simply execute:
```bash
python main.py
```
This will train an LSTM and Phased LSTM on the asynchronous sin wave classification task.

# Requirements
Requires Pytorch.  Currently tested on torch==1.8.0.
