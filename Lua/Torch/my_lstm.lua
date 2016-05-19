--My version of an LSTM. Draws inspiration from the following:
--https://apaszke.github.io/lstm-explained.html
--http://colah.github.io/posts/2015-08-Understanding-LSTMs/
require 'nn'
require 'nngraph'

local LSTM = {}

function LSTM.create(input_size, rnn_size)
   --Bogus start state at first. Later updated with real persistant state.
--   local c_storage = torch.randn(rnn_size)
--   local h_storage = torch.randn(rnn_size)
      
   --Pass through input layers.
   --Input at time t.
   local input = nn.Identity()()
   --Cell state at time t-1.
   local prev_c = nn.Identity()()
   --Output at time t-1.
   local prev_h = nn.Identity()()
   --Package as table.
   local inputs = {input, prev_c, prev_h}

   --Sometimes people run input and prev through xforms separately and add.
   --I'm simply doing a concatenation of last hidden state (output) and input.
   local cat = nn.JoinTable(1)({input, prev_h})

   local size_cat = input_size + rnn_size

   ---------- The gates ---------- 
   --These can be combined for speed, but the goal here is clarity. 
   --Forget
   local f = nn.Sigmoid()(nn.Linear(size_cat, rnn_size)(cat))
   --Input
   local i = nn.Sigmoid()(nn.Linear(size_cat, rnn_size)(cat))
   --Output
   local o = nn.Sigmoid()(nn.Linear(size_cat, rnn_size)(cat))

   ---------- Data ---------- 
   --Pass the concatenation through linear transform then squish.
   local xs = nn.Tanh()(nn.Linear(size_cat, rnn_size)(cat))
   --Let gate decide what part gets through.
   local xs = nn.CMulTable()({i, xs})

   ---------- Cell Memory ---------- 
   --Remove anything we want to forget.
   local c = nn.CMulTable()({prev_c, f})
   --Add anything we want from the current data.
   local c = nn.CAddTable()({c, xs})

      ---------- Output State ---------- 
   --Squish cell state.
   local h = nn.Tanh()(c)
   --Only pass forward what the gate desires.
   local h = nn.CMulTable()({h, o})
   
   --Put the hidden state through a linear layer to 
   local outputs = {c, h}
   
   --Package up into module 
   gmod = nn.gModule( inputs, outputs ) 

   return gmod
   
end

function LSTM.train( model, x, y, criterion, learningRate, i)

   
   --Forward execution, produces { c, h}
   local prediction = model:forward(x)
   --Determine error signal
   local err = criterion:forward(prediction[2], y)
--   if i % 100 == 1 then print('error for iteration ' .. i  .. ' is ' .. err) end
   local gradOutputs = criterion:backward(prediction[2], y)
   model:backward( x ,  { gradOutputs, gradOutputs}  ) --best of the 3 for learning identity
   --   model:backward( x ,  { nil, gradOutputs}  ) --trains but not as well
   --   model:backward( x ,  { gradOutputs, nil}  ) --error 1 gradOutputs instead of 2
   model:updateParameters(learningRate)
   model:zeroGradParameters()
   return prediction
end

return LSTM  

