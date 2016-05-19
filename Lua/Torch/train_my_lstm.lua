--My attempt at a LSTM. It's not optimized at all.
LSTM = require 'my_lstm.lua'
num_input = 1
num_hid   = 5

function test(model)
   print("testing")
   a = torch.Tensor(num_input)
   --Initial cell and hid state. Not tuned, just guesses.
   local out = {}
   out[1] = torch.Tensor(num_hid):fill(-1.0)
   out[2] = torch.Tensor(num_hid):fill(-0.1)
   for i= -1,1,.2 do
      a = a:fill(i)
      print("[" .. i .. "] -> ")
      out = model:forward({ a, out[1], out[2] })
      print(out[2][1])
      print()
   end
end

function train()

end


--Create the model
local model = LSTM.create(num_input, num_hid)
local criterion = nn.MSECriterion()
local learningRate = .003

--This is the phony "first" output state.
local out = {}
out[1] = torch.randn(num_hid)
out[2] = torch.randn(num_hid)

--Before training.
test(model)

--Attempt to learn the identity function for randomly chosen values
for i=1,10000 do
   rand_dec = 2 * torch.uniform() - 1
   x = torch.Tensor(num_input):fill(rand_dec)
   y = torch.Tensor(num_hid):fill(rand_dec)
   out = LSTM.train(model, { x, out[1], out[2]}, y, criterion, learningRate, i)
end


--After training.
test(model)

