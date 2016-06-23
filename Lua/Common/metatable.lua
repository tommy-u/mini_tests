--http://lua-users.org/wiki/MetamethodsTutorial

local func_example = setmetatable( {}, {__index = function(t,k)
						     return "key doesn't exist"
						  end})

local fallback_tbl = setmetatable({
				     foo = "bar",
				     [123] = 456,
				  }, {__index=func_example})

local fallback_example = setmetatable({}, {__index=fallback_tbl})

-- print(func_example[1])
-- print(fallback_example.foo)
-- print(fallback_example[123])
-- print(fallback_example[456])


local t = {}

local m = setmetatable({}, {__newindex = function (table, key, value)
					    t[key] = value
					 end})

m[123] = 456
print(m[123]) --> nil
print(t[123]) --> 456

--https://github.com/torch/demos/blob/master/train-a-digit-classifier/dataset-mnist.lua

local labelvector = torch.zeros(10)

local mt = 
{ 
__index = function(self, index)
	     local input = self.data[index]
	     local class = self.labels[index]
	     local label = labelvector:zero()
	     label[class] = 1 -- onehot
	     return {input, label}
	  end
}

mydat = {}
mydat.data = {1,2,3}
mydat.labels = {4,5,6}

setmetatable(mydat, mt)

print(mydat[2])
print(mydat[2][1])
print(mydat[2][2])