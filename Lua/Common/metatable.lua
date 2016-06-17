http://lua-users.org/wiki/MetamethodsTutorial
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
