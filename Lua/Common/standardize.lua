--Colwise standardize t
local t = torch.Tensor( { { 1, 2, 3 } , { 4, 5, 6 } , { 9, 8, 7 } } )

local m = t:mean(1)

local s = t:std(1)

local c = t - m:expand(3,3)

local z = torch.cdiv(c, s:expand(3,3))

print(z)