local mymod = {}

local localval = "invisible to script"

sharedval = "available to script if required"
mymod.val = "passed as element in module table"

function hiddenfun()
   print("only available within module")
end

function mymod.sharedfun()
   print("can be called by module")
end

function mymod.rettable()
   print("returns a value to script")
   return {"a", "b", "c"}, {1, 2, 3}, { {1, "a"}, {"z", 2}, {"ac", "dc"} }
end


return mymod
