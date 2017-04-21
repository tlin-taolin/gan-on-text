require "nngraph"
local params=require("./parse")
local model=require("./atten")
local Data=require("./data")

local open_train_file = io.open(params.test_file, "r")

Data:Initial(params)
while true do
  Data:read_train(open_train_file)
end
-- model:readModel()
-- model:train()
