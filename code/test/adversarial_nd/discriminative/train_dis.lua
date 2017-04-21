require "nngraph"
local params=require("./dis_parse")
local model=require("./dis_model");

model:Initial(params)
model:train()
-- model:readModel()
