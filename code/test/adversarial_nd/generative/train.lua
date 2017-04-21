require "nngraph"
local params=require("./parse")
local gan=require("./model")

gan:Initial(params)
gan:train()
