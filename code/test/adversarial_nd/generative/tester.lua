require "nngraph"
local params=require("./parse")
local model=require("./generative")

model:Initial(params)
model:train()
