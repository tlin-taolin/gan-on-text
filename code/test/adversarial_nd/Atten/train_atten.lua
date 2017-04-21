require "nngraph"
local params=require("./parse")
local model=require("./atten");

print("Batches number in training: "..10000/params.batch_size)
print("Batches number in testing: "..2000/params.batch_size)

model:Initial(params)
-- model:readModel()
model:train()
