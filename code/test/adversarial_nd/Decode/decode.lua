-- require "fbtorch"
-- require "cunn"
-- require "cutorch"
require "nngraph"
local params=require("./decode_parse")
-- cutorch.setDevice(params.gpu_index)
local decode_model=require("./decode_model")
decode_model:Initial(params)
decode_model.mode="test"
--decode_model:test()
decode_model:decode()
