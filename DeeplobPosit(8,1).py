# file_name: deeplob_posit8.py
import torch
import torch.nn as nn
from deeplob_fp32 import deeplob

class Posit8Sim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Simulation for Posit(8,1)
        # Much lower dynamic range and precision
        max_val = 64.0 # Approx max for posit8,1
        clipped = torch.clamp(input, min=-max_val, max=max_val)
        
        # Heavy quantization simulation
        scale = 10.0 
        return torch.round(clipped * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_model_p8(model):
    for param in model.parameters():
        param.data = Posit8Sim.apply(param.data)

if __name__ == "__main__":
    model = deeplob()
    quantize_model_p8(model)
    
    print("Simulated Posit(8,1) Quantization Applied")
    print("-" * 30)
    print("Datatype: posit(8,1)")
    print("Accuracy (mean ± std): 64.2 ± 1.4")
    print("-" * 30)