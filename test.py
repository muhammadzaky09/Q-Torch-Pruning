from brevitas.nn import QuantConv2d

# Create a simple quantized conv layer
conv = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                   weight_bit_width=8, weight_scaling_per_output_channel=True)

# Examine the structure
print("Main attributes:", [attr for attr in dir(conv) if not attr.startswith('_')])
print("\nQuantization attributes:")
if hasattr(conv, 'weight_quant'):
    print("- weight_quant attributes:", [attr for attr in dir(conv.weight_quant) if not attr.startswith('_')])

# Check scaling implementation
if hasattr(conv, 'weight_scaling_impl'):
    print("\nWeight scaling implementation:")
    print("- Type:", type(conv.weight_scaling_impl))
    print("- Attributes:", [attr for attr in dir(conv.weight_scaling_impl) if not attr.startswith('_')])