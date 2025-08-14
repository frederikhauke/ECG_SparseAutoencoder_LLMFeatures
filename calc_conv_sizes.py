#!/usr/bin/env python3
"""
Calculate the exact ConvTranspose1d parameters needed for SimpleAutoencoder.
"""

def calculate_conv_transpose_output(input_size, kernel_size, stride, padding, output_padding=0):
    """Calculate output size for ConvTranspose1d."""
    return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

# Forward path (encoder):
# 2700 -> 675 (Conv1d: kernel=31, stride=4, padding=15)
# 675 -> 225 (Conv1d: kernel=15, stride=3, padding=7)
# 225 -> 75 (Conv1d: kernel=9, stride=3, padding=4)
# 75 -> 25 (Conv1d: kernel=5, stride=3, padding=2)

print("Forward path calculations:")
size = 2700
print(f"Start: {size}")
size = (size + 2*15 - 31) // 4 + 1
print(f"After conv1: {size}")
size = (size + 2*7 - 15) // 3 + 1
print(f"After conv2: {size}")
size = (size + 2*4 - 9) // 3 + 1
print(f"After conv3: {size}")
size = (size + 2*2 - 5) // 3 + 1
print(f"After conv4: {size}")

print("\nReverse path (decoder) with corrected output_padding:")
# Reverse path: 25 -> 75 -> 225 -> 675 -> 2700
size = 25
print(f"Start: {size}")

# 25 -> 75 (ConvTranspose1d: kernel=5, stride=3, padding=2, output_padding=2)
size = calculate_conv_transpose_output(size, 5, 3, 2, 2)
print(f"After deconv1: {size}")

# 75 -> 225 (ConvTranspose1d: kernel=9, stride=3, padding=4, output_padding=2)
size = calculate_conv_transpose_output(size, 9, 3, 4, 2)
print(f"After deconv2: {size}")

# 225 -> 675 (ConvTranspose1d: kernel=15, stride=3, padding=7, output_padding=2)
size = calculate_conv_transpose_output(size, 15, 3, 7, 2)
print(f"After deconv3: {size}")

# 675 -> 2700 (ConvTranspose1d: kernel=31, stride=4, padding=15, output_padding=3)
size = calculate_conv_transpose_output(size, 31, 4, 15, 3)
print(f"After deconv4: {size}")

if size == 2700:
    print("✅ Final decoder output matches target size of 2700!")
else:
    print(f"❌ Final size {size} doesn't match target 2700")
