import vectorquant.core.backend as backend
import math
import cmath

def test_fft():
    b = backend.get_backend()
    print(f"Using backend: {type(b).__name__}")
    
    # Signel: 1.0, 0.0, 1.0, 0.0
    input_data = [1.0, 0.0, 1.0, 0.0]
    result = b.radix2_fft(input_data)
    
    print(f"Input: {input_data}")
    print(f"Result: {result}")
    
    # Expected DFT of [1, 0, 1, 0] is [2, 0, 2, 0]
    expected = [complex(2, 0), complex(0, 0), complex(2, 0), complex(0, 0)]
    
    for i in range(len(result)):
        assert abs(result[i] - expected[i]) < 1e-4
    print("FFT Test Passed!")

if __name__ == "__main__":
    test_fft()
